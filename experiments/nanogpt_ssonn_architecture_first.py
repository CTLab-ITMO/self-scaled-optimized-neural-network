"""Single-GPU FineWeb-Edu NanoGPT/SSONN architecture-first runner.

Examples:
    uv run python experiments/nanogpt_ssonn_architecture_first.py --gpu 6 --data-dir data/finewebedu10B --ssonn-learning-rate 3e-5
    nohup uv run python experiments/nanogpt_ssonn_architecture_first.py --gpu 6 > architecture_first_h200.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H200 single-GPU architecture-first training for residual SSONN NanoGPT.",
    )
    parser.add_argument("--gpu", default="6", help="Physical GPU id for CUDA_VISIBLE_DEVICES. Use empty string to keep the environment unchanged.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "finewebedu10B")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--micro-batch-size", type=int, default=16, help="Sequences per microbatch at the maximum context window; shorter windows scale this up.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Deprecated legacy value retained in checkpoint metadata; architecture-first training derives accumulation from tokens.")
    parser.add_argument("--target-tokens-per-update", type=int, default=524_288)
    parser.add_argument("--cooldown-fraction", type=float, default=640 / 1750)
    parser.add_argument("--cooldown-updates", type=int, default=None, help="Deprecated explicit override; prefer --cooldown-fraction.")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=None, help="Deprecated periodic-validation override; prefer --periodic-validation-tokens.")
    parser.add_argument("--periodic-validation-tokens", type=int, default=1_048_576)
    parser.add_argument("--final-validation-tokens", type=int, default=10_485_760)
    parser.add_argument("--validation-seq-len", type=int, default=1792)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--download-pretokenized", action="store_true")
    parser.add_argument("--pretokenized-train-shards", type=int, default=99)
    parser.add_argument("--ssonn-eval-batches", type=int, default=None)
    parser.add_argument("--finder-batches", type=int, default=32)
    parser.add_argument("--ssonn-learning-rate", type=float, default=1e-6)
    parser.add_argument("--architecture-fraction", type=float, default=0.05)
    parser.add_argument("--architecture-expansions", type=int, default=4)
    parser.add_argument("--architecture-cycle-steps", type=int, default=0, help="Updates per architecture cycle. Zero derives one pass over the architecture subset.")
    parser.add_argument("--training-passes", type=int, default=0, help="Full-dataset passes. Overrides --full-training-tokens when positive.")
    parser.add_argument("--full-training-tokens", type=int, default=917_504_000)
    parser.add_argument("--full-training-steps", type=int, default=0, help="Optional full-training update-count override for smoke/dry runs. Zero uses the token budget.")
    parser.add_argument("--ssonn-expansion-threshold", type=float, default=0.1)
    parser.add_argument("--ssonn-max-new-edges", type=int, default=1024)
    parser.add_argument("--backbone-grad-clip", type=float, default=1.0)
    parser.add_argument("--ssonn-grad-clip", type=float, default=1.0)
    parser.add_argument("--attention-backend", choices=("flex", "sdpa"), default="flex")
    parser.add_argument("--smoke-test", action="store_true", help="Run CPU model/schedule smoke checks and exit.")
    return parser.parse_args()


ARGS = parse_args()
if ARGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None

import ssonn.train.train as ssonn_train
from ssonn.metrics.edge_finder import EdgeFinder
from ssonn.metrics.nonlinearity_metrics import AbsGradientEdgeMetric
from ssonn.model.model import ScaledNet


DATA_DIR = ARGS.data_dir.resolve()
CHECKPOINT_DIR = ARGS.checkpoint_dir.resolve()
RUN_MODE = "architecture_first"
SEED = ARGS.seed
USE_COMPILE = ARGS.compile
TRAIN_STREAM_SEED = SEED
EVAL_STREAM_SEED = SEED + 1
FINDER_STREAM_SEED = SEED + 2

MODE_ID = {"pretrain": 0, "ssonn": 1, "mlp": 2, "architecture_first": 3}
MLP_KIND = "ssonn"
INIT_MODE = "scratch"
FIXED_CYCLE_TOKENS = True
FULL_REINIT_AFTER_EXPANSION = False

DOWNLOAD_PRETOKENIZED_DATA = ARGS.download_pretokenized
PRETOKENIZED_TRAIN_SHARDS = ARGS.pretokenized_train_shards
TRAIN_FILES: list[str] = []
VAL_FILES: list[str] = []
ARCHITECTURE_SLICES: list[ShardSlice] = []
FULL_TRAINING_SLICES: list[ShardSlice] = []
VALIDATION_SLICES: list[ShardSlice] = []
TRAIN_TOKEN_COUNT = 0
SSONN_CYCLE_TOKEN_COUNT = 0
ARCHITECTURE_TOKEN_COUNT = 0
FULL_TRAINING_TOKEN_COUNT = 0
FULL_TRAINING_PASS_STEPS = 0


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    max_seq_len: int = 1792


MODEL_CONFIG = GPTConfig()
NUM_UPDATES = 0
MICRO_BATCH_SIZE = ARGS.micro_batch_size
GRAD_ACCUM_STEPS = ARGS.grad_accum_steps
TARGET_TOKENS_PER_UPDATE = ARGS.target_tokens_per_update
WINDOW_START = 256
WINDOW_END = 1792
COOLDOWN_UPDATES = ARGS.cooldown_updates
COOLDOWN_FRACTION = ARGS.cooldown_fraction
EVAL_INTERVAL = ARGS.eval_interval
EVAL_BATCHES = ARGS.eval_batches
PERIODIC_VALIDATION_TOKENS = ARGS.periodic_validation_tokens
FINAL_VALIDATION_TOKENS = ARGS.final_validation_tokens
VALIDATION_SEQ_LEN = ARGS.validation_seq_len
CHECKPOINT_INTERVAL = ARGS.checkpoint_interval
PRETRAIN_MUON_LR = 0.02
PRETRAIN_ADAM_LR = 3e-4
SSONN_EVAL_BATCHES = ARGS.ssonn_eval_batches or EVAL_BATCHES
FINDER_BATCHES = ARGS.finder_batches
SSONN_LEARNING_RATE = ARGS.ssonn_learning_rate
SSONN_CYCLE_STEPS = ARGS.architecture_cycle_steps
ARCHITECTURE_FRACTION = ARGS.architecture_fraction
ARCHITECTURE_EXPANSIONS = ARGS.architecture_expansions
TRAINING_PASSES = ARGS.training_passes
FULL_TRAINING_TARGET_TOKENS = ARGS.full_training_tokens
FULL_TRAINING_STEPS = ARGS.full_training_steps
SSONN_EXPANSION_THRESHOLD = ARGS.ssonn_expansion_threshold
SSONN_MAX_NEW_EDGES = ARGS.ssonn_max_new_edges
BACKBONE_GRAD_CLIP = ARGS.backbone_grad_clip
SSONN_GRAD_CLIP = ARGS.ssonn_grad_clip
ATTENTION_BACKEND = ARGS.attention_backend
EOT_TOKEN_ID = 50256
REINIT_MODE = "full"

BASE_CHECKPOINT = CHECKPOINT_DIR / "finewebedu_attention_only.pt"
SSONN_CHECKPOINT = CHECKPOINT_DIR / "finewebedu_ssonn_architecture_first_best.pt"
MLP_CHECKPOINT = CHECKPOINT_DIR / "finewebedu_speedrun_mlp_best.pt"

assert MODEL_CONFIG.n_embd % MODEL_CONFIG.n_head == 0
assert WINDOW_START <= WINDOW_END <= MODEL_CONFIG.max_seq_len
assert MICRO_BATCH_SIZE >= 1
assert GRAD_ACCUM_STEPS >= 1
assert TARGET_TOKENS_PER_UPDATE >= WINDOW_START
assert 0.0 < COOLDOWN_FRACTION <= 1.0
assert PERIODIC_VALIDATION_TOKENS >= VALIDATION_SEQ_LEN
assert FINAL_VALIDATION_TOKENS >= PERIODIC_VALIDATION_TOKENS
assert WINDOW_START <= VALIDATION_SEQ_LEN <= WINDOW_END
assert 0.0 < ARCHITECTURE_FRACTION <= 1.0
assert ARCHITECTURE_EXPANSIONS >= 1
assert TRAINING_PASSES >= 0
assert FULL_TRAINING_TARGET_TOKENS > 0
assert BACKBONE_GRAD_CLIP >= 0.0
assert SSONN_GRAD_CLIP >= 0.0


def amp_dtype() -> torch.dtype | None:
    if ARGS.amp_dtype == "bf16":
        return torch.bfloat16
    if ARGS.amp_dtype == "fp16":
        return torch.float16
    return None


AMP_DTYPE = amp_dtype()
USE_AMP = AMP_DTYPE is not None
USE_SCALER = ARGS.amp_dtype == "fp16"

device = torch.device("cuda" if torch.cuda.is_available() and not ARGS.smoke_test else "cpu")
USE_FLEX_ATTENTION = ATTENTION_BACKEND == "flex" and device.type == "cuda"
if USE_FLEX_ATTENTION and (create_block_mask is None or flex_attention is None):
    raise RuntimeError("--attention-backend flex requires torch.nn.attention.flex_attention.")
if USE_FLEX_ATTENTION:
    USE_COMPILE = True
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(
        "visible cuda device:",
        torch.cuda.get_device_name(0),
        "|",
        torch.cuda.get_device_properties(0).total_memory // 2**30,
        "GiB",
        "| CUDA_VISIBLE_DEVICES=",
        os.environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
    )
elif not ARGS.smoke_test:
    raise RuntimeError("CUDA is required for training. Use --smoke-test for CPU-only checks.")
elif ATTENTION_BACKEND == "flex":
    print("CPU smoke test: using SDPA document masks instead of FlexAttention.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_pretokenized_fineweb_edu(train_shards: int) -> None:
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    repo_id = "kjj0/finewebedu10B-gpt2"
    filenames = ["finewebedu_val_000000.bin"]
    filenames.extend(f"finewebedu_train_{index:06d}.bin" for index in range(1, train_shards + 1))
    for filename in filenames:
        print("Downloading", filename)
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=DATA_DIR)


def find_shards(data_dir: Path) -> tuple[list[str], list[str]]:
    train = sorted({str(path) for path in data_dir.rglob("*fineweb*edu*train_*.bin")})
    val = sorted({str(path) for path in data_dir.rglob("*fineweb*edu*val_*.bin")})
    return train, val


def shard_token_count(path: str) -> int:
    header = np.memmap(path, dtype=np.int32, mode="r", shape=(256,))
    if int(header[0]) != 20240520:
        raise ValueError(f"Unexpected FineWeb-Edu shard header in {path}")
    return int(header[2])


def read_shard(path: str) -> np.ndarray:
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    header = np.memmap(path, dtype=np.int32, mode="r", shape=(256,))
    if int(header[0]) != 20240520:
        raise ValueError(f"Unexpected FineWeb-Edu shard header in {path}")
    token_count = int(header[2])
    return raw[512 : 512 + token_count]


@dataclass(frozen=True)
class ShardSlice:
    path: str
    start: int
    token_count: int


def allocate_balanced_slices(files: Iterable[str], token_budget: int, seed: int) -> list[ShardSlice]:
    """Allocate an exact token budget proportionally across every shard."""
    paths = sorted(files)
    counts = [shard_token_count(path) for path in paths]
    available = sum(counts)
    if token_budget <= 0 or token_budget > available:
        raise ValueError(f"Token budget must be in [1, {available}], got {token_budget}.")

    raw_quotas = [token_budget * count / available for count in counts]
    quotas = [math.floor(quota) for quota in raw_quotas]
    remainder = token_budget - sum(quotas)
    order = sorted(range(len(paths)), key=lambda index: (raw_quotas[index] - quotas[index], -index), reverse=True)
    for index in order[:remainder]:
        quotas[index] += 1

    rng = random.Random(seed)
    slices = []
    for path, available_tokens, quota in zip(paths, counts, quotas, strict=True):
        if quota == 0:
            continue
        max_start = available_tokens - quota
        start = rng.randrange(max_start + 1) if max_start > 0 else 0
        slices.append(ShardSlice(path=path, start=start, token_count=quota))
    if sum(item.token_count for item in slices) != token_budget:
        raise AssertionError("Balanced shard allocation did not preserve the requested token budget.")
    return slices


class TokenShardStream:
    def __init__(
        self,
        files: Iterable[str] | None = None,
        seed: int = 0,
        token_limit: int | None = None,
        slices: Sequence[ShardSlice] | None = None,
        interleave: bool = True,
    ):
        if slices is not None:
            selected = list(slices)
        else:
            all_files = sorted(files or [])
            if token_limit is None:
                selected = [ShardSlice(path, 0, shard_token_count(path)) for path in all_files]
            else:
                selected = allocate_balanced_slices(all_files, token_limit, seed)
        if not selected:
            raise FileNotFoundError("TokenShardStream received no shard files.")
        self.slices = selected
        self.files = [item.path for item in selected]
        self.file_token_limits = [item.token_count for item in selected]
        self.token_count = sum(self.file_token_limits)
        self.interleave = interleave
        self.shard_index = 0
        self.positions = [item.start for item in selected]
        self.ends = [item.start + item.token_count for item in selected]
        self.epoch = 0
        self.tokens = read_shard(self.files[self.shard_index])

    def _advance(self) -> None:
        self.shard_index = (self.shard_index + 1) % len(self.files)
        self.tokens = read_shard(self.files[self.shard_index])

    def _restart(self) -> None:
        self.positions = [item.start for item in self.slices]
        self.shard_index = 0
        self.epoch += 1
        self.tokens = read_shard(self.files[0])

    def _next_sequence(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        required = seq_len + 1
        for _ in range(len(self.files)):
            position = self.positions[self.shard_index]
            if position + required <= self.ends[self.shard_index]:
                block = torch.from_numpy(np.array(self.tokens[position : position + required], dtype=np.int64))
                self.positions[self.shard_index] += seq_len
                if self.interleave:
                    self._advance()
                return block[:-1], block[1:]
            self._advance()
        self._restart()
        return self._next_sequence(seq_len)

    def next_batch(self, seq_len: int, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*(self._next_sequence(seq_len) for _ in range(batch_size)), strict=True)
        x = torch.stack(xs)
        y = torch.stack(ys)
        if device.type == "cuda":
            x = x.pin_memory()
            y = y.pin_memory()
        return x, y


class FixedFinderLoader:
    def __init__(self, slices: Sequence[ShardSlice], batches: int, seq_len: int, seed: int):
        self.slices = list(slices)
        self.batches = batches
        self.seq_len = seq_len
        self.seed = seed

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        stream = TokenShardStream(seed=self.seed, slices=self.slices)
        for _ in range(self.batches):
            yield stream.next_batch(self.seq_len, MICRO_BATCH_SIZE)


def to_device(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(-2)
        if positions is None:
            cos = self.cos[:seq_len].to(dtype=x.dtype, device=x.device)[None, None]
            sin = self.sin[:seq_len].to(dtype=x.dtype, device=x.device)[None, None]
        else:
            cos = self.cos.to(device=x.device)[positions].to(dtype=x.dtype)[:, None]
            sin = self.sin.to(device=x.device)[positions].to(dtype=x.dtype)[:, None]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rotated.flatten(-2)


def document_ids_and_positions(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Treat EOT as the first token of a document and reset its rotary position."""
    batch, seq_len = tokens.shape
    starts = tokens.eq(EOT_TOKEN_ID)
    starts[:, 0] = True
    document_ids = starts.long().cumsum(dim=1) - 1
    indices = torch.arange(seq_len, device=tokens.device).expand(batch, -1)
    start_indices = torch.where(starts, indices, torch.zeros_like(indices))
    latest_start = torch.cummax(start_indices, dim=1).values
    return document_ids, indices - latest_start


def make_document_attention_mask(document_ids: torch.Tensor):
    batch, seq_len = document_ids.shape
    if USE_FLEX_ATTENTION:
        def document_causal_mask(batch_index, _head, query_index, key_index):
            return (query_index >= key_index) & (
                document_ids[batch_index, query_index] == document_ids[batch_index, key_index]
            )

        return create_block_mask(
            document_causal_mask,
            B=batch,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=str(document_ids.device),
            _compile=True,
        )
    causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=document_ids.device).tril_()
    same_document = document_ids[:, :, None].eq(document_ids[:, None, :])
    return same_document[:, None] & causal[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.q_proj.weight, std=0.5 * self.config.n_embd ** -0.5)
        nn.init.normal_(self.k_proj.weight, std=0.5 * self.config.n_embd ** -0.5)
        nn.init.normal_(self.v_proj.weight, std=0.5 * self.config.n_embd ** -0.5)
        nn.init.zeros_(self.o_proj.weight)
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()

    def forward(self, x: torch.Tensor, positions: torch.Tensor, attention_mask) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        q = self.rope(self.q_norm(q), positions)
        k = self.rope(self.k_norm(k), positions)
        k = k.expand(-1, self.n_head, -1, -1)
        v = v.expand(-1, self.n_head, -1, -1)
        if USE_FLEX_ATTENTION:
            y = flex_attention(q, k, v, block_mask=attention_mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, seq_len, -1))


class TokenSSONNMLP(nn.Module):
    def __init__(self, width: int, device_name: str):
        super().__init__()
        self.scaled = ScaledNet(torch.ones((width, width)), torch.zeros(width), device=device_name)
        self.scaled.dropout.p = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        main_mask = getattr(self.scaled, "sparsity_mask", None)
        if main_mask is not None:
            main_mask = main_mask.detach().clone()
        sub_masks = [
            getattr(layer, "sparsity_mask", None).detach().clone()
            if getattr(layer, "sparsity_mask", None) is not None else None
            for layer in self.scaled.sub_linears
        ]
        self.scaled.reinitialize_parameters()
        if main_mask is not None:
            self.scaled.sparsity_mask = main_mask
        for layer, mask in zip(self.scaled.sub_linears, sub_masks, strict=True):
            if mask is not None:
                layer.sparsity_mask = mask
        # The SSONN block is a residual delta. A zero output projection keeps
        # every full reinitialization functionally identical to the backbone.
        with torch.no_grad():
            self.scaled.weight.zero_()
            self.scaled.bias.zero_()
        self.scaled.dropout.p = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        return self.scaled(x.reshape(batch * seq_len, width)).reshape(batch, seq_len, width)


class SpeedrunMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, 4 * width, bias=False)
        self.c_proj = nn.Linear(4 * width, width, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.c_fc.weight, std=0.5 * self.width ** -0.5)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig, mlp_kind: str = "none"):
        super().__init__()
        if mlp_kind not in {"none", "ssonn", "speedrun_mlp"}:
            raise ValueError(f"Unsupported mlp_kind={mlp_kind!r}")
        self.mlp_kind = mlp_kind
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_mlp = RMSNorm(config.n_embd)
        if mlp_kind == "ssonn":
            self.mlp = TokenSSONNMLP(config.n_embd, str(device))
        elif mlp_kind == "speedrun_mlp":
            self.mlp = SpeedrunMLP(config.n_embd)
        else:
            self.mlp = None

    def reset_parameters(self) -> None:
        self.norm_attn.reset_parameters()
        self.attn.reset_parameters()
        self.norm_mlp.reset_parameters()
        if self.mlp is not None and hasattr(self.mlp, "reset_parameters"):
            self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, positions: torch.Tensor, attention_mask) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), positions, attention_mask)
        if self.mlp_kind == "ssonn":
            x = x + self.mlp(self.norm_mlp(x))
        elif self.mlp_kind == "speedrun_mlp":
            x = x + self.mlp(self.norm_mlp(x))
        return x


class AttentionOnlyGPT(nn.Module):
    def __init__(self, config: GPTConfig, mlp_kind: str = "none"):
        super().__init__()
        self.config = config
        self.mlp_kind = mlp_kind
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config, mlp_kind) for _ in range(config.n_layer)])
        self.norm_final = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.ssonn_layer_names: list[str] = []
        if mlp_kind == "ssonn":
            for idx, block in enumerate(self.blocks):
                name = f"ssonn_mlp_{idx}"
                setattr(self, name, block.mlp.scaled)
                self.ssonn_layer_names.append(name)

    def reset_parameters(self) -> None:
        self.wte.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.norm_final.reset_parameters()
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask=None,
    ) -> torch.Tensor:
        if positions is None or attention_mask is None:
            document_ids, positions = document_ids_and_positions(tokens)
            attention_mask = make_document_attention_mask(document_ids)
        x = self.wte(tokens)
        for block in self.blocks:
            x = block(x, positions, attention_mask)
        logits = self.lm_head(self.norm_final(x))
        return 30.0 * torch.tanh(logits / 30.0)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    return sum((parameter.detach() != 0).sum().item() for parameter in model.parameters())


def count_ssonn_sublinears(model: nn.Module) -> int:
    return sum(len(module.sub_linears) for module in model.modules() if isinstance(module, ScaledNet))


def count_ssonn_stored_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for module in model.modules() if isinstance(module, ScaledNet) for parameter in module.parameters())


def count_ssonn_active_connections(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if not isinstance(module, ScaledNet):
            continue
        main_mask = getattr(module, "sparsity_mask", None)
        total += int(main_mask.sum().item()) if main_mask is not None else module.weight.numel()
        for sub_layer in module.sub_linears:
            sub_mask = getattr(sub_layer, "sparsity_mask", None)
            total += int(sub_mask.sum().item()) if sub_mask is not None else sub_layer.weight.numel()
    return total


def assert_finite_model(model: nn.Module, label: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"{label}: non-finite parameter '{name}'.")


def assert_finite_tensor(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{label}: non-finite value.")


def checkpoint_payload(
    model: nn.Module,
    optimizer_state: dict | None,
    update: int,
    metadata: dict | None = None,
) -> dict:
    payload = {
        "checkpoint_schema_version": 2,
        "model": model.state_dict(),
        "model_config": asdict(MODEL_CONFIG),
        "mlp_kind": MLP_KIND,
        "update": update,
        "optimizer": optimizer_state,
        "recipe": {
            "source": "Deveraux-Parker/nanoGPT_1GPU_SPEEDRUN@135bc07555d47b64223c03e3da0bc6571efc6f2a",
            "stable_deviations": [
                "single GPU",
                "residual SSONN backbone is not the full modded NanoGPT architecture",
                f"{ARGS.amp_dtype} autocast",
            ],
            "num_updates": NUM_UPDATES,
            "window_start": WINDOW_START,
            "window_end": WINDOW_END,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "micro_batch_sequences": MICRO_BATCH_SIZE,
            "micro_batch_sequences_meaning": "maximum-window batch size",
            "activation_token_envelope": MICRO_BATCH_SIZE * WINDOW_END,
            "dynamic_microbatch": True,
            "dynamic_accumulation": True,
            "legacy_grad_accum_steps": GRAD_ACCUM_STEPS,
            "target_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
            "cooldown_fraction": COOLDOWN_FRACTION,
            "cooldown_updates_override": COOLDOWN_UPDATES,
            "attention_backend": "flex" if USE_FLEX_ATTENTION else "sdpa",
            "torch_compile": USE_COMPILE,
            "document_local_attention": True,
            "periodic_validation_tokens": PERIODIC_VALIDATION_TOKENS,
            "periodic_validation_scheduled_tokens": validation_scheduled_tokens(PERIODIC_VALIDATION_TOKENS),
            "final_validation_tokens": FINAL_VALIDATION_TOKENS,
            "final_validation_scheduled_tokens": validation_scheduled_tokens(FINAL_VALIDATION_TOKENS),
            "validation_seq_len": VALIDATION_SEQ_LEN,
            "train_shards": len(TRAIN_FILES),
            "train_tokens": TRAIN_TOKEN_COUNT,
            "ssonn_cycle_steps": SSONN_CYCLE_STEPS,
            "ssonn_cycle_tokens": SSONN_CYCLE_TOKEN_COUNT,
            "architecture_fraction": ARCHITECTURE_FRACTION,
            "architecture_expansions": ARCHITECTURE_EXPANSIONS,
            "architecture_tokens": ARCHITECTURE_TOKEN_COUNT,
            "architecture_search_total_tokens": ARCHITECTURE_EXPANSIONS * SSONN_CYCLE_TOKEN_COUNT,
            "architecture_cycle_steps": SSONN_CYCLE_STEPS,
            "training_passes": TRAINING_PASSES,
            "full_training_steps": FULL_TRAINING_STEPS,
            "full_training_pass_steps": FULL_TRAINING_PASS_STEPS,
            "full_training_tokens": FULL_TRAINING_TOKEN_COUNT,
            "full_training_target_tokens": FULL_TRAINING_TARGET_TOKENS,
        },
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if metadata:
        payload["architecture_first"] = metadata
    return payload


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer_state: dict | None,
    update: int,
    metadata: dict | None = None,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    assert_finite_model(model, f"Refusing to save {path.name}")
    torch.save(checkpoint_payload(model, optimizer_state, update, metadata), path)
    print("saved", path)


@torch.no_grad()
def evaluate(model: nn.Module, stream: TokenShardStream, seq_len: int, batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x_cpu, y_cpu = stream.next_batch(seq_len, MICRO_BATCH_SIZE)
        x, y = to_device(x_cpu, y_cpu)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP and device.type == "cuda"):
            document_ids, positions = document_ids_and_positions(x)
            logits = model(x, positions, make_document_attention_mask(document_ids))
            assert_finite_tensor(logits, "Evaluation logits")
            loss = lm_cross_entropy(logits, y)
        assert_finite_tensor(loss, "Evaluation loss")
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate_fixed(model: nn.Module, seq_len: int, batches: int) -> float:
    return evaluate(model, TokenShardStream(VAL_FILES, EVAL_STREAM_SEED), seq_len, batches)


def validation_scheduled_tokens(token_budget: int) -> int:
    return math.ceil(token_budget / VALIDATION_SEQ_LEN) * VALIDATION_SEQ_LEN


@torch.no_grad()
def evaluate_token_budget(
    model: nn.Module,
    token_budget: int,
    slices: Sequence[ShardSlice] | None = None,
) -> float:
    """Evaluate a deterministic sample at one fixed context length."""
    selected_slices = list(VALIDATION_SLICES if slices is None else slices)
    stream = TokenShardStream(seed=EVAL_STREAM_SEED, slices=selected_slices)
    model.eval()
    weighted_loss = 0.0
    evaluated_tokens = 0
    target_sequences = max(1, math.ceil(token_budget / VALIDATION_SEQ_LEN))
    while target_sequences > 0:
        batch_size = min(MICRO_BATCH_SIZE, target_sequences)
        x_cpu, y_cpu = stream.next_batch(VALIDATION_SEQ_LEN, batch_size)
        x, y = to_device(x_cpu, y_cpu)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP and device.type == "cuda"):
            document_ids, positions = document_ids_and_positions(x)
            logits = model(x, positions, make_document_attention_mask(document_ids))
            assert_finite_tensor(logits, "Evaluation logits")
            loss = lm_cross_entropy(logits, y)
        assert_finite_tensor(loss, "Evaluation loss")
        batch_tokens = batch_size * VALIDATION_SEQ_LEN
        weighted_loss += loss.item() * batch_tokens
        evaluated_tokens += batch_tokens
        target_sequences -= batch_size
    return weighted_loss / evaluated_tokens


class Muon(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, steps: int = 5):
        params = list(params)
        if not params:
            raise ValueError("Muon received no parameters.")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, steps=steps))

    @staticmethod
    def orthogonalize(gradient: torch.Tensor, steps: int) -> torch.Tensor:
        x = gradient.float()
        x = x / (x.norm() + 1e-7)
        transposed = x.size(0) > x.size(1)
        if transposed:
            x = x.t()
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            a_mat = x @ x.t()
            x = a * x + (b * a_mat + c * a_mat @ a_mat) @ x
        if transposed:
            x = x.t()
        return x.to(dtype=gradient.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter.grad)
                momentum = state["momentum"]
                momentum.mul_(group["momentum"]).add_(parameter.grad)
                update = parameter.grad.add(momentum, alpha=group["momentum"]) if group["nesterov"] else momentum
                update = self.orthogonalize(update, group["steps"])
                update.mul_(max(1.0, update.size(0) / update.size(1)) ** 0.5)
                parameter.add_(update, alpha=-group["lr"])


def make_pretrain_optimizers(model: nn.Module):
    matrix, adam = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2 and name not in {"wte.weight", "lm_head.weight"}:
            matrix.append(parameter)
        else:
            adam.append(parameter)
    return (
        Muon(matrix, lr=PRETRAIN_MUON_LR, momentum=0.85),
        torch.optim.AdamW(adam, lr=PRETRAIN_ADAM_LR, betas=(0.8, 0.95), weight_decay=0.0),
    )


def ssonn_parameter_ids(model: nn.Module) -> set[int]:
    return {
        id(parameter)
        for module in model.modules()
        if isinstance(module, ScaledNet)
        for parameter in module.parameters()
    }


def make_ssonn_stage_optimizers(model: nn.Module):
    ssonn_ids = ssonn_parameter_ids(model)
    muon_parameters, backbone_adam_parameters, ssonn_parameters = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in ssonn_ids:
            ssonn_parameters.append(parameter)
        elif parameter.ndim == 2 and name not in {"wte.weight", "lm_head.weight"}:
            muon_parameters.append(parameter)
        else:
            backbone_adam_parameters.append(parameter)
    return (
        Muon(muon_parameters, lr=PRETRAIN_MUON_LR, momentum=0.85),
        torch.optim.AdamW(backbone_adam_parameters, lr=PRETRAIN_ADAM_LR, betas=(0.8, 0.95), weight_decay=0.0),
        torch.optim.AdamW(ssonn_parameters, lr=0.0, weight_decay=0.0),
    )


def speedrun_window(step: int) -> int:
    fraction = step / max(NUM_UPDATES - 1, 1)
    raw = WINDOW_START + fraction * (WINDOW_END - WINDOW_START)
    return min(WINDOW_END, max(WINDOW_START, int(raw // 256) * 256))


def speedrun_lr_scale(step: int, total_updates: int = NUM_UPDATES) -> float:
    if COOLDOWN_UPDATES is None:
        progress = step / max(total_updates - 1, 1)
        return backbone_lr_scale_for_progress(progress, step, total_updates)
    cooldown_updates = min(COOLDOWN_UPDATES, total_updates)
    cooldown_start = total_updates - cooldown_updates
    if step < cooldown_start:
        return 1.0
    return max(0.0, (total_updates - step - 1) / max(cooldown_updates, 1))


def ssonn_cosine_lr(cycle_step: int, total_steps: int | None = None) -> float:
    steps = SSONN_CYCLE_STEPS if total_steps is None else total_steps
    if steps <= 0:
        raise ValueError("SSONN cosine schedule needs a positive step count.")
    denominator = max(steps - 1, 1)
    position = min(max(cycle_step, 0), denominator)
    return 0.5 * SSONN_LEARNING_RATE * (1.0 - math.cos(2.0 * math.pi * position / denominator))


def cycle_window_for_steps(cycle_step: int, cycle_steps: int) -> int:
    denominator = max(cycle_steps - 1, 1)
    fraction = min(max(cycle_step, 0), denominator) / denominator
    raw = WINDOW_START + fraction * (WINDOW_END - WINDOW_START)
    return min(WINDOW_END, max(WINDOW_START, int(raw // 256) * 256))


def cycle_window(cycle_step: int) -> int:
    return cycle_window_for_steps(cycle_step, SSONN_CYCLE_STEPS)


def cycle_token_budget(cycle_steps: int) -> int:
    sequences_per_update = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
    return sequences_per_update * sum(cycle_window_for_steps(step, cycle_steps) for step in range(cycle_steps))


@dataclass(frozen=True)
class UpdateDescriptor:
    index: int
    total_updates: int
    window: int
    sequences: int
    tokens: int
    progress: float
    backbone_lr_scale: float
    ssonn_lr_scale: float

    @property
    def micro_batch_sequences(self) -> int:
        return max(MICRO_BATCH_SIZE, (MICRO_BATCH_SIZE * WINDOW_END) // self.window)

    @property
    def accumulation_steps(self) -> int:
        return math.ceil(self.sequences / self.micro_batch_sequences)


def window_for_progress(progress: float) -> int:
    if progress >= 1.0:
        return WINDOW_END
    raw = WINDOW_START + progress * (WINDOW_END - WINDOW_START)
    return min(WINDOW_END, max(WINDOW_START, int(raw // 256) * 256))


def backbone_lr_scale_for_progress(progress: float, update: int, total_updates: int) -> float:
    if COOLDOWN_UPDATES is not None:
        cooldown = min(max(COOLDOWN_UPDATES, 1), total_updates)
        cooldown_start = total_updates - cooldown
        if update < cooldown_start:
            return 1.0
        return max(0.0, (total_updates - update - 1) / max(cooldown - 1, 1))
    cooldown_start = 1.0 - COOLDOWN_FRACTION
    if progress < cooldown_start:
        return 1.0
    return max(0.0, (1.0 - progress) / COOLDOWN_FRACTION)


def build_update_schedule(token_budget: int, forced_steps: int = 0) -> list[UpdateDescriptor]:
    if token_budget <= 0:
        raise ValueError("A token schedule needs a positive budget.")
    total_updates = forced_steps if forced_steps > 0 else math.ceil(token_budget / TARGET_TOKENS_PER_UPDATE)
    total_updates = max(total_updates, 2)
    provisional: list[tuple[int, int, float]] = []
    scheduled_tokens = 0
    for update in range(total_updates):
        progress = update / (total_updates - 1)
        window = window_for_progress(progress)
        if update == total_updates - 1 and forced_steps <= 0:
            remaining = max(window, token_budget - scheduled_tokens)
            sequences = max(1, math.ceil(remaining / window))
        else:
            sequences = max(1, round(TARGET_TOKENS_PER_UPDATE / window))
        tokens = sequences * window
        provisional.append((window, sequences, progress))
        scheduled_tokens += tokens

    schedule = []
    for update, (window, sequences, progress) in enumerate(provisional):
        schedule.append(UpdateDescriptor(
            index=update,
            total_updates=total_updates,
            window=window,
            sequences=sequences,
            tokens=window * sequences,
            progress=progress,
            backbone_lr_scale=backbone_lr_scale_for_progress(progress, update, total_updates),
            ssonn_lr_scale=0.5 * (1.0 - math.cos(2.0 * math.pi * progress)),
        ))
    return schedule


def schedule_token_count(schedule: Sequence[UpdateDescriptor]) -> int:
    return sum(update.tokens for update in schedule)


def infer_full_pass_cycle_steps(train_tokens: int) -> int:
    if train_tokens <= 0:
        raise ValueError("Cannot infer SSONN cycle length from an empty dataset.")
    sequences_per_update = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
    low = 1
    high = max(2, math.ceil(train_tokens / (sequences_per_update * WINDOW_START)))
    while low < high:
        middle = (low + high) // 2
        if cycle_token_budget(middle) >= train_tokens:
            high = middle
        else:
            low = middle + 1
    return low


def has_training_after_step(step: int, total_updates: int) -> bool:
    return step + 1 < total_updates


def lm_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits.flatten(0, 1)
    elif logits.ndim != 2:
        raise ValueError(f"Expected [batch, seq, vocab] or [seq, vocab], got {tuple(logits.shape)}")
    return F.cross_entropy(logits, targets.flatten())


def ssonn_only_weights_init(module: nn.Module) -> None:
    if isinstance(module, ScaledNet):
        module.reinitialize_parameters()


def full_model_reinit(model: nn.Module) -> None:
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    else:
        raise TypeError("Model does not expose reset_parameters().")


def layer_input_widths(model: AttentionOnlyGPT) -> dict[str, int]:
    return {name: getattr(model, name).in_features for name in model.ssonn_layer_names}


def make_edge_finder(seq_len: int, slices: Sequence[ShardSlice] | None = None) -> EdgeFinder:
    return EdgeFinder(
        metric=AbsGradientEdgeMetric(lm_cross_entropy),
        dataloader=FixedFinderLoader(
            ARCHITECTURE_SLICES if slices is None else slices,
            FINDER_BATCHES,
            seq_len,
            FINDER_STREAM_SEED,
        ),
        device=device,
        aggregation_mode="mean",
        max_to_replace=SSONN_MAX_NEW_EDGES,
    )


def comparison_metrics(
    mode: str,
    train_loss: float,
    val_loss: float,
    seq_len: int,
    cycle_index: int,
    cycle_step: int,
    cycle_tokens_seen: int,
    global_tokens_seen: int,
    tokens_per_second: float,
    grad_norm: torch.Tensor,
    elapsed: float,
) -> dict[str, float]:
    return {
        "compare/mode": MODE_ID[mode],
        "compare/train_loss": train_loss,
        "compare/val_loss": val_loss,
        "compare/val_perplexity": math.exp(min(val_loss, 20.0)),
        "compare/window": seq_len,
        "compare/cycle_index": cycle_index,
        "compare/cycle_step": cycle_step,
        "compare/cycle_tokens_seen": cycle_tokens_seen,
        "compare/global_tokens_seen": global_tokens_seen,
        "compare/effective_batch_sequences": MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS,
        "compare/tokens_per_second": tokens_per_second,
        "compare/grad_norm": float(grad_norm),
        "compare/elapsed_seconds": elapsed,
    }


def create_comet_experiment():
    if not os.environ.get("COMET_API_KEY") or not os.environ.get("COMET_WORKSPACE"):
        print("Comet logging disabled: set COMET_API_KEY and COMET_WORKSPACE to enable it.")
        return None
    from comet_ml import start

    experiment = start(
        api_key=os.environ["COMET_API_KEY"],
        workspace=os.environ["COMET_WORKSPACE"],
        project_name=os.environ.get("COMET_PROJECT_NAME", "nanogpt-ssonn"),
    )
    experiment.set_name(f"nanogpt_h200_ssonn_architecture_first_{ARGS.amp_dtype}_seed{SEED}")
    experiment.log_parameters({
        "run_mode": RUN_MODE,
        "mlp_kind": MLP_KIND,
        "init_mode": INIT_MODE,
        "fixed_cycle_tokens": FIXED_CYCLE_TOKENS,
        "full_reinit_after_expansion": FULL_REINIT_AFTER_EXPANSION,
        "full_reinit_after_architecture_search": True,
        "reinit_mode": REINIT_MODE,
        "model": asdict(MODEL_CONFIG),
        "num_updates": NUM_UPDATES,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "micro_batch_sequences": MICRO_BATCH_SIZE,
        "micro_batch_sequences_meaning": "maximum-window batch size",
        "activation_token_envelope": MICRO_BATCH_SIZE * WINDOW_END,
        "dynamic_microbatch": True,
        "dynamic_accumulation": True,
        "legacy_grad_accum_steps": GRAD_ACCUM_STEPS,
        "target_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "cooldown_fraction": COOLDOWN_FRACTION,
        "cooldown_updates_override": COOLDOWN_UPDATES,
        "amp_dtype": ARGS.amp_dtype,
        "grad_scaler": USE_SCALER,
        "gpu_arg": ARGS.gpu,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ssonn_learning_rate": SSONN_LEARNING_RATE,
        "ssonn_cycle_steps": SSONN_CYCLE_STEPS,
        "ssonn_cycle_tokens": SSONN_CYCLE_TOKEN_COUNT,
        "architecture_fraction": ARCHITECTURE_FRACTION,
        "architecture_expansions": ARCHITECTURE_EXPANSIONS,
        "architecture_tokens": ARCHITECTURE_TOKEN_COUNT,
        "architecture_cycle_steps": SSONN_CYCLE_STEPS,
        "training_passes": TRAINING_PASSES,
        "full_training_target_tokens": FULL_TRAINING_TARGET_TOKENS,
        "full_training_steps": FULL_TRAINING_STEPS,
        "full_training_pass_steps": FULL_TRAINING_PASS_STEPS,
        "full_training_tokens": FULL_TRAINING_TOKEN_COUNT,
        "train_shards": len(TRAIN_FILES),
        "train_tokens": TRAIN_TOKEN_COUNT,
        "periodic_validation_tokens": PERIODIC_VALIDATION_TOKENS,
        "periodic_validation_scheduled_tokens": validation_scheduled_tokens(PERIODIC_VALIDATION_TOKENS),
        "final_validation_tokens": FINAL_VALIDATION_TOKENS,
        "final_validation_scheduled_tokens": validation_scheduled_tokens(FINAL_VALIDATION_TOKENS),
        "validation_seq_len": VALIDATION_SEQ_LEN,
        "validation_protocol": "fixed_tokens_fixed_context_document_local",
        "attention_backend": "flex" if USE_FLEX_ATTENTION else "sdpa",
        "torch_compile": USE_COMPILE,
        "document_local_attention": True,
        "backbone_grad_clip": BACKBONE_GRAD_CLIP,
        "ssonn_grad_clip": SSONN_GRAD_CLIP,
        "ssonn_expansion_threshold": SSONN_EXPANSION_THRESHOLD,
        "ssonn_max_new_edges": SSONN_MAX_NEW_EDGES,
        "finder_batches": FINDER_BATCHES,
    })
    return experiment


def backward_loss(loss: torch.Tensor, scaler: torch.amp.GradScaler | None) -> None:
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def finish_optimizer_step(
    optimizers: tuple[torch.optim.Optimizer, ...],
    scaler: torch.amp.GradScaler | None,
    model: nn.Module,
) -> tuple[torch.Tensor, bool]:
    if scaler is not None:
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    skipped_step = not bool(torch.isfinite(grad_norm).item())
    if skipped_step:
        if scaler is not None:
            scaler.update()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        return grad_norm, True
    if scaler is not None:
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()
    else:
        for optimizer in optimizers:
            optimizer.step()
    return grad_norm, False


@dataclass(frozen=True)
class GradientStepStats:
    backbone_norm: float
    ssonn_norm: float
    backbone_clip_coefficient: float
    ssonn_clip_coefficient: float
    skipped: bool


def optimizer_parameters(*optimizers: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [parameter for optimizer in optimizers for group in optimizer.param_groups for parameter in group["params"]]


def finish_grouped_optimizer_step(
    muon: torch.optim.Optimizer,
    backbone_adam: torch.optim.Optimizer,
    ssonn_adam: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> GradientStepStats:
    optimizers = (muon, backbone_adam, ssonn_adam)
    if scaler is not None:
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
    backbone_parameters = optimizer_parameters(muon, backbone_adam)
    ssonn_parameters = optimizer_parameters(ssonn_adam)
    backbone_norm = torch.nn.utils.clip_grad_norm_(
        backbone_parameters,
        BACKBONE_GRAD_CLIP if BACKBONE_GRAD_CLIP > 0 else float("inf"),
    )
    ssonn_norm = torch.nn.utils.clip_grad_norm_(
        ssonn_parameters,
        SSONN_GRAD_CLIP if SSONN_GRAD_CLIP > 0 else float("inf"),
    )
    finite = bool(torch.isfinite(backbone_norm).item() and torch.isfinite(ssonn_norm).item())
    backbone_value = float(backbone_norm)
    ssonn_value = float(ssonn_norm)
    backbone_coefficient = min(1.0, BACKBONE_GRAD_CLIP / (backbone_value + 1e-12)) if BACKBONE_GRAD_CLIP > 0 else 1.0
    ssonn_coefficient = min(1.0, SSONN_GRAD_CLIP / (ssonn_value + 1e-12)) if SSONN_GRAD_CLIP > 0 else 1.0
    if not finite:
        if scaler is not None:
            scaler.update()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        return GradientStepStats(backbone_value, ssonn_value, backbone_coefficient, ssonn_coefficient, True)
    if scaler is not None:
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()
    else:
        for optimizer in optimizers:
            optimizer.step()
    return GradientStepStats(backbone_value, ssonn_value, backbone_coefficient, ssonn_coefficient, False)


def run_scheduled_update(
    model: nn.Module,
    stream: TokenShardStream,
    update: UpdateDescriptor,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    scaler: torch.amp.GradScaler | None,
    label: str,
) -> tuple[float, GradientStepStats]:
    muon, backbone_adam, ssonn_adam = optimizers
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    remaining_sequences = update.sequences
    train_loss = 0.0
    while remaining_sequences > 0:
        batch_size = min(update.micro_batch_sequences, remaining_sequences)
        x_cpu, y_cpu = stream.next_batch(update.window, batch_size)
        x, y = to_device(x_cpu, y_cpu)
        batch_tokens = batch_size * update.window
        loss_weight = batch_tokens / update.tokens
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP and device.type == "cuda"):
            document_ids, positions = document_ids_and_positions(x)
            logits = model(x, positions, make_document_attention_mask(document_ids))
            loss = lm_cross_entropy(logits, y)
        assert_finite_tensor(loss, label)
        backward_loss(loss * loss_weight, scaler)
        train_loss += loss.detach().item() * loss_weight
        remaining_sequences -= batch_size
    stats = finish_grouped_optimizer_step(muon, backbone_adam, ssonn_adam, scaler)
    return train_loss, stats


def train_autocast_stage(model: AttentionOnlyGPT, mode: str, experiment) -> AttentionOnlyGPT:
    if mode not in {"pretrain", "mlp"}:
        raise ValueError(f"Unsupported autocast stage mode {mode!r}.")
    model.train()
    muon, adam = make_pretrain_optimizers(model)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)
    scaler_or_none = scaler if USE_SCALER else None
    train_stream = TokenShardStream(TRAIN_FILES, seed=TRAIN_STREAM_SEED)
    checkpoint_path = BASE_CHECKPOINT if mode == "pretrain" else MLP_CHECKPOINT
    latest_path = checkpoint_path.with_name(checkpoint_path.stem.replace("_best", "") + "_latest.pt")
    started = time.perf_counter()
    tokens_seen = 0
    best_val = float("inf")

    for step in range(NUM_UPDATES):
        seq_len = speedrun_window(step)
        lr_scale = speedrun_lr_scale(step)
        muon.param_groups[0]["momentum"] = 0.85 + 0.10 * min(step / 300, 1.0)
        muon.param_groups[0]["lr"] = PRETRAIN_MUON_LR * lr_scale
        adam.param_groups[0]["lr"] = PRETRAIN_ADAM_LR * lr_scale
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)

        train_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            x_cpu, y_cpu = train_stream.next_batch(seq_len, MICRO_BATCH_SIZE)
            x, y = to_device(x_cpu, y_cpu)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP and device.type == "cuda"):
                loss = lm_cross_entropy(model(x), y) / GRAD_ACCUM_STEPS
            assert_finite_tensor(loss, f"{mode} loss at update {step + 1}")
            backward_loss(loss, scaler_or_none)
            train_loss += loss.detach().item()

        tokens_seen += GRAD_ACCUM_STEPS * MICRO_BATCH_SIZE * seq_len
        grad_norm, skipped_step = finish_optimizer_step((muon, adam), scaler_or_none, model)
        if skipped_step:
            print(f"update {step + 1:6d}/{NUM_UPDATES} | skipped non-finite {mode} gradients")
            if experiment is not None:
                experiment.log_metrics({
                    f"{mode}/skipped_step": 1,
                    f"{mode}/grad_norm_finite": 0,
                    f"{mode}/window": seq_len,
                    f"{mode}/global_tokens_seen": tokens_seen,
                }, step=step + 1)
            continue
        assert_finite_model(model, f"{mode} model after update {step + 1}")

        if (step + 1) % EVAL_INTERVAL == 0 or step == 0 or step + 1 == NUM_UPDATES:
            val_loss = evaluate_fixed(model, seq_len, EVAL_BATCHES)
            elapsed = time.perf_counter() - started
            tokens_per_second = tokens_seen / max(elapsed, 1.0)
            print(
                f"update {step + 1:6d}/{NUM_UPDATES} | window {seq_len:4d} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | {tokens_per_second:.0f} tok/s"
            )
            if experiment is not None:
                metrics = {
                    f"{mode}/train_loss": train_loss,
                    f"{mode}/val_loss": val_loss,
                    f"{mode}/val_perplexity": math.exp(min(val_loss, 20.0)),
                    f"{mode}/window": seq_len,
                    f"{mode}/global_tokens_seen": tokens_seen,
                    f"{mode}/tokens_per_second": tokens_per_second,
                    f"{mode}/elapsed_seconds": elapsed,
                    f"{mode}/grad_norm": float(grad_norm),
                    f"{mode}/grad_norm_finite": 1,
                    f"{mode}/skipped_step": 0,
                    f"{mode}/muon_lr": muon.param_groups[0]["lr"],
                    f"{mode}/adam_lr": adam.param_groups[0]["lr"],
                }
                metrics.update(comparison_metrics(
                    mode, train_loss, val_loss, seq_len, 0, step, tokens_seen, tokens_seen, tokens_per_second, grad_norm, elapsed
                ))
                experiment.log_metrics(metrics, step=step + 1)
            model.train()

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(checkpoint_path, model, {"muon": muon.state_dict(), "adam": adam.state_dict()}, step + 1)

        if (step + 1) % CHECKPOINT_INTERVAL == 0 or step + 1 == NUM_UPDATES:
            save_checkpoint(latest_path, model, {"muon": muon.state_dict(), "adam": adam.state_dict()}, step + 1)
    return model


def train_ssonn_stage(model: AttentionOnlyGPT, experiment) -> AttentionOnlyGPT:
    if RUN_MODE != "ssonn":
        raise ValueError("Set --mode ssonn for this stage.")
    full_model_reinit(model)
    ssonn_train.weights_init = ssonn_only_weights_init
    muon, backbone_adam, ssonn_adam = make_ssonn_stage_optimizers(model)
    ssonn_train.apply_sparsity_mask(model)
    thresholds = {name: SSONN_EXPANSION_THRESHOLD for name in model.ssonn_layer_names}
    train_stream = TokenShardStream(TRAIN_FILES, seed=TRAIN_STREAM_SEED)
    initial_loss = evaluate_fixed(model, WINDOW_START, SSONN_EVAL_BATCHES)
    print(f"from-scratch residual SSONN initial val loss: {initial_loss:.4f}")

    best_val = float("inf")
    cycle_best_val = float("inf")
    global_tokens_seen = 0
    cycle_tokens_seen = 0
    cycle_index = 0
    cycle_start = 0
    started = time.perf_counter()

    for step in range(NUM_UPDATES):
        cycle_step = step - cycle_start
        seq_len = cycle_window(cycle_step)
        lr_scale = speedrun_lr_scale(cycle_step, SSONN_CYCLE_STEPS)
        muon.param_groups[0]["momentum"] = 0.85 + 0.10 * min(cycle_step / 300, 1.0)
        muon.param_groups[0]["lr"] = PRETRAIN_MUON_LR * lr_scale
        backbone_adam.param_groups[0]["lr"] = PRETRAIN_ADAM_LR * lr_scale
        ssonn_lr = ssonn_cosine_lr(cycle_step)
        ssonn_adam.param_groups[0]["lr"] = ssonn_lr

        model.train()
        for optimizer in (muon, backbone_adam, ssonn_adam):
            optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            x_cpu, y_cpu = train_stream.next_batch(seq_len, MICRO_BATCH_SIZE)
            x, y = to_device(x_cpu, y_cpu)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP and device.type == "cuda"):
                loss = lm_cross_entropy(model(x), y) / GRAD_ACCUM_STEPS
            assert_finite_tensor(loss, f"SSONN loss at update {step + 1}")
            backward_loss(loss, None)
            train_loss += loss.detach().item()

        step_tokens = GRAD_ACCUM_STEPS * MICRO_BATCH_SIZE * seq_len
        global_tokens_seen += step_tokens
        cycle_tokens_seen += step_tokens
        grad_norm, skipped_step = finish_optimizer_step((muon, backbone_adam, ssonn_adam), None, model)
        if skipped_step:
            print(f"update {step + 1:6d}/{NUM_UPDATES} | skipped non-finite SSONN gradients")
            continue
        assert_finite_model(model, f"SSONN model after update {step + 1}")

        expanded = False
        new_edges = 0
        full_reinit = False
        pre_expansion_val_loss = float("nan")
        post_reinit_val_loss = float("nan")
        cycle_last_pre_expansion_val_loss = float("nan")
        new_edges_by_layer = {name: 0 for name in model.ssonn_layer_names}
        selection_stats_by_layer: dict[str, dict[str, float | int]] = {}
        log_cycle_index = cycle_index
        log_cycle_step = cycle_step
        log_cycle_tokens_seen = cycle_tokens_seen
        eval_val_loss: float | None = None
        start_next_cycle = False

        cycle_finished = cycle_step + 1 >= SSONN_CYCLE_STEPS
        if cycle_finished:
            pre_expansion_val_loss = evaluate_fixed(model, seq_len, SSONN_EVAL_BATCHES)
            cycle_last_pre_expansion_val_loss = pre_expansion_val_loss
            eval_val_loss = pre_expansion_val_loss
            cycle_best_val = min(cycle_best_val, pre_expansion_val_loss)
            optimizer_state = {
                "muon": muon.state_dict(),
                "backbone_adam": backbone_adam.state_dict(),
                "ssonn_adam": ssonn_adam.state_dict(),
            }
            save_checkpoint(
                CHECKPOINT_DIR / f"finewebedu_ssonn_cycle_{cycle_index:02d}_trained.pt",
                model,
                optimizer_state,
                step + 1,
            )
            if pre_expansion_val_loss < best_val:
                best_val = pre_expansion_val_loss
                save_checkpoint(
                    SSONN_CHECKPOINT,
                    model,
                    optimizer_state,
                    step + 1,
                )
            del optimizer_state
            if has_training_after_step(step, NUM_UPDATES):
                edge_finder = make_edge_finder(seq_len)
                before_widths = layer_input_widths(model)
                expanded, new_edges = ssonn_train.expand_model(model, edge_finder, thresholds)
                after_widths = layer_input_widths(model)
                new_edges_by_layer = {name: after_widths[name] - before_widths[name] for name in model.ssonn_layer_names}
                selection_stats_by_layer = {
                    name: edge_finder.last_selection_stats.get(id(getattr(model, name)), {})
                    for name in model.ssonn_layer_names
                }
                saturated_layers = sum(int(stats.get("selection_saturated", 0)) for stats in selection_stats_by_layer.values())
                print(f"edge selection: {new_edges} selected | {saturated_layers}/{len(model.ssonn_layer_names)} layers hit the per-layer cap")
                if expanded:
                    full_model_reinit(model)
                    full_reinit = True
                    muon, backbone_adam, ssonn_adam = make_ssonn_stage_optimizers(model)
                    ssonn_train.apply_sparsity_mask(model)
                    post_reinit_val_loss = evaluate_fixed(model, WINDOW_START, SSONN_EVAL_BATCHES)
                    assert_finite_model(model, f"SSONN model after full reinit at update {step + 1}")
                train_stream = TokenShardStream(TRAIN_FILES, seed=TRAIN_STREAM_SEED)
                start_next_cycle = True
            else:
                print("final cycle complete: skipping expansion and full reinit because no training cycle remains")

        should_eval = (step + 1) % EVAL_INTERVAL == 0 or step == 0 or cycle_finished or expanded or step + 1 == NUM_UPDATES
        if should_eval:
            if eval_val_loss is None:
                eval_val_loss = evaluate_fixed(model, seq_len, SSONN_EVAL_BATCHES)
            cycle_best_val = min(cycle_best_val, eval_val_loss)
            elapsed = time.perf_counter() - started
            tokens_per_second = global_tokens_seen / max(elapsed, 1.0)
            ppl_text = ">= exp(20)" if eval_val_loss >= 20.0 else f"{math.exp(eval_val_loss):.2f}"
            total_params = count_parameters(model)
            nonzero_params = count_nonzero_parameters(model)
            print(
                f"update {step + 1:6d}/{NUM_UPDATES} | cycle {log_cycle_index:02d} "
                f"step {log_cycle_step + 1:5d}/{SSONN_CYCLE_STEPS} | window {seq_len:4d} | "
                f"train {train_loss:.4f} | val {eval_val_loss:.4f} | ppl {ppl_text} | "
                f"ssonn_lr {ssonn_lr:.2e} | expanded {expanded} edges {new_edges} | {tokens_per_second:.0f} tok/s"
            )
            if experiment is not None:
                metrics = {
                    "ssonn/train_loss": train_loss,
                    "ssonn/val_loss": eval_val_loss,
                    "ssonn/val_perplexity": math.exp(min(eval_val_loss, 20.0)),
                    "ssonn/window": seq_len,
                    "ssonn/cycle_index": log_cycle_index,
                    "ssonn/cycle_step": log_cycle_step,
                    "ssonn/cycle_tokens_seen": log_cycle_tokens_seen,
                    "ssonn/global_tokens_seen": global_tokens_seen,
                    "ssonn/tokens_per_second": tokens_per_second,
                    "ssonn/elapsed_seconds": elapsed,
                    "ssonn/grad_norm": float(grad_norm),
                    "ssonn/muon_lr": muon.param_groups[0]["lr"],
                    "ssonn/backbone_adam_lr": backbone_adam.param_groups[0]["lr"],
                    "ssonn/lr": ssonn_lr,
                    "ssonn/expanded": int(expanded),
                    "ssonn/new_edges": int(new_edges),
                    "ssonn/full_reinit": int(full_reinit),
                    "ssonn/total_params": total_params,
                    "ssonn/nonzero_params": nonzero_params,
                    "ssonn/sparsity": 1.0 - (nonzero_params / max(total_params, 1)),
                    "ssonn/sub_linear_count": count_ssonn_sublinears(model),
                    "ssonn/cycle_best_val_loss": cycle_best_val,
                    "ssonn/cycle_last_pre_expansion_val_loss": cycle_last_pre_expansion_val_loss,
                    "ssonn/pre_expansion_val_loss": pre_expansion_val_loss,
                    "ssonn/post_reinit_val_loss": post_reinit_val_loss,
                    "ssonn/candidate_edges_before_cap": sum(int(stats.get("candidate_count", 0)) for stats in selection_stats_by_layer.values()),
                    "ssonn/saturated_layer_count": sum(int(stats.get("selection_saturated", 0)) for stats in selection_stats_by_layer.values()),
                }
                metrics.update(comparison_metrics(
                    "ssonn",
                    train_loss,
                    eval_val_loss,
                    seq_len,
                    log_cycle_index,
                    log_cycle_step,
                    log_cycle_tokens_seen,
                    global_tokens_seen,
                    tokens_per_second,
                    grad_norm,
                    elapsed,
                ))
                metrics.update({f"ssonn/new_edges/{name}": int(value) for name, value in new_edges_by_layer.items()})
                for name, stats in selection_stats_by_layer.items():
                    metrics.update({
                        f"ssonn/candidate_edges/{name}": int(stats.get("candidate_count", 0)),
                        f"ssonn/selection_saturated/{name}": int(stats.get("selection_saturated", 0)),
                        f"ssonn/selected_metric_min/{name}": float(stats.get("selected_metric_min", 0.0)),
                        f"ssonn/selected_metric_max/{name}": float(stats.get("selected_metric_max", 0.0)),
                    })
                experiment.log_metrics(metrics, step=step + 1)
            model.train()

            if not expanded and eval_val_loss < best_val:
                best_val = eval_val_loss
                save_checkpoint(
                    SSONN_CHECKPOINT,
                    model,
                    {"muon": muon.state_dict(), "backbone_adam": backbone_adam.state_dict(), "ssonn_adam": ssonn_adam.state_dict()},
                    step + 1,
                )

        if expanded or (step + 1) % CHECKPOINT_INTERVAL == 0 or step + 1 == NUM_UPDATES:
            save_checkpoint(
                SSONN_CHECKPOINT.with_name("finewebedu_ssonn_latest.pt"),
                model,
                {"muon": muon.state_dict(), "backbone_adam": backbone_adam.state_dict(), "ssonn_adam": ssonn_adam.state_dict()},
                step + 1,
            )
        if start_next_cycle:
            cycle_index += 1
            cycle_start = step + 1
            cycle_tokens_seen = 0
            cycle_best_val = float("inf")
    return model


def optimizer_state_dict(muon, backbone_adam, ssonn_adam) -> dict:
    return {
        "muon": muon.state_dict(),
        "backbone_adam": backbone_adam.state_dict(),
        "ssonn_adam": ssonn_adam.state_dict(),
    }


def expand_model_with_topology(
    model: AttentionOnlyGPT,
    edge_finder: EdgeFinder,
    thresholds: dict[str, float],
    forward_model: nn.Module | None = None,
) -> tuple[bool, int, dict[str, dict[str, list[int]]]]:
    """Select all edges on a frozen topology, then apply and record them."""
    selection_model = model if forward_model is None else forward_model
    selected: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in model.ssonn_layer_names:
        layer = getattr(model, name)
        parents, children = edge_finder.choose_edges_threshold(
            selection_model,
            layer,
            thresholds[name],
            embed=False,
        )
        selected[name] = (parents, children)

    topology: dict[str, dict[str, list[int]]] = {}
    total_new_edges = 0
    for name, (parents, children) in selected.items():
        topology[name] = {
            "parents": parents.detach().cpu().tolist(),
            "children": children.detach().cpu().tolist(),
        }
        if len(parents) == 0:
            continue
        getattr(model, name).put_sub_linear(parents, children)
        total_new_edges += len(parents)
        print(f"[Expansion] Layer '{name}': adding sub-layer for {len(parents)} edges.")
    return total_new_edges > 0, total_new_edges, topology


def architecture_checkpoint_metadata(
    phase: str,
    expansion_index: int,
    phase_step: int,
    topology_history: list[dict],
) -> dict:
    return {
        "phase": phase,
        "expansion_index": expansion_index,
        "phase_step": phase_step,
        "checkpoint_schema_version": 2,
        "architecture_fraction": ARCHITECTURE_FRACTION,
        "architecture_tokens": ARCHITECTURE_TOKEN_COUNT,
        "architecture_cycle_steps": SSONN_CYCLE_STEPS,
        "architecture_cycle_scheduled_tokens": SSONN_CYCLE_TOKEN_COUNT,
        "architecture_search_total_tokens": ARCHITECTURE_EXPANSIONS * SSONN_CYCLE_TOKEN_COUNT,
        "full_training_steps": FULL_TRAINING_STEPS,
        "full_training_target_tokens": FULL_TRAINING_TARGET_TOKENS,
        "full_training_scheduled_tokens": FULL_TRAINING_TOKEN_COUNT,
        "target_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
        "cooldown_fraction": COOLDOWN_FRACTION,
        "attention_backend": "flex" if USE_FLEX_ATTENTION else "sdpa",
        "torch_compile": USE_COMPILE,
        "document_local_attention": True,
        "periodic_validation_tokens": PERIODIC_VALIDATION_TOKENS,
        "periodic_validation_scheduled_tokens": validation_scheduled_tokens(PERIODIC_VALIDATION_TOKENS),
        "final_validation_tokens": FINAL_VALIDATION_TOKENS,
        "final_validation_scheduled_tokens": validation_scheduled_tokens(FINAL_VALIDATION_TOKENS),
        "validation_seq_len": VALIDATION_SEQ_LEN,
        "topology_history": topology_history,
    }


def configure_optimizer_schedule(
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    update: UpdateDescriptor,
) -> tuple[float, float, float]:
    muon, backbone_adam, ssonn_adam = optimizers
    muon.param_groups[0]["momentum"] = 0.85 + 0.10 * min(update.index / 300, 1.0)
    muon_lr = PRETRAIN_MUON_LR * update.backbone_lr_scale
    backbone_lr = PRETRAIN_ADAM_LR * update.backbone_lr_scale
    ssonn_lr = SSONN_LEARNING_RATE * update.ssonn_lr_scale
    muon.param_groups[0]["lr"] = muon_lr
    backbone_adam.param_groups[0]["lr"] = backbone_lr
    ssonn_adam.param_groups[0]["lr"] = ssonn_lr
    return muon_lr, backbone_lr, ssonn_lr


def train_architecture_first(model: AttentionOnlyGPT, experiment) -> AttentionOnlyGPT:
    architecture_schedule = build_update_schedule(ARCHITECTURE_TOKEN_COUNT, ARGS.architecture_cycle_steps)
    full_schedule = build_update_schedule(FULL_TRAINING_TARGET_TOKENS, ARGS.full_training_steps)
    global_step = 0
    global_tokens_seen = 0
    started = time.perf_counter()
    topology_history: list[dict] = []
    thresholds = {name: SSONN_EXPANSION_THRESHOLD for name in model.ssonn_layer_names}

    set_seed(SEED)
    full_model_reinit(model)
    ssonn_train.apply_sparsity_mask(model)
    initial_val_loss = evaluate_token_budget(model, PERIODIC_VALIDATION_TOKENS)
    print(f"architecture search initial fixed val loss: {initial_val_loss:.4f}")

    for expansion_index in range(ARCHITECTURE_EXPANSIONS):
        train_stream = TokenShardStream(seed=TRAIN_STREAM_SEED, slices=ARCHITECTURE_SLICES)
        optimizers = make_ssonn_stage_optimizers(model)
        cycle_model: nn.Module = torch.compile(model) if USE_COMPILE else model
        scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)
        scaler_or_none = scaler if USE_SCALER else None
        cycle_tokens_seen = 0
        cycle_started = time.perf_counter()
        last_val_loss = float("nan")
        print(
            f"architecture cycle {expansion_index + 1}/{ARCHITECTURE_EXPANSIONS}: "
            f"{len(architecture_schedule)} updates | {schedule_token_count(architecture_schedule):,} scheduled tokens"
        )

        for update in architecture_schedule:
            muon_lr, backbone_lr, ssonn_lr = configure_optimizer_schedule(optimizers, update)
            cycle_model.train()
            train_loss, grad = run_scheduled_update(
                cycle_model,
                train_stream,
                update,
                optimizers,
                scaler_or_none,
                f"Architecture loss at cycle {expansion_index}, update {update.index + 1}",
            )
            cycle_tokens_seen += update.tokens
            global_tokens_seen += update.tokens
            global_step += 1
            if grad.skipped:
                print(f"architecture step {global_step}: skipped non-finite gradients")
            else:
                assert_finite_model(model, f"Architecture model after global update {global_step}")

            should_eval = update.index == 0 or (update.index + 1) % EVAL_INTERVAL == 0 or update.index + 1 == len(architecture_schedule)
            if should_eval:
                last_val_loss = evaluate_token_budget(cycle_model, PERIODIC_VALIDATION_TOKENS)
                elapsed = time.perf_counter() - started
                cycle_elapsed = time.perf_counter() - cycle_started
                tokens_per_second = cycle_tokens_seen / max(cycle_elapsed, 1.0)
                print(
                    f"architecture {expansion_index + 1}/{ARCHITECTURE_EXPANSIONS} | "
                    f"update {update.index + 1:4d}/{len(architecture_schedule)} | window {update.window:4d} | "
                    f"tokens/update {update.tokens:,} | accum {update.accumulation_steps:3d} | "
                    f"train {train_loss:.4f} | val {last_val_loss:.4f} | ssonn_lr {ssonn_lr:.2e} | "
                    f"{tokens_per_second:.0f} tok/s"
                )
                if experiment is not None:
                    experiment.log_metrics({
                        "architecture/phase": 0,
                        "architecture/expansion_index": expansion_index,
                        "architecture/cycle_step": update.index,
                        "architecture/cycle_progress": update.progress,
                        "architecture/train_loss": train_loss,
                        "architecture/val_loss": last_val_loss,
                        "architecture/val_perplexity": math.exp(min(last_val_loss, 20.0)),
                        "architecture/window": update.window,
                        "architecture/planned_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
                        "architecture/actual_tokens_per_update": update.tokens,
                        "architecture/micro_batch_sequences": update.micro_batch_sequences,
                        "architecture/accumulation_steps": update.accumulation_steps,
                        "architecture/subset_fraction": ARCHITECTURE_FRACTION,
                        "architecture/subset_tokens": ARCHITECTURE_TOKEN_COUNT,
                        "architecture/cycle_tokens_seen": cycle_tokens_seen,
                        "architecture/global_tokens_seen": global_tokens_seen,
                        "architecture/tokens_per_second": tokens_per_second,
                        "architecture/backbone_grad_norm": grad.backbone_norm,
                        "architecture/ssonn_grad_norm": grad.ssonn_norm,
                        "architecture/backbone_clip_coefficient": grad.backbone_clip_coefficient,
                        "architecture/ssonn_clip_coefficient": grad.ssonn_clip_coefficient,
                        "architecture/skipped_step": int(grad.skipped),
                        "architecture/muon_lr": muon_lr,
                        "architecture/backbone_adam_lr": backbone_lr,
                        "architecture/ssonn_lr": ssonn_lr,
                        "architecture/backbone_lr_scale": update.backbone_lr_scale,
                        "architecture/elapsed_seconds": elapsed,
                    }, step=global_step)

        if architecture_schedule[-1].ssonn_lr_scale != 0.0:
            raise AssertionError("Architecture expansion must happen at the zero SSONN LR trough.")
        pre_expansion_val_loss = last_val_loss if math.isfinite(last_val_loss) else evaluate_token_budget(model, PERIODIC_VALIDATION_TOKENS)
        edge_finder = make_edge_finder(WINDOW_END, ARCHITECTURE_SLICES)
        before_widths = layer_input_widths(model)
        params_before = count_parameters(model)
        ssonn_params_before = count_ssonn_stored_parameters(model)
        active_before = count_ssonn_active_connections(model)
        expanded, selected_edges, topology = expand_model_with_topology(
            model, edge_finder, thresholds, forward_model=cycle_model
        )
        del cycle_model
        if expanded:
            ssonn_train.apply_sparsity_mask(model)
        after_widths = layer_input_widths(model)
        params_after = count_parameters(model)
        ssonn_params_after = count_ssonn_stored_parameters(model)
        active_after = count_ssonn_active_connections(model)
        selection_stats = {
            name: edge_finder.last_selection_stats.get(id(getattr(model, name)), {})
            for name in model.ssonn_layer_names
        }
        unique_parents = sum(len(set(layer_topology["parents"])) for layer_topology in topology.values())
        candidate_edges = sum(int(stats.get("candidate_count", 0)) for stats in selection_stats.values())
        saturated_layers = sum(int(stats.get("selection_saturated", 0)) for stats in selection_stats.values())
        event = {
            "expansion_index": expansion_index,
            "global_step": global_step,
            "selected_edges": selected_edges,
            "new_edges": selected_edges,
            "added_nodes": selected_edges,
            "unique_parents": unique_parents,
            "stored_parameter_growth": params_after - params_before,
            "ssonn_parameter_growth": ssonn_params_after - ssonn_params_before,
            "active_connection_growth": active_after - active_before,
            "layers": topology,
        }
        topology_history.append(event)
        print(
            f"architecture expansion {expansion_index + 1}/{ARCHITECTURE_EXPANSIONS}: "
            f"expanded={expanded} selected_edges={selected_edges} unique_parents={unique_parents} "
            f"params_added={params_after - params_before:,} active_connections_added={active_after - active_before:,} "
            f"saturated_layers={saturated_layers}/{len(model.ssonn_layer_names)}"
        )
        if experiment is not None:
            total_params = count_parameters(model)
            nonzero_params = count_nonzero_parameters(model)
            metrics = {
                "architecture/expansion_event": 1,
                "architecture/expanded": int(expanded),
                "architecture/selected_edges": selected_edges,
                "architecture/new_edges": selected_edges,
                "architecture/added_nodes": selected_edges,
                "architecture/unique_parents": unique_parents,
                "architecture/pre_expansion_val_loss": pre_expansion_val_loss,
                "architecture/candidate_edges_before_cap": candidate_edges,
                "architecture/saturated_layer_count": saturated_layers,
                "architecture/stored_parameter_growth": params_after - params_before,
                "architecture/ssonn_parameter_growth": ssonn_params_after - ssonn_params_before,
                "architecture/active_connection_growth": active_after - active_before,
                "architecture/active_connections": active_after,
                "architecture/total_params": total_params,
                "architecture/nonzero_params": nonzero_params,
                "architecture/sparsity": 1.0 - nonzero_params / max(total_params, 1),
                "architecture/sublinear_count": count_ssonn_sublinears(model),
                "architecture/full_reinit": 0,
            }
            for name in model.ssonn_layer_names:
                stats = selection_stats[name]
                layer_selected = after_widths[name] - before_widths[name]
                layer_unique_parents = len(set(topology[name]["parents"]))
                metrics.update({
                    f"architecture/selected_edges/{name}": layer_selected,
                    f"architecture/new_edges/{name}": layer_selected,
                    f"architecture/unique_parents/{name}": layer_unique_parents,
                    f"architecture/candidate_edges/{name}": int(stats.get("candidate_count", 0)),
                    f"architecture/selection_saturated/{name}": int(stats.get("selection_saturated", 0)),
                    f"architecture/selected_metric_min/{name}": float(stats.get("selected_metric_min", 0.0)),
                    f"architecture/selected_metric_max/{name}": float(stats.get("selected_metric_max", 0.0)),
                })
            experiment.log_metrics(metrics, step=global_step)
        save_checkpoint(
            CHECKPOINT_DIR / f"finewebedu_architecture_expansion_{expansion_index + 1:02d}.pt",
            model,
            None,
            global_step,
            architecture_checkpoint_metadata("architecture", expansion_index + 1, len(architecture_schedule), topology_history),
        )

    # Preserve the graph, discard all learned weights once, and start the quality run.
    set_seed(SEED)
    full_model_reinit(model)
    ssonn_train.apply_sparsity_mask(model)
    post_reinit_val_loss = evaluate_token_budget(model, PERIODIC_VALIDATION_TOKENS)
    print(
        f"architecture ready: {count_ssonn_sublinears(model)} sub-linears | "
        f"zero-delta post-reinit fixed val loss {post_reinit_val_loss:.4f}"
    )
    transition_metadata = architecture_checkpoint_metadata(
        "full_training_init", ARCHITECTURE_EXPANSIONS, 0, topology_history
    )
    save_checkpoint(
        CHECKPOINT_DIR / "finewebedu_ssonn_architecture_ready.pt",
        model,
        None,
        global_step,
        transition_metadata,
    )
    if experiment is not None:
        total_params = count_parameters(model)
        nonzero_params = count_nonzero_parameters(model)
        experiment.log_metrics({
            "transition/architecture_ready": 1,
            "transition/full_reinit": 1,
            "transition/zero_delta_ssonn": 1,
            "transition/post_reinit_val_loss": post_reinit_val_loss,
            "transition/total_params": total_params,
            "transition/nonzero_params": nonzero_params,
            "transition/active_connections": count_ssonn_active_connections(model),
            "transition/sparsity": 1.0 - nonzero_params / max(total_params, 1),
            "transition/sublinear_count": count_ssonn_sublinears(model),
        }, step=global_step)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)
    scaler_or_none = scaler if USE_SCALER else None
    optimizers = make_ssonn_stage_optimizers(model)
    training_model: nn.Module = torch.compile(model) if USE_COMPILE else model
    train_stream = TokenShardStream(seed=TRAIN_STREAM_SEED, slices=FULL_TRAINING_SLICES)
    training_tokens_seen = 0
    training_started = time.perf_counter()
    best_val = float("inf")
    latest_path = CHECKPOINT_DIR / "finewebedu_ssonn_architecture_first_latest.pt"

    for update in full_schedule:
        muon_lr, backbone_lr, ssonn_lr = configure_optimizer_schedule(optimizers, update)
        training_model.train()
        train_loss, grad = run_scheduled_update(
            training_model,
            train_stream,
            update,
            optimizers,
            scaler_or_none,
            f"Full-training loss at update {update.index + 1}",
        )
        training_tokens_seen += update.tokens
        global_tokens_seen += update.tokens
        global_step += 1
        if grad.skipped:
            print(f"full-training update {update.index + 1}: skipped non-finite gradients")
        else:
            assert_finite_model(training_model, f"Full-training model after update {update.index + 1}")

        should_eval = update.index == 0 or (update.index + 1) % EVAL_INTERVAL == 0 or update.index + 1 == len(full_schedule)
        checkpoint_model = getattr(training_model, "_orig_mod", training_model)
        if should_eval:
            val_loss = evaluate_token_budget(training_model, PERIODIC_VALIDATION_TOKENS)
            elapsed = time.perf_counter() - started
            training_elapsed = time.perf_counter() - training_started
            tokens_per_second = training_tokens_seen / max(training_elapsed, 1.0)
            print(
                f"full training update {update.index + 1:4d}/{len(full_schedule)} | window {update.window:4d} | "
                f"tokens/update {update.tokens:,} | accum {update.accumulation_steps:3d} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | ssonn_lr {ssonn_lr:.2e} | "
                f"{tokens_per_second:.0f} tok/s"
            )
            if experiment is not None:
                total_params = count_parameters(checkpoint_model)
                nonzero_params = count_nonzero_parameters(checkpoint_model)
                experiment.log_metrics({
                    "training/phase": 1,
                    "training/phase_step": update.index,
                    "training/phase_progress": update.progress,
                    "training/global_step": global_step,
                    "training/train_loss": train_loss,
                    "training/val_loss": val_loss,
                    "training/val_perplexity": math.exp(min(val_loss, 20.0)),
                    "training/window": update.window,
                    "training/planned_tokens_per_update": TARGET_TOKENS_PER_UPDATE,
                    "training/actual_tokens_per_update": update.tokens,
                    "training/micro_batch_sequences": update.micro_batch_sequences,
                    "training/accumulation_steps": update.accumulation_steps,
                    "training/training_tokens_seen": training_tokens_seen,
                    "training/global_tokens_seen": global_tokens_seen,
                    "training/search_tokens_seen": ARCHITECTURE_EXPANSIONS * schedule_token_count(architecture_schedule),
                    "training/tokens_per_second": tokens_per_second,
                    "training/backbone_grad_norm": grad.backbone_norm,
                    "training/ssonn_grad_norm": grad.ssonn_norm,
                    "training/backbone_clip_coefficient": grad.backbone_clip_coefficient,
                    "training/ssonn_clip_coefficient": grad.ssonn_clip_coefficient,
                    "training/skipped_step": int(grad.skipped),
                    "training/muon_lr": muon_lr,
                    "training/backbone_adam_lr": backbone_lr,
                    "training/ssonn_lr": ssonn_lr,
                    "training/backbone_lr_scale": update.backbone_lr_scale,
                    "training/elapsed_seconds": elapsed,
                    "training/total_params": total_params,
                    "training/nonzero_params": nonzero_params,
                    "training/active_connections": count_ssonn_active_connections(checkpoint_model),
                    "training/sparsity": 1.0 - nonzero_params / max(total_params, 1),
                    "training/sublinear_count": count_ssonn_sublinears(checkpoint_model),
                }, step=global_step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    SSONN_CHECKPOINT,
                    checkpoint_model,
                    optimizer_state_dict(*optimizers),
                    global_step,
                    architecture_checkpoint_metadata("full_training", ARCHITECTURE_EXPANSIONS, update.index + 1, topology_history),
                )

        if (update.index + 1) % CHECKPOINT_INTERVAL == 0 or update.index + 1 == len(full_schedule):
            save_checkpoint(
                latest_path,
                checkpoint_model,
                optimizer_state_dict(*optimizers),
                global_step,
                architecture_checkpoint_metadata("full_training", ARCHITECTURE_EXPANSIONS, update.index + 1, topology_history),
            )

    final_val_loss = evaluate_token_budget(training_model, FINAL_VALIDATION_TOKENS)
    print(
        f"final canonical validation: loss {final_val_loss:.4f} | "
        f"ppl {math.exp(min(final_val_loss, 20.0)):.2f} | {FINAL_VALIDATION_TOKENS:,} target tokens"
    )
    checkpoint_model = getattr(training_model, "_orig_mod", training_model)
    final_metadata = architecture_checkpoint_metadata(
        "full_training_complete", ARCHITECTURE_EXPANSIONS, len(full_schedule), topology_history
    )
    final_metadata["final_validation_loss"] = final_val_loss
    save_checkpoint(latest_path, checkpoint_model, optimizer_state_dict(*optimizers), global_step, final_metadata)
    if experiment is not None:
        experiment.log_metrics({
            "final/val_loss": final_val_loss,
            "final/val_perplexity": math.exp(min(final_val_loss, 20.0)),
            "final/validation_target_tokens": FINAL_VALIDATION_TOKENS,
            "final/validation_scheduled_tokens": validation_scheduled_tokens(FINAL_VALIDATION_TOKENS),
            "final/training_tokens": training_tokens_seen,
            "final/search_tokens": ARCHITECTURE_EXPANSIONS * schedule_token_count(architecture_schedule),
            "final/total_compute_tokens": global_tokens_seen,
        }, step=global_step)
    return model


def summarize_prior_logs(log_root: Path = REPO_ROOT / "logs") -> None:
    if not log_root.exists():
        print("No prior logs found at", log_root)
        return

    def chart_series(path: Path) -> list[tuple[int | float, float]]:
        data = json.loads(path.read_text())
        if not data:
            return []
        xs = data[0].get("x", [])
        ys = data[0].get("y", [])
        return [(x, float(y)) for x, y in zip(xs, ys, strict=False) if isinstance(y, (int, float)) and math.isfinite(y)]

    print("Prior log summary")
    for path in sorted(log_root.rglob("*val_loss*chart_data.json"))[:20]:
        points = chart_series(path)
        if points:
            best_step, best_value = min(points, key=lambda pair: pair[1])
            last_step, last_value = points[-1]
            print(f"{path.name}: best {best_value:.4f} @ {best_step}, last {last_value:.4f} @ {last_step}")


def run_smoke_tests() -> None:
    global NUM_UPDATES, SSONN_CYCLE_STEPS, SSONN_LEARNING_RATE
    NUM_UPDATES = 8
    SSONN_CYCLE_STEPS = 5
    SSONN_LEARNING_RATE = 1e-6
    small = GPTConfig(vocab_size=128, n_layer=2, n_head=2, n_embd=32, max_seq_len=32)
    for kind in ("none", "ssonn", "speedrun_mlp"):
        model = AttentionOnlyGPT(small, mlp_kind=kind).to(device)
        x = torch.randint(0, small.vocab_size, (2, 16), device=device)
        y = torch.randint(0, small.vocab_size, (2, 16), device=device)
        loss = lm_cross_entropy(model(x), y)
        loss.backward()
        assert torch.isfinite(loss)
    model = AttentionOnlyGPT(small, mlp_kind="ssonn").to(device)
    full_model_reinit(model)
    model.eval()
    identity_tokens = torch.randint(0, small.vocab_size, (2, 16), device=device)
    residual_logits = model(identity_tokens)
    for block in model.blocks:
        block.mlp_kind = "none"
    backbone_logits = model(identity_tokens)
    for block in model.blocks:
        block.mlp_kind = "ssonn"
    assert torch.equal(residual_logits, backbone_logits), "Zero-delta SSONN must preserve the backbone function."

    muon, backbone_adam, ssonn_adam = make_ssonn_stage_optimizers(model)
    ssonn_ids = ssonn_parameter_ids(model)
    backbone_ids = {id(p) for group in muon.param_groups + backbone_adam.param_groups for p in group["params"]}
    optimizer_ssonn_ids = {id(p) for group in ssonn_adam.param_groups for p in group["params"]}
    assert not (ssonn_ids & backbone_ids)
    assert ssonn_ids <= optimizer_ssonn_ids
    lrs = [ssonn_cosine_lr(step, SSONN_CYCLE_STEPS) for step in range(SSONN_CYCLE_STEPS)]
    assert lrs[0] == 0.0 and lrs[-1] == 0.0 and max(lrs) > 0.9e-6
    assert cycle_window(0) == WINDOW_START
    assert cycle_window(SSONN_CYCLE_STEPS - 1) == WINDOW_END
    target_tokens = 1_000_000
    inferred_steps = infer_full_pass_cycle_steps(target_tokens)
    assert cycle_token_budget(inferred_steps) >= target_tokens
    assert inferred_steps == 1 or cycle_token_budget(inferred_steps - 1) < target_tokens
    assert has_training_after_step(4, 10)
    assert not has_training_after_step(9, 10)
    schedule = build_update_schedule(917_504_000)
    assert len(schedule) == 1750
    assert schedule[0].window == WINDOW_START and schedule[-1].window == WINDOW_END
    assert schedule[0].ssonn_lr_scale == 0.0 and schedule[-1].ssonn_lr_scale == 0.0
    assert max(update.ssonn_lr_scale for update in schedule) > 0.999
    assert schedule[-1].backbone_lr_scale == 0.0
    assert abs(schedule_token_count(schedule) - 917_504_000) < 2 * TARGET_TOKENS_PER_UPDATE

    metadata_tokens = torch.tensor([[1, EOT_TOKEN_ID, 2, 3, EOT_TOKEN_ID, 4]], device=device)
    document_ids, positions = document_ids_and_positions(metadata_tokens)
    assert document_ids.tolist() == [[0, 1, 1, 1, 2, 2]]
    assert positions.tolist() == [[0, 0, 1, 2, 0, 1]]
    attention_mask = make_document_attention_mask(document_ids)
    assert bool(attention_mask[0, 0, 3, 2])
    assert bool(attention_mask[0, 0, 2, 1])
    assert not bool(attention_mask[0, 0, 1, 0])
    assert not bool(attention_mask[0, 0, 2, 3])

    topology_model = AttentionOnlyGPT(small, mlp_kind="ssonn").to(device)
    first_layer = getattr(topology_model, topology_model.ssonn_layer_names[0])
    first_layer.put_sub_linear(
        torch.tensor([0, 1], device=device),
        torch.tensor([0, 1], device=device),
    )
    topology_before = (first_layer.in_features, len(first_layer.sub_linears))
    full_model_reinit(topology_model)
    topology_after = (first_layer.in_features, len(first_layer.sub_linears))
    assert topology_after == topology_before
    assert torch.isfinite(topology_model(torch.randint(0, small.vocab_size, (2, 16), device=device))).all()

    def write_test_shard(path: Path, values: np.ndarray) -> None:
        raw = np.zeros(512 + len(values), dtype=np.uint16)
        header = raw[:512].view(np.int32)
        header[0] = 20240520
        header[2] = len(values)
        raw[512:] = values
        raw.tofile(path)

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        first = root / "finewebedu_train_000001.bin"
        second = root / "finewebedu_train_000002.bin"
        write_test_shard(first, np.arange(30, dtype=np.uint16))
        write_test_shard(second, np.arange(60, 100, dtype=np.uint16))
        subset_a = TokenShardStream([str(first), str(second)], seed=7, token_limit=35)
        subset_b = TokenShardStream([str(first), str(second)], seed=7, token_limit=35)
        assert subset_a.token_count == 35
        assert sum(subset_a.file_token_limits) == 35
        assert len(subset_a.file_token_limits) == 2
        assert subset_a.slices == subset_b.slices
        a_x, a_y = subset_a.next_batch(4, 2)
        b_x, b_y = subset_b.next_batch(4, 2)
        assert torch.equal(a_x, b_x) and torch.equal(a_y, b_y)
        update_model = AttentionOnlyGPT(small, mlp_kind="ssonn").to(device)
        full_model_reinit(update_model)
        ssonn_train.apply_sparsity_mask(update_model)
        update_optimizers = make_ssonn_stage_optimizers(update_model)
        update = UpdateDescriptor(0, 2, 16, 2, 32, 0.0, 1.0, 0.0)
        update_loss, update_stats = run_scheduled_update(
            update_model,
            TokenShardStream(seed=7, slices=subset_a.slices),
            update,
            update_optimizers,
            None,
            "Smoke scheduled update",
        )
        assert math.isfinite(update_loss) and not update_stats.skipped
        assert math.isfinite(update_stats.backbone_norm) and math.isfinite(update_stats.ssonn_norm)
    print("smoke tests ok")


def main() -> None:
    global NUM_UPDATES, SSONN_CYCLE_STEPS, FULL_TRAINING_STEPS, FULL_TRAINING_PASS_STEPS
    global TRAIN_FILES, VAL_FILES, TRAIN_TOKEN_COUNT, SSONN_CYCLE_TOKEN_COUNT
    global ARCHITECTURE_TOKEN_COUNT, FULL_TRAINING_TOKEN_COUNT, FULL_TRAINING_TARGET_TOKENS
    global ARCHITECTURE_SLICES, FULL_TRAINING_SLICES, VALIDATION_SLICES
    global PERIODIC_VALIDATION_TOKENS, SSONN_EVAL_BATCHES
    set_seed(SEED)
    if ARGS.smoke_test:
        run_smoke_tests()
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if DOWNLOAD_PRETOKENIZED_DATA:
        download_pretokenized_fineweb_edu(PRETOKENIZED_TRAIN_SHARDS)
    TRAIN_FILES, VAL_FILES = find_shards(DATA_DIR)
    if not TRAIN_FILES or not VAL_FILES:
        raise FileNotFoundError(
            f"Missing FineWeb-Edu shards under {DATA_DIR}. Expected filenames containing train_ and val_."
        )
    TRAIN_TOKEN_COUNT = sum(shard_token_count(path) for path in TRAIN_FILES)
    ARCHITECTURE_TOKEN_COUNT = math.floor(TRAIN_TOKEN_COUNT * ARCHITECTURE_FRACTION)
    if ARCHITECTURE_TOKEN_COUNT <= WINDOW_END + 1:
        raise ValueError("Architecture subset is too small for the maximum context window.")
    if TRAINING_PASSES > 0 and FULL_TRAINING_STEPS <= 0:
        FULL_TRAINING_TARGET_TOKENS = TRAINING_PASSES * TRAIN_TOKEN_COUNT
    architecture_schedule = build_update_schedule(ARCHITECTURE_TOKEN_COUNT, SSONN_CYCLE_STEPS)
    full_schedule = build_update_schedule(FULL_TRAINING_TARGET_TOKENS, FULL_TRAINING_STEPS)
    SSONN_CYCLE_STEPS = len(architecture_schedule)
    SSONN_CYCLE_TOKEN_COUNT = schedule_token_count(architecture_schedule)
    FULL_TRAINING_STEPS = len(full_schedule)
    FULL_TRAINING_PASS_STEPS = FULL_TRAINING_STEPS
    FULL_TRAINING_TOKEN_COUNT = schedule_token_count(full_schedule)
    ARCHITECTURE_SLICES = allocate_balanced_slices(TRAIN_FILES, ARCHITECTURE_TOKEN_COUNT, TRAIN_STREAM_SEED)
    full_slice_budget = min(TRAIN_TOKEN_COUNT, FULL_TRAINING_TARGET_TOKENS, FULL_TRAINING_TOKEN_COUNT)
    FULL_TRAINING_SLICES = allocate_balanced_slices(TRAIN_FILES, full_slice_budget, TRAIN_STREAM_SEED + 10)
    validation_available = sum(shard_token_count(path) for path in VAL_FILES)
    if FINAL_VALIDATION_TOKENS > validation_available:
        raise ValueError(
            f"Final validation needs {FINAL_VALIDATION_TOKENS:,} tokens, but validation shards contain {validation_available:,}."
        )
    if ARGS.ssonn_eval_batches is not None:
        PERIODIC_VALIDATION_TOKENS = ARGS.ssonn_eval_batches * MICRO_BATCH_SIZE * VALIDATION_SEQ_LEN
    elif EVAL_BATCHES is not None:
        PERIODIC_VALIDATION_TOKENS = EVAL_BATCHES * MICRO_BATCH_SIZE * VALIDATION_SEQ_LEN
    if PERIODIC_VALIDATION_TOKENS > FINAL_VALIDATION_TOKENS:
        raise ValueError("Periodic validation cannot use more tokens than final validation.")
    SSONN_EVAL_BATCHES = math.ceil(PERIODIC_VALIDATION_TOKENS / (MICRO_BATCH_SIZE * VALIDATION_SEQ_LEN))
    VALIDATION_SLICES = allocate_balanced_slices(VAL_FILES, FINAL_VALIDATION_TOKENS, EVAL_STREAM_SEED)
    NUM_UPDATES = ARCHITECTURE_EXPANSIONS * SSONN_CYCLE_STEPS + FULL_TRAINING_STEPS
    if SSONN_CYCLE_STEPS <= 0 or FULL_TRAINING_STEPS <= 0:
        raise ValueError("Architecture and full-training step counts must be positive.")
    print("data dir:", DATA_DIR)
    print("train shards:", len(TRAIN_FILES), "val shards:", len(VAL_FILES))
    print("first train shard:", TRAIN_FILES[0])
    print("first val shard:", VAL_FILES[0])
    print("train tokens:", TRAIN_TOKEN_COUNT)
    print("run mode:", RUN_MODE, "| mlp kind:", MLP_KIND)
    print(
        "architecture subset:", ARCHITECTURE_TOKEN_COUNT, "tokens |",
        f"{100.0 * ARCHITECTURE_TOKEN_COUNT / TRAIN_TOKEN_COUNT:.6f}% actual |",
        SSONN_CYCLE_STEPS, "updates/cycle |",
        ARCHITECTURE_EXPANSIONS, "expansions",
    )
    print(
        "architecture search:", SSONN_CYCLE_TOKEN_COUNT, "scheduled tokens/cycle |",
        ARCHITECTURE_EXPANSIONS * SSONN_CYCLE_TOKEN_COUNT, "total search tokens",
    )
    print(
        "full training:", FULL_TRAINING_STEPS, "updates |", FULL_TRAINING_TARGET_TOKENS, "target tokens |",
        FULL_TRAINING_TOKEN_COUNT, "scheduled tokens |",
        len(FULL_TRAINING_SLICES), "balanced shard slices",
    )
    print(
        "total compute tokens:", ARCHITECTURE_EXPANSIONS * SSONN_CYCLE_TOKEN_COUNT + FULL_TRAINING_TOKEN_COUNT,
        "| target tokens/update:", TARGET_TOKENS_PER_UPDATE,
        "| cooldown:", COOLDOWN_UPDATES if COOLDOWN_UPDATES is not None else f"{COOLDOWN_FRACTION:.6f} fraction",
    )
    print("total optimizer updates:", NUM_UPDATES)
    print(
        "micro batch sequences at max window:", MICRO_BATCH_SIZE,
        "| activation-token envelope:", MICRO_BATCH_SIZE * WINDOW_END,
        "| dynamic microbatch and accumulation enabled",
    )
    print(
        "validation:", PERIODIC_VALIDATION_TOKENS, "periodic target tokens",
        f"({validation_scheduled_tokens(PERIODIC_VALIDATION_TOKENS)} scheduled) |",
        FINAL_VALIDATION_TOKENS, "final target tokens",
        f"({validation_scheduled_tokens(FINAL_VALIDATION_TOKENS)} scheduled) | context", VALIDATION_SEQ_LEN,
    )
    print("attention backend:", "flex" if USE_FLEX_ATTENTION else "sdpa", "| document-local causal")
    print("comparison note: upstream 3.25 is only an orientation point; this residual SSONN backbone is not full modded NanoGPT.")
    print("amp:", ARGS.amp_dtype, "| grad scaler:", USE_SCALER)

    model = AttentionOnlyGPT(MODEL_CONFIG, mlp_kind=MLP_KIND).to(device)
    print(f"parameters: {count_parameters(model) / 1e6:.1f}M | mlp kind: {MLP_KIND} | ssonn layers: {len(model.ssonn_layer_names)}")
    experiment = create_comet_experiment()
    try:
        train_architecture_first(model, experiment)
    finally:
        if experiment is not None:
            experiment.end()


if __name__ == "__main__":
    main()
