"""FactoredEdgeFinder — fast edge-metric computation and selection for SSONN.

Computes the same edge scores as EdgeFinder (aggregations over per-batch
weight gradients G_j) without ever materializing per-layer E = n_out * n_in
score buffers or running one finder pass per (layer, mode):

  capture  ONE pass over the finder loader for ALL requested targets; hooks
           store per-batch factors (X = input of the target linear,
           D = grad of its pre-activation output), fp32, GPU when they fit.
           G_j = D_j^T X_j is never formed at full size outside tiles.
  scan     exact target-metric field on a batch subsample, accumulated on
           row tiles from the factors (all four metrics are functions of
           the per-batch moments sum w|G|, sum wG, sum wG^2); with the
           default auto budget the scan field equals the stock score field.
           Top-(c*K) entries -> candidates; bottom entries -> min-witnesses
           for the stock min-max normalization.
  verify   exact stock statistics on the candidates from the stored factors
           (no extra network pass), then the stock rule: min-max normalize,
           > threshold, cap by top-max_to_replace. SKIPPED entirely in the
           exact-scan mode (auto budget covering all batches): there the
           scanned values already ARE the stock metric values.

Equivalence with the stock EdgeFinder selection was verified on live MNIST
training (selection mass and Jaccard = 1.000 across layers, events and
metrics); peak memory scales as O(n_finder * (n_out + n_in)) instead of
O(n_out * n_in) — see experiments/step2_factored/.

Metric semantics (note the stock/paper divergence, kept as-is on purpose):
  'mean'       (1/N) sum_j |G_j|         == stock aggregation_mode='mean'
  'variance'   E_j[G^2] - (E_j|G|)^2     == stock aggregation_mode='variance'
                                            (variance of |G_j|)
  'var_signed' E_j[G^2] - (E_j G)^2      == paper eq. ssonn_var_score
                                            (variance of signed G_j)
  'absmean'    |(1/N) sum_j G_j|         (GradMax-like potential-edge field)

Targets: the main projection weight (embed=False) or the last SubLinear
weight (embed=True). For the embed target X is the SubLinear input and D is
the gradient at its pre-activation output (captured behind its activation).
"""

import math
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ssonn.model.model import ScaledNet

METRICS = ("mean", "variance", "var_signed", "absmean")


class _TargetTap:
    """Captures (X, D) for one target weight of a ScaledNet layer.

    main  target: X = input of the main F.linear (output of the last
          SubLinear when present — identical in eval mode since dropout is
          identity), D = grad of the layer output.
    embed target: X = input of the last SubLinear, D = grad of its
          pre-activation linear output (grad_input of the activation).
    """

    def __init__(self, layer: ScaledNet, embed: bool):
        if embed and len(layer.sub_linears) == 0:
            raise ValueError("embed target requested but layer has no "
                             "sub_linears")
        self.X: Optional[torch.Tensor] = None
        self.D: Optional[torch.Tensor] = None
        self._handles = []
        if embed:
            sub = layer.sub_linears[-1]
            self._handles.append(
                sub.register_forward_pre_hook(self._grab_in))
            self._handles.append(
                sub.act.register_full_backward_hook(self._grab_gin))
        else:
            if len(layer.sub_linears) > 0:
                self._handles.append(
                    layer.sub_linears[-1].register_forward_hook(self._grab_out))
            else:
                self._handles.append(
                    layer.register_forward_pre_hook(self._grab_in))
            self._handles.append(
                layer.register_full_backward_hook(self._grab_gout))

    def _grab_in(self, module, inputs):
        self.X = inputs[0].detach()

    def _grab_out(self, module, inputs, output):
        self.X = output.detach()

    def _grab_gout(self, module, grad_input, grad_output):
        self.D = grad_output[0].detach()

    def _grab_gin(self, module, grad_input, grad_output):
        self.D = grad_input[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()


class _Store:
    """Per-batch factor storage + per-batch energies for one target."""

    def __init__(self, store_device, store_dtype):
        self.Ds: List[torch.Tensor] = []
        self.Xs: List[torch.Tensor] = []
        self.energy: List[float] = []
        self.store_device = store_device
        self.store_dtype = store_dtype

    def add(self, D: torch.Tensor, X: torch.Tensor):
        self.Ds.append(D.to(self.store_device, self.store_dtype))
        self.Xs.append(X.to(self.store_device, self.store_dtype))
        self.energy.append(float(D.float().norm() * X.float().norm()))

    @property
    def n_batches(self):
        return len(self.Ds)


def pick_batches(energies, m_batches: int, seed: int = 0
                 ) -> List[Tuple[int, float]]:
    """Batch subsample: deterministic heavy half + importance draws.

    At full budget (m_batches >= N) every batch enters with weight 1/N and
    the scan field is exactly the stock score field.
    """
    N = len(energies)
    m_batches = min(m_batches, N)
    if m_batches >= N:
        return [(j, 1.0 / N) for j in range(N)]
    e = torch.as_tensor(energies, dtype=torch.float64)
    order = torch.argsort(e, descending=True)
    n_det = min((m_batches + 1) // 2, N)
    picked = [(int(j), 1.0 / N) for j in order[:n_det].tolist()]
    rest = order[n_det:]
    n_draws = m_batches - n_det
    if n_draws >= rest.numel():
        picked += [(int(j), 1.0 / N) for j in rest.tolist()]
    elif n_draws > 0 and rest.numel() > 0:
        q = (e[rest] ** 2).clamp_min(1e-300)
        q = q / q.sum()
        g = torch.Generator().manual_seed(seed)
        draws = torch.multinomial(q, n_draws, replacement=True, generator=g)
        for t in draws.tolist():
            picked.append((int(rest[t]), 1.0 / (N * float(q[t]) * n_draws)))
    return picked


class FactoredEdgeFinder:
    """Drop-in fast replacement for EdgeFinder's expansion selection.

    Single-target parity API:
        choose_edges_threshold(model, layer, threshold, embed=False|True)
    Fast multi-layer API (ONE finder pass for all layers, main targets):
        select(model, {name: layer}, thresholds)   # thresholds: float | dict

    Args mirror the experiments/step2_factored prototype; `proxy_samples`
    None means auto — scan every finder batch while n * E stays under
    `scan_flop_budget` (the scan field is then exactly the stock field).
    """

    def __init__(self,
                 loss_fn: Callable,
                 dataloader,
                 device: Union[torch.device, str] = "cpu",
                 metric: str = "mean",
                 proxy_samples: Optional[int] = None,
                 candidate_factor: Optional[int] = None,
                 scan_flop_budget: float = 8e12,
                 scan_tile_bytes: int = 256 << 20,
                 quantile_sample: int = 4096,
                 min_candidates: int = 1024,
                 store_dtype: torch.dtype = torch.float32,
                 max_to_replace: Optional[int] = None,
                 seed: int = 0):
        assert metric in METRICS, f"metric must be one of {METRICS}"
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.device = device
        self.metric = metric
        self.m = proxy_samples
        self.c = candidate_factor if candidate_factor is not None else (
            4 if metric in ("mean", "absmean") else 8)
        self.scan_flop_budget = scan_flop_budget
        self.scan_tile_bytes = scan_tile_bytes
        self.quantile_sample = quantile_sample
        self.min_candidates = min_candidates
        self.store_dtype = store_dtype
        self.max_to_replace = max_to_replace
        self.seed = seed

    # ------------------------------------------------------------------ API

    def select(self, model: nn.Module, layers: Dict[str, ScaledNet],
               thresholds: Union[float, Dict[str, float]]
               ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, dict]]:
        targets = [(name, layer, False) for name, layer in layers.items()]
        return self._select_targets(model, targets, thresholds)

    def choose_edges_threshold(self, model: nn.Module, layer: ScaledNet,
                               threshold: float, embed: bool = False
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self._select_targets(model, [("t", layer, embed)], threshold)
        parents, children, _ = res["t"]
        return parents, children

    # ------------------------------------------------------------ internals

    @staticmethod
    def _target_weight(layer: ScaledNet, embed: bool) -> torch.Tensor:
        return layer.sub_linears[-1].weight if embed else layer.weight

    @staticmethod
    def _target_mask(layer: ScaledNet, embed: bool) -> Optional[torch.Tensor]:
        owner = layer.sub_linears[-1] if embed else layer
        return owner.sparsity_mask if hasattr(owner, "sparsity_mask") else None

    def _threshold_for(self, thresholds, name):
        t = thresholds[name] if isinstance(thresholds, dict) else thresholds
        assert 0 < t <= 1, "Threshold must be between 0 and 1."
        return t

    def _select_targets(self, model, targets, thresholds):
        t_info = {}
        t0 = time.perf_counter()
        stores = self._capture_pass(model, targets)
        self._sync()
        t_info["capture_pass"] = time.perf_counter() - t0

        K = self.max_to_replace if self.max_to_replace else 4096
        t0 = time.perf_counter()
        cands = {}
        for name, layer, embed in targets:
            w = self._target_weight(layer, embed)
            mask = self._target_mask(layer, embed)
            n_cand = min(self.c * K, w.numel())
            cands[name] = self._scan_candidates(stores[name], w.shape,
                                                n_cand, mask)
        self._sync()
        t_info["candidate_search"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        stats = {}
        to_verify = [(n, l, e) for n, l, e in targets if not cands[n]["exact"]]
        if to_verify:
            verified = self._verify_from_store(to_verify, stores, cands)
            for n, _, _ in to_verify:
                a = verified[n]
                stats[n] = {"idx": a["idx"], "n_top": a["n_top"],
                            "v": self._exact_metric(a)}
        for name, layer, embed in targets:
            if cands[name]["exact"]:
                # exact scan: the collected values are the stock metric
                stats[name] = {
                    "idx": torch.cat([cands[name]["top"], cands[name]["min"]]),
                    "n_top": int(cands[name]["top"].numel()),
                    "v": torch.cat([cands[name]["top_val"],
                                    cands[name]["min_val"]]),
                }
        self._sync()
        t_info["verify"] = time.perf_counter() - t0

        out = {}
        for name, layer, embed in targets:
            thr = self._threshold_for(thresholds, name)
            parents, children, sel_info = self._finalize(
                self._target_weight(layer, embed),
                self._target_mask(layer, embed), stats[name], thr)
            info = dict(sel_info)
            info["timings_sec"] = {k: round(v, 4) for k, v in t_info.items()}
            info["n_batches"] = stores[name].n_batches
            info["verify_skipped"] = bool(cands[name]["exact"])
            out[name] = (parents, children, info)
        return out

    def _sync(self):
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _store_device(self, targets) -> torch.device:
        if not (torch.cuda.is_available()
                and str(self.device).startswith("cuda")):
            return torch.device("cpu")
        try:
            n = len(self.dataloader.dataset)
        except (TypeError, AttributeError):
            n = 65536
        w = sum(self._target_weight(l, e).shape[0]
                + self._target_weight(l, e).shape[1] for _, l, e in targets)
        need = n * w * (2 if self.store_dtype == torch.float16 else 4)
        free, _ = torch.cuda.mem_get_info()
        return torch.device(self.device) if need < 0.5 * free else \
            torch.device("cpu")

    def _capture_pass(self, model, targets) -> Dict[str, _Store]:
        store_dev = self._store_device(targets)
        taps = {n: _TargetTap(l, e) for n, l, e in targets}
        stores = {n: _Store(store_dev, self.store_dtype) for n, _, _ in targets}
        model.eval()
        try:
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                model.zero_grad()
                y = model(data).squeeze()
                loss = self.loss_fn(y, target)
                loss.backward()
                for n, _, _ in targets:
                    stores[n].add(taps[n].D, taps[n].X)
        finally:
            for tap in taps.values():
                tap.remove()
            model.zero_grad()
        return stores

    def _effective_m(self, store: _Store, E: int) -> int:
        if self.m is not None:
            return self.m
        n_total = sum(d.shape[0] for d in store.Ds)
        return min(n_total, max(8192, int(self.scan_flop_budget // max(E, 1))))

    def _scan_candidates(self, store: _Store, w_shape, n_cand: int,
                         mask: Optional[torch.Tensor]
                         ) -> Dict[str, torch.Tensor]:
        """Batch-subsampled exact-metric scan -> top + bottom candidates."""
        n_out, n_in = w_shape
        batch = store.Ds[0].shape[0] if store.Ds else 1
        eff_m = self._effective_m(store, n_out * n_in)
        m_b = max(1, math.ceil(eff_m / max(batch, 1)))
        picked = pick_batches(store.energy, m_b, seed=self.seed)
        staged = [(store.Ds[j].to(self.device).float(),
                   store.Xs[j].to(self.device).float(), w)
                  for j, w in picked]

        n_min = min(self.min_candidates, n_out * n_in)
        tile = max(1, min(n_out, self.scan_tile_bytes // max(16 * n_in, 1)))
        top_v, top_i, bot_v, bot_i = [], [], [], []
        for r0 in range(0, n_out, tile):
            r1 = min(r0 + tile, n_out)
            acc2 = torch.zeros(r1 - r0, n_in, device=self.device)
            acc1 = torch.zeros(r1 - r0, n_in, device=self.device)
            acc_abs = torch.zeros(r1 - r0, n_in, device=self.device)
            for Dj, Xj, w in staged:
                G = Dj[:, r0:r1].T @ Xj
                acc2 += w * (G * G)
                acc1 += w * G
                acc_abs += w * G.abs()
            if self.metric == "variance":
                v = acc2 - acc_abs * acc_abs
            elif self.metric == "var_signed":
                v = acc2 - acc1 * acc1
            elif self.metric == "absmean":
                v = acc1.abs()
            else:
                v = acc_abs
            if mask is not None:
                v = v * mask[r0:r1]
            v = v.ravel()
            tv, ti = torch.topk(v, min(n_cand, v.numel()))
            top_v.append(tv)
            top_i.append(ti + r0 * n_in)
            bv, bi = torch.topk(v, min(n_min, v.numel()), largest=False)
            bot_v.append(bv)
            bot_i.append(bi + r0 * n_in)
        vals = torch.cat(top_v)
        idx = torch.cat(top_i)
        tv, order = torch.topk(vals, min(n_cand, vals.numel()))
        top, top_val = idx[order], tv
        vals = torch.cat(bot_v)
        idx = torch.cat(bot_i)
        bv, order = torch.topk(vals, min(n_min, vals.numel()), largest=False)
        # exact=True <=> the scan covered every batch with deterministic 1/N
        # weights, i.e. the scanned values ARE the stock metric values —
        # verification would recompute the same numbers and is skipped
        return {"top": top, "top_val": top_val,
                "min": idx[order], "min_val": bv,
                "exact": m_b >= len(store.energy)}

    def _verify_from_store(self, targets, stores, cands) -> Dict[str, dict]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 1)
        aux = {}
        for name, layer, embed in targets:
            w = self._target_weight(layer, embed)
            E = w.numel()
            n_samp = min(self.quantile_sample, E)
            samp = torch.randint(0, E, (n_samp,), generator=gen).to(self.device)
            idx = torch.cat([cands[name]["top"], cands[name]["min"], samp])
            o = idx // w.shape[1]
            i = idx % w.shape[1]
            z = torch.zeros(idx.numel(), device=self.device)
            a = {"idx": idx, "n_top": int(cands[name]["top"].numel()),
                 "sum_abs": z.clone(), "sum_g": z.clone(),
                 "sum_g2": z.clone(), "n_batches": stores[name].n_batches}
            for d, x in zip(stores[name].Ds, stores[name].Xs):
                dj = d.to(self.device).float()
                xj = x.to(self.device).float()
                g = (dj[:, o] * xj[:, i]).sum(0)
                a["sum_abs"] += g.abs()
                a["sum_g"] += g
                a["sum_g2"] += g * g
            aux[name] = a
        return aux

    def _exact_metric(self, a: dict) -> torch.Tensor:
        N = max(a["n_batches"], 1)
        if self.metric == "mean":
            return a["sum_abs"] / N
        if self.metric == "absmean":
            return (a["sum_g"] / N).abs()
        if self.metric == "variance":
            m = a["sum_abs"] / N
            return a["sum_g2"] / N - m * m
        m = a["sum_g"] / N
        return a["sum_g2"] / N - m * m

    def _finalize(self, weight, mask, a, threshold):
        v = a["v"]
        if mask is not None:
            # idempotent for the exact-scan path (already masked there)
            v = v * mask.ravel()[a["idx"]]

        n_c = a["n_top"]
        cand = a["idx"][:n_c]
        v_cand, v_all = v[:n_c], v
        vmax = v_all.max()
        vmin = v_all.min()
        if mask is not None and (mask == 0).any():
            vmin = torch.minimum(vmin, torch.zeros_like(vmin))
        norm = (v_cand - vmin) / (vmax - vmin + 1e-8)
        keep = norm > threshold
        idx = cand[keep]
        vals = v_cand[keep]
        if self.max_to_replace is not None and idx.numel() > self.max_to_replace:
            _, order = torch.topk(vals, self.max_to_replace)
            idx = idx[order]
            vals = vals[order]

        n_in = weight.shape[1]
        children = idx // n_in
        parents = idx % n_in
        boundary_gap = float(
            (vals.min() / (v_cand.min() + 1e-30)).item()) if vals.numel() else None
        info = {"n_selected": int(idx.numel()),
                "vmin_est": float(vmin), "vmax_est": float(vmax),
                "boundary_gap": boundary_gap}
        return parents, children, info
