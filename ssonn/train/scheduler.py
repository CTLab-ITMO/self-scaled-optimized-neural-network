from typing import Dict, Any, Optional, List

class SSONNWarmupScheduler:
    """
    Linearly warms up learning rate for a specific parameter group.
    """
    def __init__(self, optimizer, target_lr, warmup_epochs, param_group_idx=-1):
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.group_idx = param_group_idx
        self.current_epoch = 0

        # Сразу зануляем LR для новой группы
        if 0 <= self.group_idx < len(self.optimizer.param_groups):
            self.optimizer.param_groups[self.group_idx]['lr'] = 0.0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            self.current_epoch += 1
            ratio = self.current_epoch / self.warmup_epochs
            new_lr = self.target_lr * ratio

            if 0 <= self.group_idx < len(self.optimizer.param_groups):
                self.optimizer.param_groups[self.group_idx]['lr'] = new_lr

class SSONNPlateauScheduler:
    """
    Manages the decision logic between Learning Rate decay and Model Expansion.
    """
    def __init__(self, hyperparams: Dict[str, Any]):
        self.hp = hyperparams
        
        # State
        self.last_plateau_loss = float('inf')
        self.last_action: Optional[str] = None
        self.last_action_epoch = 0
        self.train_losses: List[float] = []

    def update_metrics(self, train_loss: float):
        self.train_losses.append(train_loss)

    def check_for_action(self, epoch: int) -> Optional[str]:
        """
        Determines the next action: 'lr', 'expand', or None.
        """
        # 1. Check cooldown period
        cooldown = max(
            self.hp["prune_after_epochs"],
            self.hp["min_epochs_between_expansions"],
            self.hp["plateau_window_size"]
        )
        
        if epoch - self.last_action_epoch <= cooldown:
            return None

        # 2. Check for Plateau
        window = self.hp["plateau_window_size"]
        if len(self.train_losses) < window:
            return None

        recent = self.train_losses[-window:]
        avg_change = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent))) / (window - 1)
        
        # Debug info
        print(f"DEBUG: Epoch {epoch}, Avg Change: {avg_change:.5f}, Threshold: {self.hp['plateau_threshold']}")

        if avg_change >= self.hp["plateau_threshold"]:
            return None # No plateau

        print(f"   -> Plateau detected (change {avg_change:.5f}).")

        # 3. Decision: LR Decay vs Expansion
        current_loss = self.train_losses[-1]
        diff_from_last_plateau = abs(self.last_plateau_loss - current_loss)
        
        # If we haven't acted yet OR the loss has improved significantly since the last plateau action -> Decay LR
        if (self.last_action is None) or (diff_from_last_plateau >= self.hp["diff_threshold"]):
            self.last_plateau_loss = current_loss
            self.last_action = 'lr'
            self.last_action_epoch = epoch
            return 'lr'
        
        # If loss hasn't improved much despite previous actions -> Expand capacity
        else:
            self.last_action = 'expand'
            self.last_action_epoch = epoch
            # Note: We do NOT update last_plateau_loss here, typically we want to see if expansion breaks it
            return 'expand'
