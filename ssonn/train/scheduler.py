from typing import Dict, Any, Optional, List


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
            self.hp.get("prune_after_epochs", 0),
            self.hp.get("min_epochs_between_expansions", 5),
            self.hp.get("plateau_window_size", 5)
        )
        
        if epoch - self.last_action_epoch <= cooldown:
            return None

        # 2. Check for Plateau
        window = self.hp.get("plateau_window_size", 5)
        if len(self.train_losses) < window:
            return None

        recent = self.train_losses[-window:]
        avg_change = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent))) / (window - 1)
        
        # Debug info
        print(f"DEBUG: Epoch {epoch}, Avg Change: {avg_change:.5f}, Threshold: {self.hp.get('plateau_threshold', 0.01)}")

        if avg_change >= self.hp.get("plateau_threshold", 0.01):
            return None # No plateau

        print(f"   -> Plateau detected (change {avg_change:.5f}).")

        # 3. Decision: LR Decay vs Expansion
        current_loss = self.train_losses[-1]
        diff_from_last_plateau = abs(self.last_plateau_loss - current_loss)
        
        # If we haven't acted yet OR the loss has improved significantly since the last plateau action -> Decay LR
        if (self.last_action is None) or (diff_from_last_plateau >= self.hp.get("diff_threshold", 0.05)):
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
