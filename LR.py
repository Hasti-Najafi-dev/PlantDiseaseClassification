from torch.optim import lr_scheduler


class LearningRate:
    """
    LR scheduler wrapper (stepped once per epoch). EVERY strategy starts with a
    linear warmup, then hands over to the chosen scheduler.

    strategy:
        'step'        -> StepLR            (step_size=30, gamma=0.1)
        'cosine'      -> CosineAnnealingLR (needs T_max or epochs)
        'exponential' -> ExponentialLR     (gamma=0.95)
        'plateau'     -> ReduceLROnPlateau (patience=10, factor=0.5, mode='min');
                         step() needs metric (ignored during warmup)

    Common kwargs:
        warmup_epochs       (default 3)    number of warmup epochs; 0 disables warmup
        warmup_start_factor (default 0.01) LR at epoch 0 = base_lr * factor
        T_max / epochs      total training epochs, INCLUDING warmup (cosine only)
        eta_min             (default 0)    final LR of cosine
    """

    STRATEGIES = ('step', 'cosine', 'exponential', 'plateau')

    def __init__(self, optimizer, strategy='step', **kwargs):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")
        self.optimizer = optimizer
        self.strategy = strategy
        self.kwargs = kwargs
        self.warmup_epochs = int(kwargs.get('warmup_epochs', 3))
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")

        self._epoch = 0
        self.warmup_scheduler = None
        self.scheduler = None

        if self.strategy == 'cosine':
            self._validate_cosine()  # fail early, not after warmup

        if self.warmup_epochs > 0:
            self.warmup_scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=kwargs.get('warmup_start_factor', 0.01),
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
        else:
            self.scheduler = self._build_main_scheduler()

    @property
    def in_warmup(self):
        return self.warmup_scheduler is not None and self.scheduler is None

    def _total_epochs(self):
        total = self.kwargs.get('T_max', self.kwargs.get('epochs'))
        if total is None:
            raise ValueError("Strategy 'cosine' needs 'T_max' or 'epochs' (total number of training epochs).")
        return int(total)

    def _validate_cosine(self):
        total = self._total_epochs()
        if total <= self.warmup_epochs:
            raise ValueError(f"Total epochs ({total}) must be larger than warmup_epochs ({self.warmup_epochs})")

    def _build_main_scheduler(self):
        """Built right after warmup, when the LR is back at its base value."""
        if self.strategy == 'step':
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.kwargs.get('step_size', 30),
                gamma=self.kwargs.get('gamma', 0.1),
            )
        if self.strategy == 'cosine':
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self._total_epochs() - self.warmup_epochs,
                eta_min=self.kwargs.get('eta_min', 0),
            )
        if self.strategy == 'exponential':
            return lr_scheduler.ExponentialLR(self.optimizer, gamma=self.kwargs.get('gamma', 0.95))
        # plateau
        return lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.kwargs.get('mode', 'min'),
            patience=self.kwargs.get('patience', 10),
            factor=self.kwargs.get('factor', 0.5),
        )

    def step(self, epoch=None, metric=None):
        if self.in_warmup:
            self.warmup_scheduler.step()
            self._epoch += 1
            if self._epoch >= self.warmup_epochs:
                for group in self.optimizer.param_groups:
                    group['lr'] = group['initial_lr']
                self.scheduler = self._build_main_scheduler()
            return

        if self.strategy == 'plateau':
            if metric is None:
                raise ValueError("The 'plateau' strategy requires step(metric=val_loss)")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]