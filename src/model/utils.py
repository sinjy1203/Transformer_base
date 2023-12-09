from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps=4000, d_model=512):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer)

    def get_lr(self):
        lr = self.d_model ** (-0.5) * min(
            self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5)
        )
        return [lr] * self.num_param_groups


class TranslationLoss(nn.Module):
    def __init__(self, pad_idx, vocab_size, label_smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_idx, label_smoothing=label_smoothing
        )

    def forward(self, logits, tgt):
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        return loss
