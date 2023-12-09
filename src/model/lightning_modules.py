import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model.models import Transformer
from src.model.utils import LearningRateWarmupScheduler, TranslationLoss


class TransformerModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        warmup_steps=4000,
        label_smoothing=0.1,
        d_model=512,
        max_seq_len=256,
        h=8,
        d_ff=2048,
        p_drop=0.1,
        N=6,
        betas=(0.9, 0.98),
        eps=1e-9,
        device="cuda",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = Transformer(
            vocab_size,
            d_model,
            max_seq_len,
            h,
            d_ff,
            p_drop,
            N,
            device=torch.device(device),
        )
        self.loss_fn = TranslationLoss(pad_idx, vocab_size, label_smoothing)

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(logits, tgt[:, 1:])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        logits = self(src, tgt[:, :-1])

        loss = self.loss_fn(logits, tgt[:, 1:])

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        src, tgt = batch

        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(logits, tgt[:, 1:])

        self.log("test_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), betas=self.hparams.betas, eps=self.hparams.eps
        )
        scheduler = {
            "scheduler": LearningRateWarmupScheduler(
                optimizer,
                warmup_steps=self.hparams.warmup_steps,
                d_model=self.hparams.d_model,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
