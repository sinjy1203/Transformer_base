import warnings
import wandb
import torch
from torchtext import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model.lightning_modules import TransformerModule
from src.data.datamodules import Multi30kDataModule
from src.data.utils import VocabTransform, Tokenizer
from extras.paths import *
from extras.constants import *

warnings.filterwarnings("ignore")


def main():
    vocab_de = VocabTransform(VOCAB_DE_PATH, language="de")
    vocab_en = VocabTransform(VOCAB_EN_PATH, language="en")

    transform_de = transforms.Sequential(*[Tokenizer(language="de"), vocab_de])
    transform_en = transforms.Sequential(
        *(
            [Tokenizer(language="en"), vocab_en]
            + [
                transforms.AddToken(token=SPECIAL_TOKENS_IDX["SOS"], begin=True),
                transforms.AddToken(token=SPECIAL_TOKENS_IDX["EOS"], begin=False),
            ]
        )
    )

    dm = Multi30kDataModule(
        batch_size=32,
        num_workers=4,
        src_transform=transform_de,
        tgt_transform=transform_en,
    )
    dm.setup(stage="fit")

    module = TransformerModule(
        vocab_de_size=len(vocab_de),
        vocab_en_size=len(vocab_en),
        pad_idx=SPECIAL_TOKENS_IDX["PAD"],
        warmup_steps=10000,
        label_smoothing=0.1,
        d_model=128,
        max_seq_len=5000,
        h=8,
        d_ff=512,
        p_drop=0.3,
        N=2,
        betas=(0.9, 0.98),
        eps=1e-9,
        device="cuda",
    )
    logger = WandbLogger(project="Transformer-Base", entity="sinjy1203")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/",
            filename="Transformer-Base-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer = Trainer(max_epochs=20, logger=logger, callbacks=callbacks, devices=1)
    trainer.fit(module, dm)

    # trainer.test(datamodule=dm)
    wandb.finish()


if __name__ == "__main__":
    main()
