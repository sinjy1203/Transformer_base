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
from src.data.datamodules import EnDeDataModule
from src.data.utils import get_vocab
from extras.paths import *
from extras.constants import *


def main():
    vocab_en_de = get_vocab()
    transform = transforms.Sequential(
        transforms.VocabTransform(vocab_en_de),
        transforms.Truncate(max_seq_len=MAX_SEQ_LEN - 2),
        transforms.AddToken(
            token=vocab_en_de.__getitem__(SPECIAL_TOKENS["start"]), begin=True
        ),
        transforms.AddToken(
            token=vocab_en_de.__getitem__(SPECIAL_TOKENS["end"]), begin=False
        ),
        transforms.ToTensor(),
        transforms.PadTransform(
            max_length=MAX_SEQ_LEN,
            pad_value=vocab_en_de.__getitem__(SPECIAL_TOKENS["pad"]),
        ),
    )
    dm = EnDeDataModule(
        train_en_path=TRAIN_EN_PATH,
        train_de_path=TRAIN_DE_PATH,
        test_en_path=TEST_EN_PATH,
        test_de_path=TEST_DE_PATH,
        train_ratio=0.8,
        batch_size=32,
        num_workers=4,
        transform=transform,
    )
    dm.setup()

    module = TransformerModule(
        vocab_size=len(vocab_en_de),
        pad_idx=vocab_en_de.__getitem__(SPECIAL_TOKENS["pad"]),
        warmup_steps=6000,
        label_smoothing=0.1,
        d_model=512,
        max_seq_len=256,
        h=8,
        d_ff=2048,
        p_drop=0.1,
        N=6,
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

    trainer = Trainer(max_steps=1000000, logger=logger, callbacks=callbacks, devices=1)
    trainer.fit(module, dm)

    trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
