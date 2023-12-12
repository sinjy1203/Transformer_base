import warnings
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

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
        src_transform=transform_de, tgt_transform=transform_en, **cfg.datamodule
    )
    dm.setup(stage="fit")

    module = TransformerModule(
        vocab_de_size=len(vocab_de),
        vocab_en_size=len(vocab_en),
        vocab_de_idx2token=vocab_de.idx2token,
        vocab_en_idx2token=vocab_en.idx2token,
        pad_idx=SPECIAL_TOKENS_IDX["PAD"],
        device="cuda",
        **cfg.model
    )
    logger = WandbLogger(project="Transformer-Base", entity="sinjy1203")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/",
            filename="Transformer-Base-{epoch:02d}-{val_loss:.4f}",
            save_top_k=5,
            mode="min",
        ),
    ]

    trainer = Trainer(logger=logger, callbacks=callbacks, devices=1, **cfg.trainer)
    trainer.fit(module, dm)

    wandb.finish()


if __name__ == "__main__":
    main()
