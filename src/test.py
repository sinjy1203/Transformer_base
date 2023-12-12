import warnings
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torchtext import transforms
from torchtext.data.metrics import bleu_score
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
    dm.setup(stage="test")

    ckpt_paths = [ckpt_path for ckpt_path in CKPT_DIR.iterdir()]
    module = TransformerModule.load_from_checkpoint(ckpt_paths[-1])

    outputs = []
    targets = []

    dataloader = dm.test_dataloader()
    for src, tgt in tqdm(dataloader, total=len(dataloader)):
        pred = module.translator(src)

        input_texts = [
            vocab_en.idx2token[token_idx] for token_idx in tgt.squeeze().numpy()
        ]
        outputs += [pred]
        targets += [[input_texts[1:-1]]]

    print("test bleu score: ", bleu_score(outputs, targets))


if __name__ == "__main__":
    main()
