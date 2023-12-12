from pathlib import Path

# DATA_DIR = Path("/media/shin/T7/nmt_data/wmt16_de_en")

# TRAIN_EN_PATH = DATA_DIR / "train.tok.clean.bpe.32000.en"
# TRAIN_DE_PATH = DATA_DIR / "train.tok.clean.bpe.32000.de"
# TEST_EN_PATHS = DATA_DIR.glob("newstest*.tok.clean.bpe.32000.en")
# TEST_DE_PATHS = DATA_DIR.glob("newstest*.tok.clean.bpe.32000.de")
# TEST_EN_PATH = DATA_DIR / "test.tok.clean.bpe.32000.en"
# TEST_DE_PATH = DATA_DIR / "test.tok.clean.bpe.32000.de"


# VOCAB_PATH = DATA_DIR / "vocab.bpe.32000"
# VOCAB_EN_DE_PATH = DATA_DIR / "vocab.en.de.pickle"

DATA_DIR = Path("/media/shin/T7/nmt_data/multi30k")
VOCAB_EN_PATH = DATA_DIR / "vocab_en.txt"
VOCAB_DE_PATH = DATA_DIR / "vocab_de.txt"

CKPT_DIR = Path("./checkpoints")
