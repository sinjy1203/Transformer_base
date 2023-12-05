import pickle
from torchtext.vocab import vocab
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras.paths import *
from extras.constants import *


def load_data(path):
    with open(path, "r") as f:
        data = [line.rstrip() for line in f]
    return data


def get_vocab():
    if VOCAB_EN_DE_PATH.exists():
        with open(VOCAB_EN_DE_PATH, "rb") as f:
            vocab_en_de = pickle.load(f)
    else:
        with open(VOCAB_PATH, "r") as f:
            vocab_en_de = {token.rstrip(): 1 for token in f}
        vocab_en_de = vocab(vocab_en_de, specials=SPECIAL_TOKENS.keys())
        with open(VOCAB_EN_DE_PATH, "wb") as f:
            pickle.dump(vocab_en_de, f)

    return vocab_en_de
