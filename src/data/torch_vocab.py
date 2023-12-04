from torchtext.vocab import vocab
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras.paths import *
from extras.constants import *


def build_vocab():
    with open(VOCAB_PATH, "r") as f:
        vocab_en_de = {token.rstrip(): 1 for token in f}
    vocab_en_de = vocab(vocab_en_de, specials=SPECIAL_TOKENS)
    return vocab_en_de


if __name__ == "__main__":
    vocab_en_de = build_vocab()
    print(len(vocab_en_de))
