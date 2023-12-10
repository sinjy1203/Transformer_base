import pickle
from typing import Any
from tqdm import tqdm
import spacy
from collections import Counter
from torch.nn import Module
from torchtext.datasets import Multi30k
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras.paths import *
from extras.constants import *


# def load_data(path):
#     with open(path, "r") as f:
#         data = [line.rstrip() for line in f]
#     return data


# def get_vocab():
#     if VOCAB_EN_DE_PATH.exists():
#         with open(VOCAB_EN_DE_PATH, "rb") as f:
#             vocab_en_de = pickle.load(f)
#     else:
#         with open(VOCAB_PATH, "r") as f:
#             vocab_en_de = {token.rstrip(): 1 for token in f}
#         vocab_en_de = vocab(vocab_en_de, specials=SPECIAL_TOKENS.keys())
#         with open(VOCAB_EN_DE_PATH, "wb") as f:
#             pickle.dump(vocab_en_de, f)

#     return vocab_en_de


class VocabTransform(Module):
    def __init__(self, vocab_path, language):
        self.vocab = VocabTransform.load_vocab(vocab_path, language)
        self.lookup = {token: i for i, token in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def __call__(self, tokens):
        return [self.lookup.get(token, SPECIAL_TOKENS_IDX["UNK"]) for token in tokens]

    @staticmethod
    def load_vocab(vocab_path, language):
        if vocab_path.exists():
            with vocab_path.open(mode="r") as f:
                vocab = f.read().splitlines()
        else:
            vocab = VocabTransform.build_vocab(vocab_path, language)

        return vocab

    @staticmethod
    def build_vocab(vocab_path: Path, language):
        datasets = Multi30k(split="train", language_pair=("de", "en"))
        src_texts, tgt_texts = list(zip(*datasets))

        if language == "de":
            tokenizer = spacy.load("de_core_news_sm")
            texts = src_texts
        else:
            tokenizer = spacy.load("en_core_web_sm")
            texts = tgt_texts

        counter = Counter()
        for doc in tqdm(tokenizer.pipe(texts), total=len(texts)):
            token_texts = []
            for token in doc:
                token_text = token.text.strip()
                if len(token_text) > 0:
                    token_texts.append(token_text)
            counter.update(token_texts)

        vocab = [
            SPECIAL_TOKENS[special_token] for special_token in SPECIAL_TOKENS_ORDER
        ]
        for token, count in counter.most_common():
            vocab += [token]

        if vocab_path.parent.exists() is False:
            vocab_path.parent.mkdir(parents=True)

        with vocab_path.open(mode="w") as f:
            f.write("\n".join(vocab))
        return vocab


class Tokenizer(Module):
    def __init__(self, language):
        if language == "en":
            self.tokenizer = spacy.load("en_core_web_sm")
        else:
            self.tokenizer = spacy.load("de_core_news_sm")

    def __call__(self, text) -> Any:
        return [token.text for token in self.tokenizer(text)]
