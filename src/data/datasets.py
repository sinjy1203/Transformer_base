from torch.utils.data import Dataset
from torchtext.datasets import Multi30k
from torchtext import transforms
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras.constants import *

# class WMTDataset(Dataset):
#     def __init__(self, en_path, de_path, transform=None):
#         self.en_lines = load_data(en_path)
#         self.de_lines = load_data(de_path)
#         assert len(self.en_lines) == len(self.de_lines)

#         self.transform = transform

#     def __len__(self):
#         return len(self.en_lines)

#     def __getitem__(self, idx):
#         en_line = self.en_lines[idx]
#         de_line = self.de_lines[idx]

#         en_line = en_line.split()
#         de_line = de_line.split()

#         if self.transform:
#             en_ids, de_ids = self.transform(en_line), self.transform(de_line)

#         return en_ids, de_ids


class Multi30kDataset(Dataset):
    def __init__(self, split, src_transform=None, tgt_transform=None):
        dataset = Multi30k(split=split, language_pair=("de", "en"))
        src_texts, tgt_texts = list(zip(*dataset))
        self.src_texts, self.tgt_texts = [], []
        for i in range(len(src_texts)):
            if src_texts[i] and tgt_texts[i]:
                self.src_texts.append(src_texts[i])
                self.tgt_texts.append(tgt_texts[i])

        self.src_transform = src_transform
        self.tgt_transform = tgt_transform

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        if self.src_transform:
            src_text = self.src_transform(src_text)

        if self.tgt_transform:
            tgt_text = self.tgt_transform(tgt_text)

        return src_text, tgt_text
