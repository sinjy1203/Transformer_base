from torch.utils.data import Dataset
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.utils import load_data


class EnDeDataset(Dataset):
    def __init__(self, en_path, de_path, transform=None):
        self.en_lines = load_data(en_path)
        self.de_lines = load_data(de_path)
        assert len(self.en_lines) == len(self.de_lines)

        self.transform = transform

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        en_line = self.en_lines[idx]
        de_line = self.de_lines[idx]

        en_line = en_line.split()
        de_line = de_line.split()

        if self.transform:
            en_ids, de_ids = self.transform(en_line), self.transform(de_line)

        return en_ids, de_ids
