
import torch
from torch.utils.data import Dataset
import pandas as pd


class RapLinesDataset(Dataset):
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)

        # number of rhyme IDs = max ID + 1
        self.num_rhyme_ids = int(self.df["rhyme_id"].max()) + 1

        # compute max syllables for normalization
        syll_col = pd.to_numeric(self.df["syllables"], errors="coerce").astype(float)
        self.max_syllables = float(syll_col.max())

    def __len__(self):
        return len(self.df)

    def parse_to_list(self, s: str):
        s = str(s).strip()
        if s == "":
            return []
        return [int(x) for x in s.split()]

    def __getitem__(self, idx: int):
        prev_str = self.df.iloc[idx]["prev_indices"]
        next_str = self.df.iloc[idx]["next_indices"]
        syll_raw = self.df.iloc[idx]["syllables"]
        rhyme_orig = int(self.df.iloc[idx]["rhyme_id"])

        prev = torch.tensor(self.parse_to_list(prev_str), dtype=torch.long)
        next_l = torch.tensor(self.parse_to_list(next_str), dtype=torch.long)

        if prev.numel() != 30 or next_l.numel() != 30:
            raise ValueError(f"Expected sequence length 30, got prev={prev.numel()}, next={next_l.numel()} at idx {idx}")

        # normalize syllables
        syll = float(syll_raw)
        syll = syll / self.max_syllables
        syll_tensor = torch.tensor(syll, dtype=torch.float32)

        # rhyme ID already 0-based and contiguous
        rhyme_tensor = torch.tensor(rhyme_orig, dtype=torch.long)

        return prev, next_l, syll_tensor, rhyme_tensor


def collate_fn(batch):
    prev_list = [item[0] for item in batch]
    next_list = [item[1] for item in batch]
    syll_list = [item[2] for item in batch]
    rhyme_list = [item[3] for item in batch]

    prev_batch = torch.stack(prev_list)
    next_batch = torch.stack(next_list)
    syll_batch = torch.stack(syll_list)
    rhyme_batch = torch.stack(rhyme_list)

    return prev_batch, next_batch, syll_batch, rhyme_batch
