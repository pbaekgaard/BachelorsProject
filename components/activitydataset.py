import torch
from torch.utils.data import Dataset


class ActivityDataset(Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        sequence, label = self.sequence[idx]
        return dict(sequence=torch.Tensor(sequence.to_numpy()), label=torch.tensor(label).long())
