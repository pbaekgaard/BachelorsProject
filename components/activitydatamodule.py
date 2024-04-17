from torch.utils.data import DataLoader
import pytorch_lightning as pl
from components.activitydataset import ActivityDataset
from multiprocessing import cpu_count


class ActivityDataModule(pl.LightningDataModule):
    def __init__(self, train_sequence, test_sequence, batch_size):
        super().__init__()
        self.train_sequence = train_sequence
        self.test_sequence = test_sequence
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_data = ActivityDataset(self.train_sequence)
        self.test_data = ActivityDataset(self.test_sequence)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=cpu_count())

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())
