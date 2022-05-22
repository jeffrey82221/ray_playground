
from load_data import load_data
from torch.utils.data import random_split
import torch
class CiFarModule:
    def __init__(self, data_dir=None, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        trainset, testset = load_data(self.data_dir)
        test_abs = int(len(trainset) * 0.8)
        self.train_subset, self.val_subset = random_split(
            trainset, [test_abs, len(trainset) - test_abs])
        
    def train_dataloader(self, batch_size):
        trainloader = torch.utils.data.DataLoader(
            self.train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1)
        return trainloader

    def val_dataloader(self, batch_size):
        valloader = torch.utils.data.DataLoader(
            self.val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1)
        return valloader
