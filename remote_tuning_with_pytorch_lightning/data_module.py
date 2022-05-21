import pytorch_lightning as pl
from torchvision.datasets import MNIST
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=None, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        with FileLock(os.path.expanduser("~/.data.lock")):
            return MNIST(data_dir, train=True, download=True, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)
        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=int(self.batch_size))

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=int(self.batch_size))

