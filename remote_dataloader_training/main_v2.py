"""
Here is a DEMO of using the same remote dataloader producer
with multiple different remote consumer 

Might Encouter Collision Problem!!
"""
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataloader_adaptor import SharableAdaptor
import ray


import logging
@ray.remote
class Consumer:
    def __init__(self, client_dataloader):
        self.dataloader = client_dataloader
    def run(self):
        for i, batch in enumerate(dataloader):
            logging.info(f'i: {i}')
            logging.info(batch)
            if i > 10:
                break
ray.init()
batch_size = 10
client_count = 2
dataset = datasets.FashionMNIST(
        root="~/data",
        train=True,
        download=True,
        transform=ToTensor(),
)

sharable_dataloader = SharableAdaptor.convert(DataLoader)(dataset, batch_size = batch_size)
del dataset
dataloader_1 = sharable_dataloader.get_client()
dataloader_2 = sharable_dataloader.get_client()
print('client dataloaders built')
consumer_1 = Consumer.remote(dataloader_1)
consumer_2 = Consumer.remote(dataloader_2)
print('consumer built')
consumer_1.run.remote()
print('consumer_1 run finished')
consumer_2.run.remote()
print('consumer_2 run finished')
dataloader_1.close()
dataloader_2.close()
ray.shutdown()

