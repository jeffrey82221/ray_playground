"""
Here is a DEMO of using the same remote dataloader producer
with multiple different remote consumer 

Might Encouter Collision Problem!!
"""
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import ray
import uuid
@ray.remote
class RemoteDataLoader:
    def __init__(self, worker_batch_size, consumer_count=1):
        # Download training data from open datasets.
        self.dataset = datasets.FashionMNIST(
            root="~/data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        self.worker_batch_size = worker_batch_size
        self.dataloaders = {}

    def build_connection(self):
        client_id = uuid.uuid4()
        self.dataloaders[client_id] = iter(DataLoader(self.dataset, batch_size=self.worker_batch_size))
        return client_id

    def close_connection(self, client_id):
        assert client_id in self.dataloaders.keys()
        del self.dataloaders[client_id]

    def get_next_batch(self, client_id):
        return next(self.dataloaders[client_id])

class ClientDataLoader:
    def __init__(self, shared_dataloader):
        self.shared_dataloader = shared_dataloader
        self.client_id = ray.get(self.shared_dataloader.build_connection.remote())
    def __iter__(self):
        while True:
            yield ray.get(self.shared_dataloader.get_next_batch.remote(self.client_id))
    def close(self):
        self.shared_dataloader.close_connection.remote(self.client_id)

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
remote_data_loader = RemoteDataLoader.remote(batch_size, client_count)
dataloader_1 = ClientDataLoader(remote_data_loader)
dataloader_2 = ClientDataLoader(remote_data_loader)
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

