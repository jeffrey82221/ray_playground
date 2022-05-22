import ray
import uuid

class SharableAdaptor:
    @staticmethod
    def convert(dataloader_cls):
        class SharableDataLoader(dataloader_cls):
            def __init__(self, *args, **kargs):
                self.remote_data_loader = RemoteDataLoader.remote(
                    dataloader_cls, init_args = args, init_kargs=kargs
                )
            def get_client(self):
                return ClientDataLoader(self.remote_data_loader)
        return SharableDataLoader
    
@ray.remote
class RemoteDataLoader:
    def __init__(self, DataloaderClass, init_args=[], init_kargs=dict()):
        # Download training data from open datasets.
        self.__DataloaderClass = DataloaderClass
        self.__init_args = init_args
        self.__init_kargs = init_kargs
        self.__dataloaders = dict()

    def build_connection(self):
        client_id = uuid.uuid4()
        self.__dataloaders[client_id] = iter(
            self.__DataloaderClass(*self.__init_args, **self.__init_kargs)
        )
        return client_id

    def close_connection(self, client_id):
        assert client_id in self.__dataloaders.keys()
        del self.__dataloaders[client_id]

    def get_next_batch(self, client_id):
        return next(self.__dataloaders[client_id])

class ClientDataLoader:
    def __init__(self, shared_dataloader):
        self.shared_dataloader = shared_dataloader
        self.client_id = ray.get(self.shared_dataloader.build_connection.remote())
    def __iter__(self):
        while True:
            yield ray.get(self.shared_dataloader.get_next_batch.remote(self.client_id))
    def close(self):
        self.shared_dataloader.close_connection.remote(self.client_id)
