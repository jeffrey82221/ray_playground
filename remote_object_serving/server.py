import ray

@ray.remote
class RemoteServer:
    def __init__(self):
        self.storage = dict()
    def store(self, key, value):
        self.storage[key] = value
    def get_value(self, key):
        return self.storage[key]

storage_server = RemoteServer.remote()

if __name__ == '__main__':
    # Let the Server Run forever
    while True:
        pass
