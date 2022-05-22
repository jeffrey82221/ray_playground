import ray
import uuid
class Sender:
    def __init__(self):
        from server import storage_server
        self.remote_storage = storage_server
    def send_message(self, msg):
        session_id = uuid.uuid4()
        self.remote_storage.store.remote(session_id, msg)
        return session_id
    
class Reciever:
    def __init__(self):
        from server import storage_server
        self.remote_storage = storage_server
    def recieve_message(self, session_id):
        msg = ray.get(
            self.remote_storage.get_value.remote(session_id)
        )
        return msg
import time
a_large_message = "".join(['g']*100000)
iter_count = 1000
start = time.process_time()
sender = Sender()
reciever = Reciever()
end = time.process_time()
print('Create Time of Sender/Reciever', end - start)
start = time.process_time()
for i in range(iter_count):
    ans = a_large_message
    # print(ans)
end = time.process_time()
print('Creation of 1000 messages', end - start)
start = time.process_time()
for i in range(iter_count):
    session_id = sender.send_message(a_large_message)
    ans = reciever.recieve_message(session_id)
    # print(ans)
end = time.process_time()
print('Communication Time of 1000 messages', end - start)
import pickle
start = time.process_time()
for i in range(iter_count):
    with open(f'{i}.pickle', 'wb') as f:
        pickle.dump(a_large_message, f)
    with open(f'{i}.pickle', 'rb') as f:
        ans = pickle.load(f)
end = time.process_time()
print('Disk Store/Load Time of 1000 messages', end - start)


