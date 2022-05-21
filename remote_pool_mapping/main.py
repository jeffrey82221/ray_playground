import pandas as pd
import numpy as np
import ray
from ray.util import ActorPool
from datetime import datetime

def get_pandas_list(size, table_size=1000):
    result = []
    for i in range(size):
        result.append(pd.DataFrame(
            np.ones((table_size, table_size)) * i
            ))
    return result
    
@ray.remote
class Actor:
    def double(self, n):
        return sum(n * 2)

SIZE = 100
pool = ActorPool([Actor.remote(), Actor.remote()])

input_list = get_pandas_list(SIZE)

# RAY
now = datetime.now() # time object
gen = pool.map(lambda a, v: a.double.remote(v), input_list)
result = list(gen)
print("Ray Time:", datetime.now() - now)

# Single Process
now = datetime.now() # time object
gen = map(lambda a: a * 2, input_list)
result = list(gen)
print("Single Process Time:", datetime.now() - now)
