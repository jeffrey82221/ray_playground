import ray

ray.init()

@ray.remote
class Actor:
    def __init__(self, init_value):
        self.i = init_value

    def inc(self, x):
        self.i += x

    def get(self):
        return self.i

@ray.remote
def combine(x, y):
    return x + y

actor_1 = Actor.bind(10)  # Instantiate Actor with init_value 10.
result_of_get = actor_1.get.bind()  # ClassMethod that returns value from get() from
                     # the actor created.
ans = ray.get(result_of_get.execute())
print(ans)
assert ans == 10


actor_2 = Actor.bind(10) # Instantiate another Actor with init_value 10.
actor_1.inc.bind(2)  # Call inc() on the actor created with increment of 2.
actor_2.inc.bind(4)  # Call inc() on the actor created with increment of 4.
actor_2.inc.bind(6)  # Call inc() on the actor created with increment of 6.

ans = ray.get(actor_1.get.bind().execute()) 
print(ans)
assert ans == 12
ans = ray.get(actor_2.get.bind().execute())
print(ans)
assert ans == 20

# Combine outputs from a1.get() and a2.get()
dag = combine.bind(actor_1.get.bind(), actor_2.get.bind())

# a1 +  a2 + inc(2) + inc(4) + inc(6)
# 10 + (10 + ( 2   +    4    +   6)) = 32
ans = ray.get(dag.execute())
print(ans)
assert ans == 32