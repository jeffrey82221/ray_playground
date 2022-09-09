import ray
from ray.serve.deployment_graph import InputNode

ray.init()


@ray.remote
def func_a(user_input):
    return user_input * 2

@ray.remote
def func_b(user_input):
    return user_input + 1

@ray.remote
def func_c(x, y):
    return x + y

@ray.remote
def func_d(a, b):
    return a, a + b

@ray.remote
def get(multi_input, index):
    return multi_input[index]

with InputNode() as dag_input:
    a_ref = func_a.bind(dag_input)
    b_ref = func_b.bind(dag_input)
    c_ref = func_c.bind(a_ref, b_ref)
    de_ref = func_d.bind(a_ref, b_ref)
    d_ref = get.bind(de_ref, 0)
    e_ref = get.bind(de_ref, 1)

#   a(2)  +   b(2)  = c
# (2 * 2) + (2 * 1)
assert ray.get(a_ref.execute(2)) == 4
assert ray.get(b_ref.execute(2)) == 3
assert ray.get(c_ref.execute(2)) == 7
assert ray.get(de_ref.execute(2)) == (4, 7)
assert ray.get(d_ref.execute(2)) == 4
assert ray.get(e_ref.execute(2)) == 7

