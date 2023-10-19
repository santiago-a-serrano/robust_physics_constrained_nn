import jax.numpy as jnp
import jax.random as random
from jax import jit
import numpy as np


@jit
def add_gaussian_noise(key, array, scale=0.1):
    noise = random.normal(key, array.shape) * scale
    return array + noise


# array = jnp.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]],
#                    [[[10], [11], [12]], [[13], [14], [15]], [[16], [17], [18]]]])
# key = random.PRNGKey(0)
# print(add_gaussian_noise(key, array, scale=0.1))
