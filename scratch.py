import jax
import jax.numpy as np
from functools import partial

@remat
def g(x):
  return np.sin(np.sin(x))

def value_and_grad(x, t):
  y, vjp = jax.vjp(g, x)
  return y, vjp(t)

print(jax.make_jaxpr(value_and_grad)(1., 1.))
# print(jax.make_jaxpr(partial(jax.jvp, g))([1.], [1.]))
value_and_grad(1., 1.)
