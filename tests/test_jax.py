"""Runs the example from https://github.com/google/jax#automatic-differentiation-with-grad ."""

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#


import jax.numpy as np
from jax import grad


def test_jax():
    """Test if jax is correctly installed."""

    def tanh(x):  # Define a function
        y = np.exp(-2.0 * x)
        return (1.0 - y) / (1.0 + y)

    grad_tanh = grad(tanh)  # Obtain its gradient function
    print(grad_tanh(1.0))  # Evaluate it at x = 1.0
    assert (grad_tanh(1.0) - 0.4199743) < 0.000001
