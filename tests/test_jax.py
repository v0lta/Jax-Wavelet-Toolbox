"""Runs the example from https://github.com/google/jax#automatic-differentiation-with-grad ."""

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#


import chex
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax import grad


class JaxTest(parameterized.TestCase):
    """Test for jax."""

    @chex.all_variants()
    def test_jax(self):
        """Test if jax is correctly installed."""

        @self.variant
        def tanh(x):  # Define a function
            y = jnp.exp(-2.0 * x)
            return (1.0 - y) / (1.0 + y)

        grad_tanh = grad(tanh)  # Obtain its gradient function
        print(grad_tanh(1.0))  # Evaluate it at x = 1.0
        assert (grad_tanh(1.0) - 0.4199743) < 0.000001


if __name__ == "__main__":
    absltest.main()
