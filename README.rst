*************************
Jax-Wavelet-Toolbox (jwt)
*************************


.. image:: https://github.com/v0lta/Jax-Wavelet-Toolbox/actions/workflows/tests.yml/badge.svg 
    :target: https://github.com/v0lta/Jax-Wavelet-Toolbox/actions/workflows/tests.yml
    :alt: GitHub Actions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black code style


Differentiable and GPU enabled fast wavelet transforms in JAX. 

Features
""""""""
- 1d forward and backward fwt are implemented in `src/jaxlets/conv_fwt.py`.
- 2d forward and backard fwt methods are part of the `src/jaxlets/conv_fwt_2d.py` module.

Installation
""""""""""""
Head to https://github.com/google/jax#installation and follow the procedure described there, then do the 
following to install the code in development mode:

.. code-block:: sh

    $ git clone https://github.com/v0lta/jaxlets
    $ cd jaxlets
    $ pip install -e .

If you want it ready to go, do the following:

.. code-block:: sh

    $ git clone https://github.com/v0lta/jaxlets
    $ cd jaxlets
    $ pip install git+https://github.com/v0lta/jaxlets.git

If you aren't able to follow the JAX installation instructions, you can install it in CPU-only mode
using the `jax_cpu` extra. This means you have to use development mode and install like this:

.. code-block:: sh

    $ git clone https://github.com/v0lta/jaxlets
    $ cd jaxlets
    $ pip install -e .[jax_cpu]


Transform Example:
""""""""""""""""""

.. code-block:: python

  import pywt
  import numpy as np;
  import jax.numpy as jnp
  import src.jwt as jwt
  # generate an input of even length.
  data = jnp.array([0., 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
  wavelet = pywt.Wavelet('haar')
  
  # compare the forward fwt coefficients
  print(pywt.wavedec(np.array(data), wavelet, mode='zero', level=2))
  print(jwt.wavedec(data, wavelet, mode='zero', level=2))
  
  # invert the fwt.
  print(jwt.waverec(jwt.wavedec(data, wavelet, mode='zero', level=2), wavelet))


Testing
"""""""
Unit tests are handled by ``tox``. Clone the repository and run it with the following:

.. code-block:: sh

    $ pip install tox
    $ git clone https://github.com/v0lta/jaxlets
    $ cd jaxlets
    $ tox

Goals
"""""
- In the spirit of jax the aim is to be 100% pywt compatible. Whenever possible, interfaces should be the same
  results identical.
