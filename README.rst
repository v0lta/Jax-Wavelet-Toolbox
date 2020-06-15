jaxlets
=======
Differentiable and gpu enabled fast wavelet transforms in jax. 

Features
--------
-1d forward and backward fwt implemented in `conv_fwt.py`.
-2d forward and backard fwt methods are part of the `conv_fwt_2d.py` module.

Installation
------------
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

Testing
-------
Unit tests are handled by ``tox``. Clone the repository and run it with the following:

.. code-block:: sh

    $ pip install tox
    $ git clone https://github.com/v0lta/jaxlets
    $ cd jaxlets
    $ tox

Test outputs can also be generated with ``tox -e plots``.

Goals
-----
- In the spirit of jax the aim is to be 100% pywt compatible. Whenever possible, interfaces should be the same
  results identical.

Coming up
---------
- Wavelet packets (TODO)
- Interface improvements
- Extended jit support.

Feedback
--------
This is an early release, if you have questions, suggestions or run into issues, please open an issue.
