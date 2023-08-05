***************************
Jax Wavelet Toolbox (jaxwt)
***************************


.. image:: https://github.com/v0lta/Jax-Wavelet-Toolbox/actions/workflows/tests.yml/badge.svg 
    :target: https://github.com/v0lta/Jax-Wavelet-Toolbox/actions/workflows/tests.yml
    :alt: GitHub Actions

.. image:: https://readthedocs.org/projects/jax-wavelet-toolbox/badge/?version=latest
    :target: https://jax-wavelet-toolbox.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/pyversions/jaxwt
    :target: https://pypi.org/project/jaxwt/
    :alt: PyPI Versions

.. image:: https://img.shields.io/pypi/v/jaxwt
    :target: https://pypi.org/project/jaxwt/
    :alt: PyPI - Project

.. image:: https://img.shields.io/pypi/l/jaxwt
    :target: https://github.com/v0lta/Jax-Wavelet-Toolbox/blob/master/LICENSE
    :alt: PyPI - License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black code style

.. image:: https://static.pepy.tech/personalized-badge/jaxwt?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
    :target: https://pepy.tech/project/jaxwt
    :alt: PyPi - downloads


Differentiable and GPU-enabled fast wavelet transforms in JAX. 

Features
""""""""
- 1d analysis and synthesis transforms are implemented in ``src/jaxwt/conv_fwt.py``.
  Try ``wavedec`` and ``waverec``.
- 2d analysis and synthesis transforms are part of the ``src/jaxwt/conv_fwt_2d.py`` module.
  The two functions are called ``wavedec2`` and ``waverec2``.
- Furthermore, 3d transforms are provided by the ``wavedec3`` and ``waverec3`` functions.
- ``cwt``-function supports 1d continuous wavelet transforms.
- The ``WaveletPacket`` object supports 1d wavelet packet transforms.
- ``WaveletPacket2d`` implements two-dimensional wavelet packet transforms.
- ``swt`` computes a single dimensional stationary transform ``iswt`` inverts it. 

This toolbox extends `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ .
``jaxwt`` additionally provides GPU and gradient support via a Jax backend.

Installation
""""""""""""
To install Jax, head over to https://github.com/google/jax#installation and follow the procedure described there.
Afterward, type ``pip install jaxwt`` to install the Jax-Wavelet-Toolbox. You can uninstall it later by typing ``pip uninstall jaxwt``.

Documentation
"""""""""""""
The documentation is available at: https://jax-wavelet-toolbox.readthedocs.io/en/latest/jaxwt.html .


Transform Examples:
"""""""""""""""""""

One-dimensional fast wavelet transform:

.. code-block:: python

  import pywt
  import numpy as np;
  import jax.numpy as jnp
  import jaxwt as jwt
  # generate an input of even length.
  data = jnp.array([0., 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
  wavelet = pywt.Wavelet('haar')
  
  # compare the forward fwt coefficients
  print(pywt.wavedec(np.array(data), wavelet, mode='zero', level=2))
  print(jwt.wavedec(data, wavelet, mode='zero', level=2))
  
  # invert the fwt.
  print(jwt.waverec(jwt.wavedec(data, wavelet, mode='zero', level=2),
                    wavelet))


Two-dimensional fast wavelet transform:

.. code-block:: python

  import pywt, scipy.datasets
  import jaxwt as jwt
  import jax.numpy as jnp
  face = jnp.transpose(
      scipy.datasets.face(), [2, 0, 1]).astype(jnp.float64)
  transformed = jwt.wavedec2(face, pywt.Wavelet("haar"), 
                             level=2, mode="reflect")
  reconstruction = jwt.waverec2(transformed, pywt.Wavelet("haar"))
  jnp.max(jnp.abs(face - reconstruction))



Testing
"""""""
Unit tests are handled by ``nox``. Clone the repository and run it with the following:

.. code-block:: sh

    $ pip install nox
    $ git clone https://github.com/v0lta/Jax-Wavelet-Toolbox
    $ cd Jax-Wavelet-Toolbox
    $ nox -s test

Goals
"""""
- In the spirit of Jax, the aim is to be 100% pywt compatible. Whenever possible, interfaces should be the same
  results identical.


64-Bit floating-point numbers
"""""""""""""""""""""""""""""
If you need 64-bit floating point support, set the Jax config flag: 

.. code-block:: python

    from jax.config import config
    config.update("jax_enable_x64", True)


Citation
"""""""""""

If you use this work in a scientific context, please cite:

.. code-block::

  @phdthesis{handle:20.500.11811/9245,
    urn: https://nbn-resolving.org/urn:nbn:de:hbz:5-63361,
    author = {{Moritz Wolter}},
    title = {Frequency Domain Methods in Recurrent Neural Networks for Sequential Data Processing},
    school = {Rheinische Friedrich-Wilhelms-Universität Bonn},
    year = 2021,
    month = jul,
    url = {https://hdl.handle.net/20.500.11811/9245}
  }
