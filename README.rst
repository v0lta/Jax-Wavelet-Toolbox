.. |favicon| image:: https://raw.githubusercontent.com/v0lta/Jax-Wavelet-Toolbox/master/docs/favicon/favicon.ico
    :alt: Shannon-wavelet favicon
    :width: 32
    :target: https://pypi.org/project/jaxwt/

*************************************
|favicon| Jax Wavelet Toolbox (jaxwt)
*************************************


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
- ``wavedec`` and ``waverec`` implement 1d analysis and synthesis transforms.
- Similarly, ``wavedec2`` and ``waverec2`` provide 2d transform support.
- The ``cwt``-function supports 1d continuous wavelet transforms.
- The ``WaveletPacket`` object supports 1d wavelet packet transforms.
- ``WaveletPacket2d`` implements two-dimensional wavelet packet transforms.
- ``swt`` and ``iswt`` allow 1d-stationary transformations.

This toolbox extends `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_. 
We additionally provide GPU and gradient support via a Jax backend.

Installation
""""""""""""
To install Jax, head over to https://github.com/google/jax#installation and follow the procedure described there.
Afterward, type ``pip install jaxwt`` to install the Jax-Wavelet-Toolbox. You can uninstall it later by typing ``pip uninstall jaxwt``.

Documentation
"""""""""""""
Complete documentation of all toolbox functions is available at
`readthedocs <https://jax-wavelet-toolbox.readthedocs.io/en/latest/jaxwt.html>`_.


Transform Examples:
"""""""""""""""""""

To compute a one-dimensional fast wavelet transform, consider the code snippet below:

.. code-block:: python

  import jax.numpy as jnp
  import jaxwt as jwt

  import pywt
  import numpy as np;

  # generate an input of even length.
  data = jnp.array([0., 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
  
  # compare the forward fwt coefficients
  print(pywt.wavedec(np.array(data), 'haar', mode='zero', level=2))
  print(jwt.wavedec(data, 'haar', mode='zero', level=2))
  
  # invert the fwt.
  print(jwt.waverec(jwt.wavedec(data, 'haar', mode='zero', level=2),
                    'haar'))


The snipped also evaluates the `pywt` implementation to demonstrate that the coefficients are the same.
Use `jaxwt` if you require gradient or GPU support.

The process for two-dimensional fast wavelet transforms works similarly:

.. code-block:: python

  import jaxwt as jwt
  import jax.numpy as jnp
  from scipy.datasets import face

  image = jnp.transpose(
      face(), [2, 0, 1]).astype(jnp.float32)
  transformed = jwt.wavedec2(image, "haar", 
                             level=2, mode="reflect")
  reconstruction = jwt.waverec2(transformed, "haar")
  jnp.max(jnp.abs(image - reconstruction))


``jaxwt`` allows transforming batched data.
The example above moves the color channel to the front because wavedec2 transforms the last two axes by default.
We can avoid doing so by using the ``axes`` argument. Consider the batched example below:

.. code-block:: python

  import jaxwt as jwt
  import jax.numpy as jnp
  from scipy.datasets import face

  image = jnp.stack(
      [face(), face(), face()], axis=0
       ).astype(jnp.float32)
  transformed = jwt.wavedec2(image, "haar", 
                             level=2, mode="reflect",
                             axes=(1,2))
  reconstruction = jwt.waverec2(transformed, "haar", axes=(1,2))
  jnp.max(jnp.abs(image - reconstruction))


For more code examples, follow the documentation link above or visit
the `examples <https://github.com/v0lta/Jax-Wavelet-Toolbox/tree/master/examples>`_ folder.



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

If you use this work in a scientific context, please cite the following:

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
