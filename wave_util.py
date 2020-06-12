import pywt
import jax.numpy as np
from collections import namedtuple

JaxWavelet = namedtuple('JaxWavelet', ['dec_lo', 'dec_hi', 'rec_lo', 'rec_hi'])
