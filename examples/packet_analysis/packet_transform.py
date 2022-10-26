import pywt
import jax.numpy as np
from jaxwt.packets import WaveletPacket
import matplotlib.pyplot as plt
import scipy.signal as signal

t = np.linspace(0, 10, 5001)
wavelet = pywt.Wavelet("db4")
w = signal.chirp(t, f0=0.00001, f1=20, t1=10, method="linear")

wp = WaveletPacket(data=w, wavelet=wavelet, mode="reflect")
nodes = wp.get_level(7)
np_lst = []
for node in nodes:
    np_lst.append(wp[node])
viz = np.concatenate(np_lst)

fig, axs = plt.subplots(2)
axs[0].plot(t, w)
axs[0].set_title("Linear Chirp, f(0)=.00001, f(10)=20")
axs[0].set_xlabel("t (sec)")

axs[1].set_title("Wavelet analysis")
axs[1].imshow(viz[:20, :])
axs[1].set_xlabel("time")
axs[1].set_ylabel("frequency")
plt.show()
