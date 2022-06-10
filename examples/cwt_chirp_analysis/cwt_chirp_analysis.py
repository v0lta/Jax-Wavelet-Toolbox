import jax.numpy as jnp
import jaxwt as jwt
import matplotlib.pyplot as plt
import scipy.signal as signal

if __name__ == "__main__":
    t = jnp.linspace(-2, 2, 800, endpoint=False)
    sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
    widths = jnp.arange(1, 31)
    cwtmatr, freqs = jwt.cwt(
        sig, widths, "mexh", sampling_period=(4 / 800) * jnp.pi
    )
    fig, axs = plt.subplots(2)
    axs[0].plot(t, sig)
    axs[0].set_ylabel("magnitude")
    axs[1].imshow(
        cwtmatr,
        cmap="PRGn",
        aspect="auto",
        vmax=abs(cwtmatr).max(),
        vmin=-abs(cwtmatr).max(),
        extent=[min(t), max(t), min(freqs), max(freqs)],
    )
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("frequency")
    plt.show()