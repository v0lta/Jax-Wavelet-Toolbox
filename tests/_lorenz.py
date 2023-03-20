"""Code for simulation of the chaotic lorenz system."""
#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#
import jax.numpy as np

default_x0 = np.array([2.0, 1.0, 3.0])


def generate_lorenz(
    a: float = 10.0,
    b: float = 28.0,
    c: float = 8.0 / 3.0,
    dt: float = 0.01,
    tmax: float = 5.11,
    x0: np.ndarray = default_x0,
) -> np.ndarray:
    """Generate test data using the lorenz-system.

    The implementation follows the description at:
    https://en.wikipedia.org/wiki/Lorenz_system

    Args:
        a (float): Parameter a. Defaults to 10.0.
        b (float): Parameter b. Defaults to 28.0.
        c (float): Parameter c. Defaults to 8.0/3.0.
        dt (float): The time step for the forward euler method.
            Defaults to 0.01.
        tmax (float): Where to stop the simulation. Defaults to 5.11.
        x0 (np.array): The initial state. Defaults to np.array([2.0, 1.0, 3.0]).

    Returns:
        np.array: The simulation result.
    """
    steps = tmax / dt
    x_lst = []

    def compute_dx_dt(x: np.ndarray) -> np.ndarray:
        dx0dt = a * (x[1] - x[0])
        dx1dt = x[0] * (b - x[2]) - x[1]
        dx2dt = x[0] * x[1] - c * x[2]
        return np.array([dx0dt, dx1dt, dx2dt])

    x = x0
    x_lst.append(x0)
    for _ in range(int(steps)):
        x = dt * compute_dx_dt(x) + x
        x_lst.append(x)

    x_sim = np.stack(x_lst)
    return x_sim
