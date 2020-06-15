# -*- coding: utf-8 -*-

#
# Created on Thu Jun 11 2020
# Copyright (c) 2020 Moritz Wolter
#

import os

import click
import jax
import jax.numpy as np


@jax.jit
def generate_lorenz(
    a=10.,
    b=28.,
    c=8. / 3.,
    dt=0.01,
    tmax=5.11,
    x0=np.array([2., 1., 3.]),
):
    steps = tmax / dt
    x_lst = []

    def compute_dx_dt(x):
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


@click.command()
@click.option('--directory', default=os.getcwd())
def main(directory):
    print('Simulation of the lorenz equation.')
    lorenz = generate_lorenz()
    print(lorenz.shape)
    # plt.plot(lorenz[:, 0])
    # plt.show()

    os.environ["DISPLAY"] = ":1"
    import matplotlib.pyplot as plt
    # matplotlib.use('Qt5Agg')

    print('Backend', plt.get_backend())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(lorenz[:, 0], lorenz[:, 1], lorenz[:, 2])
    plt.savefig(os.path.join(directory, 'lorenz_result.pdf'))


if __name__ == '__main__':
    main()
