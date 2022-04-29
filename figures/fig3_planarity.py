#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':

    # load data
    simdir = os.environ["POSDWDDIR"]
    data = np.load(f"{simdir}/sim_noDW.npz")
    pos0 = data['x']
    vel0 = data['v']
    t0 = data['t']
    data = np.load(f"{simdir}/sim_DW.npz")
    pos1 = data['x']
    vel1 = data['v']
    t1 = data['t']
    N_sat = pos0.shape[1]
    N_snaps = pos0.shape[0]
    assert pos1.shape[1] == N_sat
    np.testing.assert_allclose(t0, t1)

    # find satellites within 400kpc
    m0 = np.linalg.norm(pos0, axis=-1) < 400
    m1 = np.linalg.norm(pos1, axis=-1) < 400

    # rms height
    h0 = np.zeros(N_snaps)
    h1 = np.zeros(N_snaps)
    for i in range(N_snaps):
        h0[i] = np.sqrt(np.mean(pos0[i, m0[i], 2]**2))
        h1[i] = np.sqrt(np.mean(pos1[i, m1[i], 2]**2))

    # inertia tensor c/a
    f0 = np.zeros(N_snaps)
    f1 = np.zeros(N_snaps)
    for i in range(N_snaps):
        a0 = np.sum(pos0[i, m0[i]]**2, axis=-1)[:, None, None]
        a0 = a0 * np.ones((m0[i].sum(), 3, 3)) * np.eye(3)[None]
        b0 = (pos0[i, m0[i], :, None] * pos0[i, m0[i], None])
        I0 = np.sum(a0 - b0, axis=0)
        e0 = np.linalg.eigvals(I0)
        f0[i] = e0.min() / e0.max()

        a1 = np.sum(pos1[i, m1[i]]**2, axis=-1)[:, None, None]
        a1 = a1 * np.ones((m1[i].sum(), 3, 3)) * np.eye(3)[None]
        b1 = (pos1[i, m1[i], :, None] * pos1[i, m1[i], None])
        I1 = np.sum(a1 - b1, axis=0)
        e1 = np.linalg.eigvals(I1)
        f1[i] = e1.min() / e1.max()

    # plot settings
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    c1 = 'goldenrod'
    c2 = 'teal'

    # set up figure
    X0 = 0.14
    X1 = 0.98
    Y0 = 0.125
    Y1 = 0.98
    dX = X1 - X0
    dY = (Y1 - Y0) / 2
    fig = plt.figure(figsize=(3.4, 3), dpi=150)
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X0, Y0 + dY, dX, dY])

    # plot
    ax0.plot(t1, h0, label="No domain wall", c=c1)
    ax0.plot(t1, h1, label="Domain wall at $z=0$", c=c2)
    ax1.plot(t1, f0, c=c1)
    ax1.plot(t1, f1, c=c2)

    # limits etc
    for ax in [ax0, ax1]:
        ax.tick_params(direction='inout', right=True, top=True)
        ax.set_xlim(t1[0], t1[-1])
    ax0.set_xlabel("Time [Gyr]")
    ax0.set_ylabel("RMS Height")
    ax1.set_ylabel("Minor-Major Axis Ratio")
    ax0.set_ylim(0, 170)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(labelbottom=False)
    ax0.legend(frameon=False)

    # save figure
    fig.savefig("fig3_planarity.pdf")
