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
    simdir = os.environ["POSDWDIR"]
    data = np.load(f"{simdir}/sim_noDW.npz")
    x0 = data['x']
    v0 = data['v']
    t0 = data['t']
    data = np.load(f"{simdir}/sim_DW.npz")
    x1 = data['x']
    v1 = data['v']
    t1 = data['t']
    N_sat = x0.shape[1]
    assert x1.shape[1] == N_sat
    np.testing.assert_allclose(t0, t1)

    # calculate angular momenta
    L0 = np.cross(x0, v0)
    L1 = np.cross(x1, v1)
    Lav0 = np.average(L0, axis=0)
    Lav1 = np.average(L1, axis=0)
    Lx0, Ly0, Lz0 = L0[-1, :, 0], L0[-1, :, 1], L0[-1, :, 2]
    Lx1, Ly1, Lz1 = L1[-1, :, 0], L1[-1, :, 1], L1[-1, :, 2]
    Lxav0, Lyav0, Lzav0 = Lav0[:, 0], Lav0[:, 1], Lav0[:, 2]
    Lxav1, Lyav1, Lzav1 = Lav1[:, 0], Lav1[:, 1], Lav1[:, 2]

    # calculate orbital pole directions
    tht00 = np.arctan2(Lz0, np.sqrt(Lx0**2 + Ly0**2))
    phi00 = np.arctan2(Ly0, Lx0)
    tht10 = np.arctan2(Lz1, np.sqrt(Lx1**2 + Ly1**2))
    phi10 = np.arctan2(Ly1, Lx1)
    tht01 = np.arctan2(Lzav0, np.sqrt(Lxav0**2 + Lyav0**2))
    phi01 = np.arctan2(Lyav0, Lxav0)
    tht11 = np.arctan2(Lzav1, np.sqrt(Lxav1**2 + Lyav1**2))
    phi11 = np.arctan2(Lyav1, Lxav1)

    # plot settings
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    c0 = 'goldenrod'
    c1 = 'teal'
    sargs0 = {'s': 0.1, 'c': c0, 'rasterized': True}
    sargs1 = {'s': 0.1, 'c': c1, 'rasterized': True}
    proj = 'lambert'

    # set up figure
    asp = 7 / 6.25
    Xgap = 0.02
    Ygap = asp * 0.02
    X0 = 0.1275
    X1 = 0.98
    Y0 = 0.005
    dX = (X1 - X0 - Xgap) / 2
    dY = asp * dX
    fig = plt.figure(figsize=(7, 7 / asp), dpi=150)
    ax00 = fig.add_axes([X0, Y0 + dY + Ygap, dX, dY], projection=proj)
    ax01 = fig.add_axes([X0, Y0, dX, dY], projection=proj)
    ax10 = fig.add_axes([X0 + Xgap + dX, Y0 + dY + Ygap, dX, dY], projection=proj)
    ax11 = fig.add_axes([X0 + Xgap + dX, Y0, dX, dY], projection=proj)

    # plot
    ax00.scatter(phi00, tht00, **sargs0)
    ax01.scatter(phi01, tht01, **sargs0)
    ax10.scatter(phi10, tht10, **sargs1)
    ax11.scatter(phi11, tht11, **sargs1)

    # labels, ticks etc.
    for ax in fig.axes:
        ax.grid(True)
        for i in range(11):
            if (i + 1) % 3:
                t = ax.xaxis.get_major_ticks()[i]
                t.label.set_visible(False)
        args = {"fontsize": 8, "ha": 'center'}
        ax.text(0, -np.pi / 3, "-60°", **args)
        ax.text(0, -np.pi / 6, "-30°", **args)
        ax.text(0, np.pi / 6, "30°", **args)
        ax.text(0, np.pi / 3, "60°", **args)
        ax.scatter([0, 0], [-np.pi/2, np.pi/2], marker='.', c='k', s=4)
    ax00.text(0.5, 1.02, "No domain wall", ha='center', va='bottom', transform=ax00.transAxes)
    ax10.text(0.5, 1.02, "Domain wall at $z=0$", ha='center', va='bottom', transform=ax10.transAxes)
    ax00.text(-0.02, 0.5, "Instantaneous", ha='right', va='center', transform=ax00.transAxes)
    ax01.text(-0.02, 0.5, "Time-averaged", ha='right', va='center', transform=ax01.transAxes)


    # save
    fig.savefig("fig_orbital_poles.pdf")