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
    t0 = data['t']
    data = np.load(f"{simdir}/sim_DW.npz")
    x1 = data['x']
    t1 = data['t']
    N_sat = x0.shape[1]
    assert x1.shape[1] == N_sat
    np.testing.assert_allclose(t0, t1)

    # plot settings
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    c0 = 'goldenrod'
    c1 = 'teal'
    sargs0 = {'s': 1, 'c': c0, 'rasterized': True}
    sargs1 = {'s': 1, 'c': c1, 'rasterized': True}
    largs0 = {'c': c0, 'alpha': 0.05, 'rasterized': True}
    largs1 = {'c': c1, 'alpha': 0.05, 'rasterized': True}

    # set up figure
    asp = 7 / 6.75
    Xgap = 0.035
    left = 0.08
    right = 0.98
    bottom = 0.06
    dX = (right - left - Xgap) / 2
    dY = asp * dX
    fig = plt.figure(figsize=(7, 7 / asp), dpi=150)
    ax00 = fig.add_axes([left, bottom + dY, dX, dY])
    ax01 = fig.add_axes([left, bottom, dX, dY])
    ax10 = fig.add_axes([left + Xgap + dX, bottom + dY, dX, dY])
    ax11 = fig.add_axes([left + Xgap + dX, bottom, dX, dY])

    # plot satellites positions
    ax00.scatter(x0[-1, :, 0], x0[-1, :, 2], **sargs0)
    ax10.scatter(x1[-1, :, 0], x1[-1, :, 2], **sargs1)

    # plot orbits
    ax01.plot(x0[:, :500, 0], x0[:, :500, 2], **largs0)
    ax11.plot(x1[:, :500, 0], x1[:, :500, 2], **largs1)

    # labels, limits etc.
    for ax in fig.axes:
        ax.set_xlim(-420, 420)
        ax.set_ylim(-420, 420)
        ax.tick_params(right=True, top=True, direction='inout')
    for ax in [ax00, ax10]:
        ax.tick_params(labelbottom=False)
    for ax in [ax10, ax11]:
        ax.tick_params(labelleft=False)
    ax00.set_ylabel(r"$z\ [\mathrm{kpc}]$")
    ax01.set_ylabel(r"$z\ [\mathrm{kpc}]$")
    ax01.set_xlabel(r"$x\ [\mathrm{kpc}]$")
    ax11.set_xlabel(r"$x\ [\mathrm{kpc}]$")
    ax00.set_title("No domain wall", fontsize=11)
    ax10.set_title("Domain wall at $z=0$", fontsize=11)

    # save figure
    fig.savefig("fig_sat_positions.pdf")
