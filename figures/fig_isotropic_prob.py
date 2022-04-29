#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:52:32 2022

@author: ppzapn
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap as LSCmap
import os
from scipy.stats import multinomial


def calc_prob(pos, theta, phi):

    # rotation matrices
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)
    R1 = np.array([[ cp, -sp,  0],
                   [ sp,  cp,  0],
                   [  0,   0,  1]])
    R2 = np.array([[ ct,   0, st],
                   [  0,   1,  0],
                   [-st,   0, ct]])
    R = np.matmul(R2, R1)

    # rotate positions
    posT = np.matmul(R, pos.T).T

    # get polar angles
    th = np.arctan2(np.sqrt(posT[..., 0]**2 + posT[..., 1]**2), posT[..., 2])

    # bin
    N_bins = 21
    bins = np.linspace(0, np.pi, N_bins + 1)
    x = np.histogram(th, bins=bins)[0]

    # construct multinomial dist
    p = -0.5 * np.diff(np.cos(bins))
    dist = multinomial(n=N_sat, p=p)

    # get prob
    return dist.pmf(x)


if __name__ == '__main__':

    # load data
    simdir = os.environ["POSDWDDIR"]
    data = np.load(f"{simdir}/sim_noDW.npz")
    pos0 = data['x']
    data = np.load(f"{simdir}/sim_DW.npz")
    pos1 = data['x']
    N_sat = pos0.shape[1]
    N_snaps = pos0.shape[0]
    assert pos1.shape[1] == N_sat

    # get probabilities
    N_phi = 30
    N_theta = 50
    phi_arr = np.linspace(0, 2 * np.pi, N_phi)
    theta_arr = np.linspace(0, np.pi / 2, N_theta)
    phi_grid, theta_grid = np.meshgrid(phi_arr, theta_arr)
    p0 = np.zeros((N_theta, N_phi))
    p1 = np.zeros((N_theta, N_phi))
    for i in range(N_theta):
        for j in range(N_phi):
            p0[i, j] = calc_prob(pos0[-1], theta_arr[i], phi_arr[j])
            p1[i, j] = calc_prob(pos1[-1], theta_arr[i], phi_arr[j])

    # plot settings
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    c1 = 'goldenrod'
    c2 = 'teal'
    norm1 = LogNorm(vmax=1e-28, vmin=1e-60)
    norm2 = LogNorm(vmax=1, vmin=1e-32)
    cmap1 = LSCmap.from_list("", ['k', "teal","w"])
    cmap2 = LSCmap.from_list("", ["darkgoldenrod", 'w'])
    args1 = {'norm': norm1, 'cmap': cmap1, 'shading': 'gouraud'}
    args2 = {'norm': norm2, 'cmap': cmap2, 'shading': 'gouraud'}

    # set up figure
    X0 = 0.035
    X1 = 0.95
    Xgap = 0.095
    Y0 = 0.22
    Y1 = 0.91
    dX = (X1 - X0 - 2 * Xgap) / 3
    dY = Y1 - Y0
    fig = plt.figure(figsize=(7, 3.2), dpi=150)
    ax0 = fig.add_axes([X0, Y0, dX, dY], projection='polar')
    ax1 = fig.add_axes([X0 + dX + Xgap, Y0, dX, dY], projection='polar')
    ax2 = fig.add_axes([X0 + 2 * (dX + Xgap), Y0, dX, dY], projection='polar')

    # plot
    im0 = ax0.pcolormesh(phi_grid, theta_grid, p0, **args1)
    im1 = ax1.pcolormesh(phi_grid, theta_grid, p1, **args1)
    im2 = ax2.pcolormesh(phi_grid, theta_grid, p1 / p0, **args2)

    # colourbars
    CdY = 0.035
    CY = Y0 - 2.5 * CdY
    cax0 = fig.add_axes([X0, CY, 2 * dX + Xgap, CdY])
    cax1 = fig.add_axes([X0 + 2 * (dX + Xgap), CY, dX, CdY])
    cbar1 = plt.colorbar(im1, cax=cax0, orientation='horizontal')
    cbar2 = plt.colorbar(im2, cax=cax1, orientation='horizontal')

    # ticks etc
    for ax in [ax0, ax1, ax2]:
        ax.grid(True, c='k', lw=0.25)
        labs = [r'$\hat{\phi} = 0$', r'$\pi / 4$', r'$\pi / 2$', r'$3\pi / 4$',
                r'$\pi$', r'$5\pi / 4$', r'$3\pi / 2$', r'$7\pi / 4$']
        ax.xaxis.set_ticklabels(labs)
        ax.set_yticks([np.pi / 4, np.pi / 2])
        labs = [r"$\hat{\theta} = \pi / 4$", r"$\hat{\theta} = \pi / 2$"]
        ax.yaxis.set_ticklabels(labs)
    cbar2.set_ticks([1e-32, 1e-24, 1e-16, 1e-8, 1e+0])
    cbar1.set_label(r"$\mathcal{P}$")
    cbar2.set_label(r"$\mathcal{P}_\mathrm{DW} / \mathcal{P}_\mathrm{no\,DW}$")
    for i, ax in enumerate([ax0, ax1, ax2]):
        t = ["No domain wall", "Domain wall at $z=0$", "Ratio"][i]
        ax.text(0.5, 1.2, t, ha='center', va='bottom', transform=ax.transAxes)

    fig.savefig("fig_isotropic_prob.pdf")
