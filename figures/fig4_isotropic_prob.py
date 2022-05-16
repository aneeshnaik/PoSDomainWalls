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
import matplotlib.patches as patches


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
    norm1 = LogNorm(vmax=1e-30, vmin=1e-60)
    norm2 = LogNorm(vmax=1, vmin=1e-32)
    clist1 = ['#007f7f', '#108686', '#1e8d8d', '#299494',
              '#339b9b', '#3ca2a2', '#44aaa9', '#4db1b0',
              '#55b8b8', '#5dc0bf', '#64c7c6', '#6ccfce',
              '#74d6d5', '#7cdedd', '#83e5e4', '#8bedec']
    clist2 = ['#583500', '#634000', '#6f4a00', '#7c5500',
              '#896000', '#976c00', '#a47700', '#b28300',
              '#c08f00', '#ce9b0e', '#dda822', '#ebb431',
              '#f9c13f', '#ffce4c', '#ffdb59', '#ffe866']
    cmap1 = LSCmap.from_list("", clist1)
    cmap2 = LSCmap.from_list("", clist2)
    args1 = {'norm': norm1, 'cmap': cmap1, 'shading': 'gouraud'}
    args2 = {'norm': norm2, 'cmap': cmap2, 'shading': 'gouraud'}

    # set up figure
    X0 = 0.04
    X1 = 0.98
    Xgap = 0.01
    Y0 = 0.22
    Y1 = 0.91
    dX = (X1 - X0 - 2 * Xgap) / 3
    dY = Y1 - Y0
    fig = plt.figure(figsize=(6.05, 3), dpi=150)
    ax0 = fig.add_axes([X0, Y0, dX, dY], projection='polar')
    ax1 = fig.add_axes([X0 + dX + Xgap, Y0, dX, dY], projection='polar')
    ax2 = fig.add_axes([X0 + 2 * (dX + Xgap), Y0, dX, dY], projection='polar')

    # plot
    im0 = ax0.pcolormesh(phi_grid, theta_grid, p0, **args1)
    im1 = ax1.pcolormesh(phi_grid, theta_grid, p1, **args1)
    im2 = ax2.pcolormesh(phi_grid, theta_grid, p1 / p0, **args2)

    # colourbars
    CdY = 0.035
    CY = Y0 - 2.75 * CdY
    cax0 = fig.add_axes([X0, CY, 2 * dX + Xgap, CdY])
    cax1 = fig.add_axes([X0 + 2 * (dX + Xgap), CY, dX, CdY])
    cbar1 = plt.colorbar(im1, cax=cax0, orientation='horizontal')
    cbar2 = plt.colorbar(im2, cax=cax1, orientation='horizontal')

    # ticks etc
    for ax in [ax0, ax1, ax2]:
        ax.grid(True, c='k', lw=0.25)
        ax.set_yticks([np.pi / 4, np.pi / 2])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    labs = ['', '', '', '',
            r'$\pi$', '', r'$3\pi/2$', '']
    ax0.xaxis.set_ticklabels(labs)
    labs = [r"$\pi / 4$", r"$\pi / 2$"]
    ax0.yaxis.set_ticklabels(labs)
    ax0.set_rlabel_position(135)
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="k", clip_on=False)
    a1 = patches.FancyArrowPatch(
        (np.pi, 0.55 * np.pi), (1.5 * np.pi, 0.55 * np.pi),
        connectionstyle="arc3, rad=0.414", **kw,
    )
    a2 = patches.FancyArrowPatch(
        (0.75 * np.pi, 0.05 * np.pi), (0.75 * np.pi, 0.5 * np.pi), **kw
    )
    ax0.add_patch(a1)
    ax0.add_patch(a2)
    ax0.text(0.85 * np.pi, 0.3 * np.pi, r"$\hat{\theta}$")
    ax0.text(1.25 * np.pi, 0.65 * np.pi, r'$\hat{\phi}$')

    cbar2.set_ticks([1e-32, 1e-24, 1e-16, 1e-8, 1e+0])
    cbar1.set_label(r"$\mathcal{P}$")
    cbar2.set_label(r"$\mathcal{P}_\mathrm{DW} / \mathcal{P}_\mathrm{no\,DW}$")
    for i, ax in enumerate([ax0, ax1, ax2]):
        t = ["No domain wall", "Domain wall at $z=0$", "Ratio"][i]
        ax.text(0.5, 1.07, t, ha='center', va='bottom', transform=ax.transAxes)

    # save figure
    fig.savefig("fig4_isotropic_prob.pdf")
