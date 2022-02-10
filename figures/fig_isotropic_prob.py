#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:52:32 2022

@author: ppzapn
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from scipy.stats import multinomial


# load data
simdir = os.environ["POSDWDIR"]
data = np.load(f"{simdir}/sim_noDW.npz")
pos0 = data['x']
data = np.load(f"{simdir}/sim_DW.npz")
pos1 = data['x']
N_sat = pos0.shape[1]
N_snaps = pos0.shape[0]
assert pos1.shape[1] == N_sat


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


N_phi = 30
N_theta = 50

phi_arr = np.linspace(0, 2 * np.pi, N_phi)
theta_arr = np.linspace(0, np.pi / 2, N_theta)
p0 = np.zeros((N_theta, N_phi))
p1 = np.zeros((N_theta, N_phi))
for i in range(N_theta):
    for j in range(N_phi):
        p0[i, j] = calc_prob(pos0[-1], theta_arr[i], phi_arr[j])
        p1[i, j] = calc_prob(pos1[-1], theta_arr[i], phi_arr[j])
        
phi_grid, theta_grid = np.meshgrid(phi_arr, theta_arr)


fig = plt.figure(figsize=(12, 12))
ax0 = fig.add_subplot(131, projection='polar')
ax1 = fig.add_subplot(132, projection='polar')
ax2 = fig.add_subplot(133, projection='polar')
im0 = ax0.pcolormesh(phi_grid, theta_grid, p0, norm=LogNorm(vmax=1e-28, vmin=1e-60), cmap='Spectral')
im1 = ax1.pcolormesh(phi_grid, theta_grid, p1, norm=LogNorm(vmax=1e-28, vmin=1e-60), cmap='Spectral')
im2 = ax2.pcolormesh(phi_grid, theta_grid, p1/p0, norm=LogNorm(), cmap='Spectral')

plt.colorbar(im2)