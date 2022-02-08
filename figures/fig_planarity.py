#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # load data
    data = np.load("../simulations/sim_noDW.npz")
    pos0 = data['x']
    vel0 = data['v']
    t0 = data['t']
    data = np.load("../simulations/sim_DW.npz")
    pos1 = data['x']
    vel1 = data['v']
    t1 = data['t']
    N_sat = pos0.shape[1]
    assert pos1.shape[1] == N_sat
    np.testing.assert_allclose(t0, t1)
    
    r0 = np.linalg.norm(pos0, axis=-1)
    r1 = np.linalg.norm(pos1, axis=-1)
    
    N_snaps = 5001
    h0 = np.zeros(N_snaps)
    h1 = np.zeros(N_snaps)
    f0 = np.zeros(N_snaps)
    f1 = np.zeros(N_snaps)
    for i in range(N_snaps):
        m0 = r0[i] < 400
        m1 = r1[i] < 400
        x0 = pos0[i, m0]
        x1 = pos1[i, m1]
        
        h0[i] = np.sqrt(np.mean(x0[:, 2]**2))
        h1[i] = np.sqrt(np.mean(x1[:, 2]**2))
        
# =============================================================================
#         t1 = (r0[i, m0]**2)[:, None, None] * np.ones((m0.sum(), 3, 3)) * np.eye(3)[None]
#         t2 = (x0[..., None] * x0[:, None])
#         I = t1 - t2
#         I = np.sum(I, axis=0)
#         e = np.linalg.eigvals(I)
#         f0[i] = e.min() / e.max()
#         
#         t1 = (r1[i, m1]**2)[:, None, None] * np.ones((m1.sum(), 3, 3)) * np.eye(3)[None]
#         t2 = (x1[..., None] * x1[:, None])
#         I = t1 - t2
#         I = np.sum(I, axis=0)
#         e = np.linalg.eigvals(I)
#         f1[i] = e.min() / e.max()
# =============================================================================
        
    # plot settings
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    
    # set up figure
    asp = 3.4 / 3.4
    X0 = 0.1275
    X1 = 0.98
    Y0 = 0.005
    dX = X1 - X0
    dY = asp * dX
    fig = plt.figure(figsize=(3.4, 3.4 / asp), dpi=150)
    ax1 = fig.add_axes([X0, Y0, dX, dY])

    # plot
    ax1.plot(t1, h0, label="No domain wall")
    ax1.plot(t1, h1, label="Domain wall")
