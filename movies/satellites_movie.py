#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(i):

    # update sat positions
    coll1.set_offsets(x0[i, :, [0, 2]].T)
    coll2.set_offsets(x1[i, :, [0, 2]].T)
    coll3.set_offsets(x0[i, :, [0, 1]].T)
    coll4.set_offsets(x1[i, :, [0, 1]].T)

    # update trajectories
    for j in range(N_sat):
        lines1[j].set_data(x0[:i, j, 0], x0[:i, j, 2])
        lines2[j].set_data(x1[:i, j, 0], x1[:i, j, 2])
        lines3[j].set_data(x0[:i, j, 0], x0[:i, j, 1])
        lines4[j].set_data(x1[:i, j, 0], x1[:i, j, 1])

    # update timer text
    t_str = f"{t0[i]:.1f}"
    text.set_text(r"$t = " + t_str + r"\ \mathrm{Gyr}$")

    return coll1, coll2, coll3, coll4, lines1, lines2, lines3, lines4, text


# load data
data = np.load("../simulations/sim_noDW.npz")
x0 = data['x']
t0 = data['t']
data = np.load("../simulations/sim_DW.npz")
x1 = data['x']
t1 = data['t']
N_sat = x0.shape[1]
assert x1.shape[1] == N_sat
np.testing.assert_allclose(t0, t1)

# set up figure
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
asp = 9 / 9
fig = plt.figure(figsize=(9, 9 / asp))
left = 0.085
right = 0.875
bottom = 0.075
gap = 0.075
dX = (right - left) / 2
dY = asp * dX
ax1 = fig.add_axes([left, bottom + dY + gap, dX, dY])
ax2 = fig.add_axes([left + dX, bottom + dY + gap, dX, dY])
ax3 = fig.add_axes([left, bottom, dX, dY])
ax4 = fig.add_axes([left + dX, bottom, dX, dY])

# limits, labels etc
for ax in fig.axes:
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_xlabel(r"$x\ [\mathrm{kpc}]$")
    ax.tick_params(direction='inout', right=True, top=True)
ax1.set_ylabel(r"$z\ [\mathrm{kpc}]$")
ax3.set_ylabel(r"$y\ [\mathrm{kpc}]$")
ax2.tick_params(labelleft=False)
ax4.tick_params(labelleft=False)
ax1.text(0.5, 1.02, "No Wall", ha='center', va='bottom',
         transform=ax1.transAxes, zorder=100, fontsize=18)
ax2.text(0.5, 1.02, "Wall", ha='center', va='bottom',
         transform=ax2.transAxes, zorder=100, fontsize=18)
ax2.text(1.04, 0.5, "Edge-on", ha='left', va='center',
         transform=ax2.transAxes, zorder=100, fontsize=18)
ax4.text(1.04, 0.5, "Face-on", ha='left', va='center',
         transform=ax4.transAxes, zorder=100, fontsize=18)


# MW centre
for ax in fig.axes:
    ax.scatter([0], [0], marker='x', c='k', s=80)

# initial sat positions
c = plt.cm.viridis(np.linspace(0, 1, N_sat))
coll1 = ax1.scatter(x0[0, :, 0], x0[0, :, 2], c=c, zorder=10)
coll2 = ax2.scatter(x1[0, :, 0], x1[0, :, 2], c=c, zorder=10)
coll3 = ax3.scatter(x0[0, :, 0], x0[0, :, 1], c=c, zorder=10)
coll4 = ax4.scatter(x1[0, :, 0], x1[0, :, 1], c=c, zorder=10)

# initial trajectories
lines1 = []
lines2 = []
lines3 = []
lines4 = []
for i in range(N_sat):
    line = ax1.plot(x0[:0, i, 0], x0[:0, i, 2], ls='dotted', c='grey', lw=1)[0]
    lines1.append(line)
    line = ax2.plot(x1[:0, i, 0], x1[:0, i, 2], ls='dotted', c='grey', lw=1)[0]
    lines2.append(line)
    line = ax3.plot(x0[:0, i, 0], x0[:0, i, 1], ls='dotted', c='grey', lw=1)[0]
    lines3.append(line)
    line = ax4.plot(x1[:0, i, 0], x1[:0, i, 1], ls='dotted', c='grey', lw=1)[0]
    lines4.append(line)

# initial timer text
text = ax1.text(1, 1.02, r"$t = 0.0\ \mathrm{Gyr}$",
                transform=ax1.transAxes, fontsize=16, va='bottom', ha='center')

# create animation
ani = FuncAnimation(fig, animate, 5001, interval=3)
ani.save("satellites_movie.mp4")
