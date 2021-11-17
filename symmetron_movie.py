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
kpc = 3.0857e+19
year = 31557600.0
Gyr = 1e+9 * year


def animate(i, im, rect, text, phi, z, times):

    # update image data
    im.set_data(phi[i].T)

    # update rect position
    rect.set_xy((-2, z[i] - 2))

    # update timer text
    t_str = f"{times[i]:.2f}"
    text.set_text(r"$t = " + t_str + r"\ \mathrm{Gyr}$")

    return


# load data
data = np.load("symmetron_DW_sim.npz")
phi = data['phi']
z = data['z'] / kpc
times = data['times'] / Gyr

# stack phi with reflection
phi = np.hstack((np.flip(phi, axis=1), phi))

# set up figure
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
fig = plt.figure(figsize=(9, 9))
left = 0.07
bottom = 0.1
dX = 0.8
dY = dX
cdY = 0.035
gap = 0.01
ax = fig.add_axes([left, bottom, dX, dY])
cax = fig.add_axes([left + dX + gap, bottom, cdY, dY])
imargs = {
    'origin': 'lower',
    'cmap': 'Spectral_r',
    'vmin': -1, 'vmax': 1,
    'extent': [-50, 50, -50, 50]
}

# labels, limits etc
ax.set_xlabel(r"$x\ [\mathrm{kpc}]$")
ax.set_ylabel(r"$z\ [\mathrm{kpc}]$")

# plot scalar field
im = ax.imshow(phi[0].T, **imargs)

# plot object outline
rect = plt.Rectangle((-2, z[0] - 2), 4, 4, fc='none', ec="black")
rect = ax.add_patch(rect)

# timer text
text = ax.text(0.98, 0.98, r"$t = 0.00\ \mathrm{Gyr}$", c='w',
               transform=ax.transAxes, fontsize=16, va='top', ha='right')

# colourbar
cbar = plt.colorbar(im, cax=cax)
cbar.set_label(r"$\varphi / \varphi_0$")

# animate
fargs = (im, rect, text, phi, z, times)
ani = FuncAnimation(fig, animate, 501, interval=25, fargs=fargs)
#ani.save("symmetron_movie.mp4")
#ani.save("symmetron_movie_small_dt.mp4")
ani.save("symmetron_movie_accs.mp4")