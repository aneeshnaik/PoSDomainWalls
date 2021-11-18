#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

def construct_Rz(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    return R


def construct_Ry(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    return R


def convert_angles(theta, phi, theta0, phi0):
    # convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pos = np.stack((x, y, z))

    # rotate
    R1 = construct_Rz(-phi0)
    R2 = construct_Ry(-theta0)
    pos_new = np.matmul(R2, np.matmul(R1, pos))
    x_new = pos_new[0]
    y_new = pos_new[1]
    z_new = pos_new[2]

    # get angles
    phi_new = np.arctan2(y_new, x_new)
    theta_new = np.arctan2(np.sqrt(x_new**2 + y_new**2), z_new)
    return theta_new, phi_new


# load MW data
data = np.loadtxt("mw_sat_coords.txt", skiprows=1, usecols=[1, 2, 3])
phi = data[:, 0] * pi / 180
b = data[:, 1] * pi / 180
theta = pi / 2 - b
d_kpc = data[:, 2]

# convert to helio Cartesians
x_HC = d_kpc * np.sin(theta) * np.cos(phi)
y_HC = d_kpc * np.sin(theta) * np.sin(phi)
z_HC = d_kpc * np.cos(theta)

# convert to galactocentric Cartesians
x_MW = x_HC - 8
y_MW = np.copy(y_HC)
z_MW = np.copy(z_HC)


# load data
data = np.loadtxt("m31_sat_coords.txt", skiprows=1, usecols=[1, 2, 3])
phi = data[:, 0] * pi / 180
b = data[:, 1] * pi / 180
theta = pi / 2 - b
d_kpc = data[:, 2]

# convert to helio Cartesians
x_HC = d_kpc * np.sin(theta) * np.cos(phi)
y_HC = d_kpc * np.sin(theta) * np.sin(phi)
z_HC = d_kpc * np.cos(theta)

# convert to galactocentric Cartesians
x_M31 = x_HC - 8
y_M31 = np.copy(y_HC)
z_M31 = np.copy(z_HC)

# set up figure
fig = plt.figure(figsize=(8, 8))
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
ax = fig.add_subplot(projection='3d')

# plot satellites
ax.scatter(x_MW, y_MW, z_MW, label="MW satellites")
ax.scatter(x_M31, y_M31, z_M31, label="M31 satellites")

# plot MW marker
ax.scatter([0], [0], marker='x', s=15, c='k')

# legend, limits, labels
ax.legend(frameon=False)
ax.set_xlabel(r"$x\ [\mathrm{kpc}]$")
ax.set_ylabel(r"$y\ [\mathrm{kpc}]$")
ax.set_zlabel(r"$z\ [\mathrm{kpc}]$")
ax.set_xlim(-800, 800)
ax.set_ylim(-800, 800)
ax.set_zlim(-800, 800)
