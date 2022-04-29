#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to sample initial conditions (i.e. satellite positions + velocities).

Created: February 2022
Author: A. P. Naik
"""
import numpy as np
from .constants import KPC, PI
from .nfw import nfw_potential


def sample_satellites(N_sat, M_vir, c_vir, f_vesc, seed=42):
    """
    Sample initial positions and velocities of satellites.

    Parameters
    ----------
    N_sat : int
        Number of satellites to sample.
    M_vir : float
        Virial mass of host galaxy. UNITS: kg.
    c_vir : float
        Virial concentration of host galaxy.
    f_vesc : float
        Satellite speeds are uniformly sampled between 0 and f_vesc * v_esc
    seed : int, optional
        Random seed. The default is 42.

    Returns
    -------
    pos : numpy array, shape (N, 3)
        Positions of satellites (x, y, z). UNITS: m.
    vel : numpy array, shape (N, 3)
        Velocities of satellites (vx, vy, vz). UNITS: m/s.

    """
    # random seed
    rng = np.random.default_rng(seed)

    # sample positions
    r = rng.uniform(low=0, high=400 * KPC, size=N_sat)
    phi = rng.uniform(low=0, high=2 * PI, size=N_sat)
    theta = np.arccos(1 - 2 * rng.uniform(size=N_sat))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pos = np.stack((x, y, z), axis=-1)

    # velocities
    v_esc = np.sqrt(-2 * nfw_potential(pos, M_vir=M_vir, c_vir=c_vir))
    v = rng.uniform(low=np.zeros_like(v_esc), high=f_vesc * v_esc, size=N_sat)
    phi = rng.uniform(low=0, high=2 * PI, size=N_sat)
    theta = np.arccos(1 - 2 * rng.uniform(size=N_sat))
    vx = v * np.sin(theta) * np.cos(phi)
    vy = v * np.sin(theta) * np.sin(phi)
    vz = v * np.cos(theta)
    vel = np.stack((vx, vy, vz), axis=-1)

    return pos, vel
