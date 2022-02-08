#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np
from .constants import KPC, PI
from .nfw import nfw_potential


def sample_satellites(N_sat, M_vir, c_vir, seed=42, f_vesc=0.5):

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
