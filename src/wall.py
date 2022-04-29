#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain wall acceleration.

Created: January 2022
Author: A. P. Naik
"""
import numpy as np


def wall_acceleration(pos, a_DW, l_DW):
    """
    Acceleration due to domain wall.

    Parameters
    ----------
    pos : numpy array, shape (3) or (..., 3)
        Position(s) at which to evaluate acceleration. UNITS: m.
    a_DW : float
        DW characteristic acceleration parameter.
    l_DW : TYPE
        DW width parameter.

    Returns
    -------
    acc : numpy array, shape same as pos
        Accelerations at given positions. UNITS: m/s^2.

    """
    if pos.ndim == 1:
        z = pos[2]
    else:
        z = pos[..., 2]
    acc = np.zeros_like(pos)
    acc[..., 2] = -a_DW * np.tanh(z / (2 * l_DW)) / np.cosh(z / (2 * l_DW))**2
    return acc
