#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import numpy as np


def wall_acceleration(pos, a_DW, l_DW):
    if pos.ndim == 1:
        z = pos[2]
    else:
        z = pos[..., 2]
    acc = np.zeros_like(pos)
    acc[..., 2] = -a_DW * np.tanh(z / (2 * l_DW)) / np.cosh(z / (2 * l_DW))**2
    return acc
