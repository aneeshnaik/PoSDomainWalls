#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate satellites evolving in MW potential with domain wall.

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import os
sys.path.append("../..")
from src.constants import MSUN, KPC, GYR
from src.simulation import Simulation
from src.satellites import sample_satellites


if __name__ == '__main__':

    # MW and DW params
    M_vir = 1e+12 * MSUN
    c_vir = 10
    a_DW = 1e-11
    l_DW = 10 * KPC

    # sample satellites
    x, v = sample_satellites(N_sat=2500, M_vir=M_vir, c_vir=c_vir, f_vesc=0.98)

    # set up simulations
    sim = Simulation(
        pos0=x, vel0=v,
        M_vir=M_vir, c_vir=c_vir,
        a_DW=a_DW, l_DW=l_DW
    )

    # run simulation
    sim.run(t_max=3e+18, N_snapshots=5000, dt=1e+12)
    savedir = os.environ['POSDWDDIR']
    np.savez(f"{savedir}/extra_params/sim_alo",
             x=sim.positions / KPC, v=sim.velocities / 1000,
             t=sim.times / GYR)
