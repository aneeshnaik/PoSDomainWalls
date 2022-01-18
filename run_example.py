#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate satellites evolving in MW potential with domain wall.

Created: October 2021
Author: A. P. Naik
"""
import sys
sys.path.append("..")
from src.constants import MSUN, KPC, GYR
from src.simulation import Simulation
from src.satellites import sample_satellites


if __name__ == '__main__':

    # MW and DW params
    M_vir = 1e+12 * MSUN
    c_vir = 10
    a_DW = 5e-10
    l_DW = 10 * KPC

    # sample satellites
    pos, vel = sample_satellites(N_sat=100, M_vir=M_vir, c_vir=c_vir)

    # set up simulation
    sim = Simulation(
        pos0=pos, vel0=vel,
        M_vir=M_vir, c_vir=c_vir,
        a_DW=a_DW, l_DW=l_DW
    )

    # run simulation
    sim.run(t_max=3e+17)
    x = sim.positions / KPC
    t = sim.times / GYR
