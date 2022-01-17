#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate satellites evolving in MW potential with domain wall.

Created: October 2021
Author: A. P. Naik
"""
import sys
import numpy as np
from tqdm import trange
sys.path.append("..")
from src.constants import MSUN, KPC, PI, GYR
from src.nfw import nfw_acceleration, nfw_potential



def wall_acceleration(pos, a_DW, l_DW):
    if pos.ndim == 1:
        z = pos[2]
    else:
        z = pos[..., 2]
    acc = np.zeros_like(pos)
    acc[..., 2] = -a_DW * np.tanh(z / (2 * l_DW)) / np.cosh(z / (2 * l_DW))**2
    return acc


class Simulation:

    def __init__(self, pos0, vel0, M_vir, c_vir, a_DW, l_DW):
        # need to define: self.acc
        assert pos0.shape == vel0.shape
        assert pos0.ndim == 2

        self.x = pos0
        self.v = vel0
        self.N = vel0.shape[0]

        def acc(pos):
            a = nfw_acceleration(pos, M_vir=M_vir, c_vir=c_vir)
            a += wall_acceleration(pos, a_DW=a_DW, l_DW=l_DW)
            return a
        self.acc = acc

        return

    def run(self, t_max, N_snapshots=500):
        """
        Run simulation.

        Parameters
        ----------
        t_max : float
            Total run time. Default is 1e+17 seconds, around 3 Gyr.
            UNITS: seconds
        N_snapshots : int
            Number of snapshots to store, EXCLUDING initial snapshot.
            Default is 500 (so 501 snapshots are saved overall).
        """
        # set up time parameters
        self.times = np.array([0])
        self.t_max = t_max
        dt = 1e+12
        self.dt = dt

        # N_iter is number of timesteps, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        self.N_snapshots = N_snapshots
        N_iter = int(t_max / dt)
        if N_iter % N_snapshots != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        snap_interval = int(N_iter / N_snapshots)

        # create arrays in which outputs are stored
        snapcount = 0
        positions = np.zeros((N_snapshots + 1, self.N, 3))
        velocities = np.zeros((N_snapshots + 1, self.N, 3))
        positions[0] = self.x
        velocities[0] = self.v

        # calculate initial accelerations, then desynchronise velocities
        # for leapfrog integration
        acc = self.acc(self.x)
        v_half = self.v - 0.5 * dt * acc

        # main loop
        t = 0
        print("Main loop...")
        for i in trange(N_iter):

            # calculate accelerations
            acc = self.acc(self.x)

            # timestep
            t += self.dt
            v_half = v_half + acc * dt
            self.x = self.x + v_half * dt

            # snapshot
            if (i + 1) % snap_interval == 0:

                snapcount += 1
                self.times = np.append(self.times, t)

                # resynchronise tracer velocities
                self.v = v_half - 0.5 * acc * dt

                # store positions and velocities
                positions[snapcount] = self.x
                velocities[snapcount] = self.v

        # store pos/vel arrays
        self.positions = positions
        self.velocities = velocities

        # resynchronise velocities
        self.v = v_half - 0.5 * dt * acc

        return


if __name__ == '__main__':

    # MW and DW params
    M_vir = 1e+12 * MSUN
    c_vir = 10
    a_DW = 5e-10
    l_DW = 10 * KPC

    # random satellite positions
    rng = np.random.default_rng(42)
    N_sat = 100
    r = rng.uniform(low=0, high=400 * KPC, size=N_sat)
    phi = rng.uniform(low=0, high=2 * PI, size=N_sat)
    theta = np.arccos(1 - 2 * rng.uniform(size=N_sat))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pos = np.stack((x, y, z), axis=-1)

    # velocities
    v_esc = np.sqrt(-2 * nfw_potential(pos, M_vir=M_vir, c_vir=c_vir))
    v = rng.uniform(low=np.zeros_like(v_esc), high=0.5 * v_esc, size=N_sat)
    phi = rng.uniform(low=0, high=2 * PI, size=N_sat)
    theta = np.arccos(1 - 2 * rng.uniform(size=N_sat))
    vx = v * np.sin(theta) * np.cos(phi)
    vy = v * np.sin(theta) * np.sin(phi)
    vz = v * np.cos(theta)
    vel = np.stack((vx, vy, vz), axis=-1)

    # simulate
    sim = Simulation(pos0=pos, vel0=vel, M_vir=M_vir, c_vir=c_vir, a_DW=a_DW, l_DW=l_DW)
    sim.run(t_max=3e+17)
    x = sim.positions / KPC
    t = sim.times / GYR

    # save
    #np.savez("wall", x=x, t=t)
