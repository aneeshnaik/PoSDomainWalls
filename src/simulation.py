#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation code.

Created: January 2022
Author: A. P. Naik
"""
import numpy as np
from tqdm import trange
from .nfw import nfw_acceleration
from .wall import wall_acceleration


class Simulation:
    """
    Simulation of point satellites around NFW halo and domain wall.

    Parameters
    ----------
    pos0 : numpy array, shape (N, 3)
        Initial positions of N satellites. UNITS: m.
    vel0 : numpy array, shape (N, 3)
        Initial velocities of N satellites. UNITS: m/s.
    M_vir : float
        Virial mass of host galaxy.
    c_vir : float
        Virial concentration of host galaxy.
    a_DW : float
        Domain wall acceleration.
    l_DW : float
        Domain wall width.

    Methods
    -------
    run(t_max, N_snapshots=500, dt=1e+12):
        Run simulation.

    """

    def __init__(self, pos0, vel0, M_vir, c_vir, a_DW, l_DW):
        """Initialise Simulation. See class docstring for details."""

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

    def run(self, t_max, N_snapshots=500, dt=1e+12):
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
        dt : float, optional
            Timestep size. UNITS: seconds. Default is 1e+12 seconds.
        """
        # set up time parameters
        self.times = np.array([0])
        self.t_max = t_max
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
