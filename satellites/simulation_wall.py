#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate satellites evolving in MW potential with domain wall.

Created: October 2021
Author: A. P. Naik
"""
import numpy as np
from tqdm import trange
G = 6.67408e-11
kpc = 3.0857e+19
pc = 3.0857e+16
pi = np.pi
M_sun = 1.9885e+30
Gyr = 31536000.0 * 1e+9


def NFW_param_conversion(M_vir, c_vir, delta=200, h=0.7):
    """
    For given virial mass and concentration, calculate NFW scale density and
    scale radius.

    Note that the virial radius here means the radius containing a region of
    average density equal to delta times the cosmic critical density. The
    virial mass is then the mass enclosed within this radius, while the virial
    concentration is the ratio of the virial radius to the NFW scale radius.

    Note that this conversion is very mildly cosmology dependent,
    as the definition of the virial radius depends on the critical density,
    which in turn depends on h.

    Parameters
    ----------
    M_vir: float
        Virial mass of halo. UNITS: kilograms.
    c_vir: float
        Virial concentration of halo. Dimensionless.
    delta: float, optional
        Virial ratio --- See note above. Dimensionless.
    h : float, optional
        Dimensionless Hubble constant. Default is 0.7

    Returns
    -------
    rho_0: float
        NFW scale density. UNITS: kilograms/metre^3
    R_s: float
        NFW scale radius. UNITS: metres.
    """

    # calculate critical density
    H0 = h * 100 * 1000 / (1e+6 * pc)
    rho_c = 3 * H0**2 / (8 * pi * G)

    # calculate virial radius
    R_vir = (3 * M_vir / (4 * pi * delta * rho_c))**(1 / 3)

    # calculate scale radius
    R_s = R_vir / c_vir

    # calculate rho_0
    denom = 4 * pi * R_s**3 * (np.log(1 + c_vir) - (c_vir / (1 + c_vir)))
    rho_0 = M_vir / denom

    return rho_0, R_s


def NFW_potential(pos, M_vir, c_vir):
    """
    Potential of NFW halo at given position.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_vir: float
        Virial mass of halo. UNITS: kilograms.
    c_vir: float
        Virial concentration of halo. Dimensionless.

    Returns
    -------
    phi : (N,) array or float, depending on shape of 'pos' parameter.
        Potential at given positions. UNITS: metres^2/seconds^2.
    """

    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    r = np.linalg.norm(pos, axis=-1)
    phi = - (4 * pi * G * rho_0 * r_s**3 / r) * np.log(1 + r / r_s)
    return phi


def NFW_acceleration(pos, M_vir, c_vir):
    """
    Acceleration due to NFW halo at given position.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate acceleration. UNITS: metres.
    M_vir: float
        Virial mass of halo. UNITS: kilograms.
    c_vir: float
        Virial concentration of halo. Dimensionless.

    Returns
    -------
    acc : array, same shape as 'pos' parameter.
        Acceleration at given positions. UNITS: m/s^2.
    """
    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    if pos.ndim == 1:
        r = np.linalg.norm(pos)
    else:
        r = np.linalg.norm(pos, axis=-1)[:, None]

    prefac = 4 * pi * G * rho_0 * r_s**3
    term1 = np.log(1 + r / r_s) / r**2
    term2 = 1 / (r * (r_s + r))
    acc = -prefac * (term1 - term2) * (pos / r)
    return acc


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
            a = NFW_acceleration(pos, M_vir=M_vir, c_vir=c_vir)
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
    M_vir = 1e+12 * M_sun
    c_vir = 10
    a_DW = 5e-10
    l_DW = 10 * kpc

    # random satellite positions
    rng = np.random.default_rng(42)
    N_sat = 100
    r = rng.uniform(low=0, high=400 * kpc, size=N_sat)
    phi = rng.uniform(low=0, high=2 * pi, size=N_sat)
    theta = np.arccos(1 - 2 * rng.uniform(size=N_sat))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pos = np.stack((x, y, z), axis=-1)

    # velocities
    v_esc = np.sqrt(-2 * NFW_potential(pos, M_vir=M_vir, c_vir=c_vir))
    v = rng.uniform(low=np.zeros_like(v_esc), high=0.5 * v_esc, size=N_sat)
    phi = rng.uniform(low=0, high=2 * pi, size=N_sat)
    theta = np.arccos(1 - 2 * rng.uniform(size=N_sat))
    vx = v * np.sin(theta) * np.cos(phi)
    vy = v * np.sin(theta) * np.sin(phi)
    vz = v * np.cos(theta)
    vel = np.stack((vx, vy, vz), axis=-1)

    # simulate
    sim = Simulation(pos0=pos, vel0=vel, M_vir=M_vir, c_vir=c_vir, a_DW=a_DW, l_DW=l_DW)
    sim.run(t_max=3e+17)
    x = sim.positions / kpc
    t = sim.times / Gyr

    # save
    #np.savez("wall", x=x, t=t)
