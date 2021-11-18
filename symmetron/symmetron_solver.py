#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D (axisymmetric cylindricals) time-dependent symmetron EOM solver.

Created: November 2021
Author: A. P. Naik
"""
import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import trange
kpc = 3.0857e+19
Mpc = 1e+3 * kpc
pi = np.pi
G = 6.6743e-11
M_sun = 1.988409870698051e+30
year = 31557600.0
Gyr = 1e+9 * year
c = 299792458.0


def L_op(phi, p, q):
    """Evaluate L operator."""
    L = phi * phi * phi + p * phi + q
    return L


def dLdphi_op(phi, p):
    """Evaluate L operator."""
    dLdphi = 3 * phi**2 + p
    return dLdphi


class Symm2DSolver:
    """2D symmetron solver."""

    def __init__(self, R_max, z_max, N_R, N_z):
        # check argument types
        if type(N_R) not in [int, np.int32, np.int64]:
            raise TypeError("Need integer N_R")
        if type(N_z) not in [int, np.int32, np.int64]:
            raise TypeError("Need integer N_z")
        if type(R_max) not in [float, np.float32, np.float64]:
            raise TypeError("Need float R_max")
        if type(z_max) not in [float, np.float32, np.float64]:
            raise TypeError("Need float z_max")

        # ensure N_z odd
        if not (N_z % 2):
            raise ValueError("Need odd N_z")

        # save arguments as attributes
        self.R_max = R_max
        self.z_max = z_max
        self.N_R = N_R
        self.N_z = N_z

        # index of mid-plane (i.e. x[:, idx] evaluates x along disc-plane)
        self.mid_idx = (N_z - 1) // 2

        # set up radial and angular cells
        R_edges = np.linspace(0, R_max, N_R + 1)
        R_cen = 0.5 * (R_edges[1:] + R_edges[:-1])
        z_edges = np.linspace(-z_max, z_max, N_z + 1)
        z_cen = 0.5 * (z_edges[1:] + z_edges[:-1])
        self.hR = np.diff(R_cen)[0]
        self.hz = np.diff(z_cen)[0]

        # save some useful coordinate grids
        R, z = np.meshgrid(R_cen, z_cen, indexing='ij')
        Rout, zout = np.meshgrid(R_edges[1:], z_edges[1:], indexing='ij')
        Rin, zin = np.meshgrid(R_edges[:-1], z_edges[:-1], indexing='ij')
        self.R_1D = R_cen
        self.z_1D = z_cen
        self.Rout_1D = R_edges[1:]
        self.Rin_1D = R_edges[:-1]
        self.zout_1D = z_edges[1:]
        self.zin_1D = z_edges[:-1]
        self.R = R
        self.z = z
        self.Rout = Rout
        self.Rin = Rin
        self.zout = zout
        self.zin = zin
        self.pos = np.stack((R, np.zeros_like(R), z), axis=2)

        # assign red/black cells; first cell (0, 0) is red
        self.black = (np.indices((N_R, N_z)).sum(axis=0) % 2).astype(bool)
        self.red = ~self.black

        # signs: +ve at z>0, -ve at z<0, zero at z=0
        self.sgns = np.sign(self.z)
        self.sgns[:, self.mid_idx] = 0

        return

    def solve_static(self, rho, rho_SSB, L_comp, verbose=False, tol=1e-8,
                     imin=100, imax=500000, set_zero_region=True):
        """Solve symmetron EOM for given density profile."""

        # check argument types
        if type(rho) is not np.ndarray:
            raise TypeError("Need numpy array for rho")
        if rho.dtype != np.float64:
            raise TypeError("rho should be np.float64, not", rho.dtype)
        if rho.shape != (self.N_R, self.N_z):
            raise ValueError("rho has wrong shape: should be (N_R, N_z)")
        if type(rho_SSB) not in [float, np.float32, np.float64]:
            raise TypeError("Need float M_symm")
        if type(L_comp) not in [float, np.float32, np.float64]:
            raise TypeError("Need float l0_symm")

        # start timer
        t0 = time()

        # compute mean matter density
        h = 0.7
        omega_m = 0.3
        H0 = h * 100.0 * 1000.0 / Mpc
        rhocrit = 3.0 * H0**2 / (8.0 * pi * G)
        rho_mean = omega_m * rhocrit

        # if SSB not happened yet, scalar field is 0 everywhere
        if rho_SSB < rho_mean:
            self.u = np.zeros_like(rho)
            self.u_inf = 0.
            self.time_taken = 0.
            self.n_iter = 0
            return

        # some useful coeffs
        C1 = 2 * L_comp**2 * self.Rout / (self.R * self.hR**2)
        C2 = 2 * L_comp**2 * self.Rin / (self.R * self.hR**2)
        C3 = 2 * L_comp**2 / self.hz**2

        # calculate p
        t1 = rho / rho_SSB
        t2 = 2 * L_comp**2 * (self.Rout + self.Rin) / (self.R * self.hR**2)
        t3 = 2 * L_comp**2 * (2 / self.hz**2)
        p = t1 + t2 + t3 - 1

        # start with guess: vev if rho<rhocrit, 0 otherwise
        self.u_inf = np.sqrt(1 - rho_mean / rho_SSB)
        u_guess = np.ones_like(rho) * self.u_inf
        if set_zero_region:
            u_guess[rho > rho_SSB] = 0
        u = np.copy(u_guess)

        # set to positive vev above mid-plane, negative below, 0 at mid-plane
        u = self.sgns * u

        # impose BCs
        u_ext = self._get_extended_field(u)

        # gauss-seidel relaxation
        i = 0
        du_max = 1
        while ((i < imin) or (du_max > tol)):

            # print progress if requested
            if verbose and (i % 1000 == 0):
                print(i, du_max, flush=True)

            # reimpose BCs
            u_ext = self._get_extended_field(u)

            # get offset arrays
            uip = u_ext[2:, 1:-1]
            uim = u_ext[:-2, 1:-1]
            ujp = u_ext[1:-1, 2:]
            ujm = u_ext[1:-1, :-2]

            # red sweep
            q = -C1 * uip - C2 * uim
            q += -C3 * (ujp + ujm)
            L = L_op(u, p, q)
            dL = dLdphi_op(u, p)
            du = -(self.red * np.divide(L, dL))
            u += du

            # reimpose BCs
            u_ext = self._get_extended_field(u)

            # get offset arrays
            uip = u_ext[2:, 1:-1]
            uim = u_ext[:-2, 1:-1]
            ujp = u_ext[1:-1, 2:]
            ujm = u_ext[1:-1, :-2]

            # black sweep
            q = -C1 * uip - C2 * uim
            q += -C3 * (ujp + ujm)
            L = L_op(u, p, q)
            dL = dLdphi_op(u, p)
            du = -(self.black * np.divide(L, dL))
            u += du

            # calculate residual
            du_max = np.max(np.abs(du))

            i += 1
            if i == imax:
                break

        self.u = u

        # stop timer
        t1 = time()
        self.time_taken = t1 - t0
        self.n_iter = i
        if verbose:
            print(f"Took {self.time_taken:.2f} seconds")
        return

    def solve_dynamic(self,
                      a0, rho_SSB, L_comp,
                      M, L_R, L_z,
                      z0, v0, dt, t_max,
                      N_snapshots, verbose):

        # N_iter is number of timesteps, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        self.N_snapshots = N_snapshots
        N_iter = int(t_max / dt)
        if N_iter % N_snapshots != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        snap_interval = int(N_iter / N_snapshots)

        # create arrays in which outputs are stored
        snapcount = 0
        self.phi = np.zeros((N_snapshots + 1, self.N_R, self.N_z))
        self.z = np.zeros((N_snapshots + 1))
        self.v = np.zeros((N_snapshots + 1))
        self.times = np.zeros((N_snapshots + 1))
        self.z[0] = z0
        self.v[0] = v0
        self.times[0] = 0

        # density profile: background
        rho_bg = calc_rho_mean(h=0.7, omega_m=0.3) * np.ones_like(self.R)

        # density profile: object
        rho = rho_bg + calc_rho_obj(M=M, L_R=L_R, L_z=L_z, z=z0, grid=self)

        # solve for initial field
        if verbose:
            print("Solving for initial field configuration...", flush=True)
        self.solve_static(rho=rho,
                          rho_SSB=rho_SSB, L_comp=L_comp,
                          verbose=verbose)
        u = self.u
        self.phi[0] = u

        # randomly initialise initial du/dt
        q = np.random.uniform(low=-1e-60, high=1e-60, size=u.shape)

        # desynchronise then main loop
        t = 0
        z = z0
        v = v0
        f = self._calc_ddotphi(u=u, rho=rho, rho_SSB=rho_SSB, L_comp=L_comp)
        a = self._calc_acc(z=z, u=u, a0=a0, L_comp=L_comp, L_z=L_z)
        q_half = q - f * dt / 2
        v_half = v - a * dt / 2
        if verbose:
            iterator = trange
        else:
            iterator = range
        for i in iterator(N_iter):

            # calculate scalar acc
            f = self._calc_ddotphi(u, rho, rho_SSB, L_comp)
            a = self._calc_acc(z, u, a0, L_comp, L_z)

            # timestep
            t += dt
            q_half = q_half + f * dt
            v_half = v_half + a * dt
            u = u + q_half * dt
            z = z + v_half * dt
            rho = rho_bg + calc_rho_obj(M=M, L_R=L_R, L_z=L_z, z=z, grid=self)

            # snapshot
            if (i + 1) % snap_interval == 0:

                snapcount += 1
                self.times[snapcount] = t
                self.z[snapcount] = z
                self.v[snapcount] = v_half - a * dt / 2
                self.phi[snapcount] = u

        return

    def _get_extended_field(self, u):
        # radial BCs, 0 deriv at centre, vev at infinity
        u_ext = np.vstack((u, self.u_inf * self.sgns[-1]))
        u_ext = np.vstack((u[0], u_ext))

        # vertical BCs, vev at +-z_max
        inf_layer = self.u_inf * np.ones(self.N_R + 2)[:, None]
        u_ext = np.hstack((u_ext, inf_layer))
        u_ext = np.hstack((-1 * inf_layer, u_ext))
        return u_ext

    def _calc_ddotphi(self, u, rho, rho_SSB, L_comp):

        # get extended field with BCs
        u_ext = self._get_extended_field(u)

        # get offset arrays
        uip = u_ext[2:, 1:-1]
        uim = u_ext[:-2, 1:-1]
        ujp = u_ext[1:-1, 2:]
        ujm = u_ext[1:-1, :-2]

        # laplacian
        t1 = (self.Rout * uip + self.Rin * uim) / (self.R * self.hR**2)
        t2 = -u * (self.Rout + self.Rin) / (self.R * self.hR**2)
        t3 = (ujp + ujm - 2 * u) / self.hz**2
        D = t1 + t2 + t3

        # ddotphi
        c2 = c**2
        t1 = D
        t2 = -((rho / rho_SSB - 1) * u) / (2 * L_comp**2)
        t3 = -u * u * u / (2 * L_comp**2)
        ddotphi = c2 * (t1 + t2 + t3)

        return ddotphi

    def _calc_acc(self, z, u, a0, L_comp, L_z, N_int=40):

        # set up interpolation grid
        z_min = z - L_z
        z_max = z + L_z
        N_int = 40
        z_int = np.linspace(z_min, z_max, N_int)

        # fifth force
        z_cen = self.z_1D[1:-1]
        u_cen = u[0, 1:-1]
        dudz = (u[0, 2:] - u[0, :-2]) / (self.z_1D[2:] - self.z_1D[:-2])
        a5 = -a0 * L_comp * u_cen * dudz

        # interpolate
        a_obj = np.interp(z_int, z_cen, a5).mean()
        return a_obj


def calc_rho_mean(h, omega_m):
    H0 = h * 100.0 * 1000.0 / Mpc
    rhocrit = 3.0 * H0**2 / (8.0 * pi * G)
    rho_mean = omega_m * rhocrit
    return rho_mean


def calc_rho_obj(M, L_R, L_z, z, grid):

    Rout = grid.Rout_1D
    zin = grid.zin_1D
    zout = grid.zout_1D

    rho_obj = M / (pi * L_R**2 * (2 * L_z))
    z_min = z - L_z
    z_max = z + L_z
    i1 = np.where(Rout < L_R)[0][-1]
    j0 = np.where(zin > z_min)[0][0]
    j1 = np.where(zout < z_max)[0][-1]
    fR1 = (L_R**2 - Rout[i1]**2) / (Rout[i1 + 1]**2 - Rout[i1]**2)
    fz0 = (zin[j0] - z_min) / grid.hz
    fz1 = (z_max - zout[j1]) / grid.hz
    rho = np.zeros_like(grid.R)
    rho[0:i1 + 1, j0:j1 + 1] += rho_obj
    rho[i1 + 1, j0:j1 + 1] = fR1 * rho_obj
    rho[0:i1 + 1, j0 - 1] += fz0 * rho_obj
    rho[0:i1 + 1, j1 + 1] += fz1 * rho_obj
    rho[i1 + 1, j0 - 1] += fR1 * fz0 * rho_obj
    rho[i1 + 1, j1 + 1] += fR1 * fz1 * rho_obj

    return rho


if __name__ == '__main__':

    # symmetron parameters
    rho_SSB = 1e-24
    L_comp = kpc
    a0 = 1e-10

    # object parameters
    M = 1e+8 * M_sun
    L_R = 2 * kpc
    L_z = 2 * kpc

    # initial object pos/vel
    z0 = -15 * kpc
    v0 = 200 * kpc / Gyr
    dt = 1e-7 * Gyr
    t_max = 0.25 * Gyr

    # set up solver
    R_max = 20 * kpc
    z_max = 50 * kpc
    s = Symm2DSolver(R_max=R_max, z_max=z_max, N_R=80, N_z=401)

    # solve
    s.solve_dynamic(a0=a0, rho_SSB=rho_SSB, L_comp=L_comp,
                    M=M, L_R=L_R, L_z=L_z,
                    z0=z0, v0=v0, dt=dt, t_max=t_max, N_snapshots=500,
                    verbose=True)

    # save
    np.savez("symmetron_DW_sim", times=s.times, z=s.z, v=s.v, phi=s.phi)
