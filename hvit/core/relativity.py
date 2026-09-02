"""
Post-Newtonian correction factors for relativistic orbital dynamics around a
Supermassive Black Hole (SMBH) including Lense-Thirring frame dragging.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Physical Constants
G: float = 6.67430e-11
C: float = 2.99792458e8
MSUN: float = 1.9884e30


class RelativisticPotential:
    def __init__(self, M_smbh: float, spin_a: float = 0.9):
        """
        :param M_smbh: SMBH mass in kg.
        :param spin_a: Dimensionless Kerr spin parameter |a| <= 1.
        """
        self.M_smbh = M_smbh
        self.spin_a = min(max(spin_a, 0.0), 0.998)
        self.rg = (G * M_smbh) / (C**2)  # Gravitational radius

    def _regularized_radius(self, r_mag: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r_eff = r_mag - (2.0 * self.rg)
        np.maximum(r_eff, 1.0e-3 * self.rg, out=r_eff)
        return r_eff

    def potential_energy_per_mass(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Paczynski-Wiita pseudo-Newtonian potential [J/kg]."""
        r_mag = np.linalg.norm(pos, axis=1)
        r_eff = self._regularized_radius(r_mag)
        return -(G * self.M_smbh) / r_eff

    def compute_acceleration(
        self, pos: npt.NDArray[np.float64], vel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Paczynski-Wiita pseudo-Newtonian potential with 1.5PN Lense-Thirring frame-dragging.
        Vectorized over N particles.
        """
        r_vec = pos
        r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)

        # Paczynski-Wiita radial acceleration magnitude: a_r = -G M / (r - 2 r_g)^2
        r_eff = self._regularized_radius(r_mag.squeeze())
        r_eff = r_eff[:, np.newaxis]
        a_pw_mag = -(G * self.M_smbh) / (r_eff**2 * r_mag)
        a_pw = a_pw_mag * r_vec

        # Lense-Thirring acceleration with SMBH spin axis aligned to +z.
        # a_lt = (2G / c^2 r^3) * [3/r^2 (J . (r x v)) r + (v x J)]
        J_vec = np.array([0.0, 0.0, self.spin_a * self.M_smbh * self.rg * C], dtype=np.float64)

        r_cross_v = np.cross(r_vec, vel)
        J_dot_r_cross_v = np.sum(J_vec * r_cross_v, axis=1, keepdims=True)
        v_cross_J = np.cross(vel, J_vec)

        factor = (2.0 * G) / ((C**2) * (r_mag**3))
        a_lt = factor * ((3.0 / (r_mag**2)) * J_dot_r_cross_v * r_vec + v_cross_J)

        return a_pw + a_lt

    def gravitational_energy(self, pos: npt.NDArray[np.float64], mass: npt.NDArray[np.float64]) -> float:
        """Total Paczynski-Wiita binding energy [J]."""
        phi = self.potential_energy_per_mass(pos)
        return float(np.sum(mass * phi))
