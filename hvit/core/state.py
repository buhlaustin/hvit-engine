"""
Core particle state matrix encapsulating SPH fluid arrays.
Vectorized layouts strictly enforce contiguous memory blocks for GPU transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class ParticleStateMatrix:
    N: int
    pos: npt.NDArray[np.float64]  # (N, 3) Position [m]
    vel: npt.NDArray[np.float64]  # (N, 3) Velocity [m/s]
    mass: npt.NDArray[np.float64]  # (N,)   Mass [kg]
    u: npt.NDArray[np.float64]  # (N,)   Specific Internal Energy [J/kg]
    rho: npt.NDArray[np.float64]  # (N,)   Density [kg/m^3]
    h: npt.NDArray[np.float64]  # (N,)   Smoothing Length [m]
    p: npt.NDArray[np.float64]  # (N,)   Pressure [Pa]
    ids: npt.NDArray[np.int64]  # (N,)   Unique Particle IDs

    @classmethod
    def allocate(cls, num_particles: int) -> Self:
        """Allocates zero-initialized C-contiguous arrays for optimal memory layout."""
        return cls(
            N=num_particles,
            pos=np.zeros((num_particles, 3), dtype=np.float64, order="C"),
            vel=np.zeros((num_particles, 3), dtype=np.float64, order="C"),
            mass=np.zeros(num_particles, dtype=np.float64),
            u=np.zeros(num_particles, dtype=np.float64),
            rho=np.zeros(num_particles, dtype=np.float64),
            h=np.zeros(num_particles, dtype=np.float64),
            p=np.zeros(num_particles, dtype=np.float64),
            ids=np.arange(num_particles, dtype=np.int64),
        )

    def compute_eos_ideal_gas(self, gamma: float = 5 / 3) -> None:
        """Vectorized Ideal Gas Equation of State update: P = (gamma - 1) * rho * u."""
        np.multiply((gamma - 1.0) * self.rho, self.u, out=self.p)

    def kinetic_energy(self) -> float:
        """Total kinetic energy [J]."""
        v_sq = np.einsum("ij,ij->i", self.vel, self.vel)
        return float(0.5 * np.sum(self.mass * v_sq))

    def internal_energy(self) -> float:
        """Total internal (thermal) energy [J]."""
        return float(np.sum(self.mass * self.u))

    def total_momentum(self) -> npt.NDArray[np.float64]:
        """Total linear momentum vector [kg m/s]."""
        return np.sum(self.mass[:, np.newaxis] * self.vel, axis=0)
