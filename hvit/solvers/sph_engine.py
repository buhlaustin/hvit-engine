"""
SPH Solver Interface leveraging AMUSE execution harness for hydrodynamics
coupled with explicit internal relativistic integrations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hvit.core.relativity import MSUN, RelativisticPotential
from hvit.core.state import ParticleStateMatrix


@dataclass(slots=True)
class SimulationDiagnostics:
    step: int
    time: float
    kinetic_energy: float
    potential_energy: float
    internal_energy: float
    total_energy: float
    momentum_x: float
    momentum_y: float
    momentum_z: float

    @property
    def momentum_magnitude(self) -> float:
        return float(np.sqrt(self.momentum_x**2 + self.momentum_y**2 + self.momentum_z**2))


class HVITSolver:
    def __init__(
        self,
        state: ParticleStateMatrix,
        smbh_mass_msun: float,
        spin_a: float = 0.9,
        gamma: float = 5 / 3,
    ):
        self.state = state
        self.gamma = gamma
        self.relativity = RelativisticPotential(
            M_smbh=smbh_mass_msun * MSUN,
            spin_a=spin_a,
        )
        self._initial_total_energy: float | None = None

    def compute_diagnostics(self, step: int, time: float) -> SimulationDiagnostics:
        """Energy and momentum monitors for numerical stability tracking."""
        kinetic = self.state.kinetic_energy()
        potential = self.relativity.gravitational_energy(self.state.pos, self.state.mass)
        internal = self.state.internal_energy()
        total = kinetic + potential + internal
        momentum = self.state.total_momentum()

        if self._initial_total_energy is None:
            self._initial_total_energy = total

        return SimulationDiagnostics(
            step=step,
            time=time,
            kinetic_energy=kinetic,
            potential_energy=potential,
            internal_energy=internal,
            total_energy=total,
            momentum_x=float(momentum[0]),
            momentum_y=float(momentum[1]),
            momentum_z=float(momentum[2]),
        )

    def step_hydro_and_relativity(self, dt: float) -> None:
        """
        Symplectic Kick-Drift-Kick (Leapfrog) step integrating SPH fluid dynamics
        and relativistic post-Newtonian acceleration fields.

        SPH hydrodynamics (density/pressure forces) will be delegated to AMUSE
        Gadget2/Fi workers in a subsequent coupling pass; this minimal loop applies
        the relativistic external field and ideal-gas EOS closure.
        """
        # Half-step velocity update (Kick)
        acc_rel = self.relativity.compute_acceleration(self.state.pos, self.state.vel)
        self.state.vel += 0.5 * acc_rel * dt

        # Full-step position update (Drift)
        self.state.pos += self.state.vel * dt

        # Synchronize fluid states & re-calculate pressure field
        self.state.compute_eos_ideal_gas(gamma=self.gamma)

        # Final half-step velocity update (Kick)
        acc_rel_next = self.relativity.compute_acceleration(self.state.pos, self.state.vel)
        self.state.vel += 0.5 * acc_rel_next * dt

    def to_amuse_particles(self):
        """
        Export current state to an AMUSE Particles instance for downstream
        SPH/gravity worker coupling.
        """
        from amuse.datamodel import Particles
        from amuse.units import units

        particles = Particles(self.state.N)
        particles.x = self.state.pos[:, 0] | units.m
        particles.y = self.state.pos[:, 1] | units.m
        particles.z = self.state.pos[:, 2] | units.m
        particles.vx = self.state.vel[:, 0] | units.m / units.s
        particles.vy = self.state.vel[:, 1] | units.m / units.s
        particles.vz = self.state.vel[:, 2] | units.m / units.s
        particles.mass = self.state.mass | units.kg
        particles.u = self.state.u | units.m * units.m / units.s**2
        particles.rho = self.state.rho | units.kg / units.m**3
        particles.h_smooth = self.state.h | units.m
        return particles
