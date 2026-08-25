"""
SPH Solver Interface leveraging AMUSE execution harness for hydrodynamics
coupled with explicit internal relativistic integrations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from hvit.core.relativity import MSUN, RelativisticPotential
from hvit.core.state import ParticleStateMatrix

logger = logging.getLogger(__name__)

HydroBackend = Literal["native", "amuse-fi", "amuse-gadget2"]

# Monaghan (1992) cubic spline normalization in 3D.
_CUBIC_SPLINE_FLOOR: float = 1.0e-30


def cubic_spline_kernel(r: npt.NDArray[np.float64], h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cubic spline kernel W(r, h) for pairwise distances r and smoothing lengths h."""
    h_safe = np.maximum(h, _CUBIC_SPLINE_FLOOR)
    q = r / h_safe
    sigma = 1.0 / (np.pi * h_safe**3)

    w = np.zeros_like(r, dtype=np.float64)
    inner = (q >= 0.0) & (q <= 1.0)
    outer = (q > 1.0) & (q <= 2.0)

    qi = q[inner]
    qo = q[outer]
    w[inner] = sigma[inner] * (1.0 - 1.5 * qi**2 + 0.75 * qi**3)
    w[outer] = sigma[outer] * 0.25 * (2.0 - qo) ** 3
    return w


def cubic_spline_kernel_gradient(
    dr: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Gradient ∇W with respect to the first particle in each pair, dr = r_i - r_j."""
    h_safe = np.maximum(h, _CUBIC_SPLINE_FLOOR)
    q = r / h_safe
    sigma = 1.0 / (np.pi * h_safe**3)

    dw_dq = np.zeros_like(r, dtype=np.float64)
    inner = (q > 0.0) & (q <= 1.0)
    outer = (q > 1.0) & (q <= 2.0)

    qi = q[inner]
    qo = q[outer]
    dw_dq[inner] = sigma[inner] * (-3.0 * qi + 2.25 * qi**2)
    dw_dq[outer] = sigma[outer] * (-0.75 * (2.0 - qo) ** 2)

    dw_dr = dw_dq / h_safe
    r_safe = np.maximum(r, _CUBIC_SPLINE_FLOOR)
    return dw_dr[..., np.newaxis] * (dr / r_safe[..., np.newaxis])


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


@dataclass(slots=True)
class HydroForces:
    acceleration: npt.NDArray[np.float64]
    du_dt: npt.NDArray[np.float64]


class NativeSPHHydro:
    """Vectorized NumPy SPH hydrodynamics with blocked pairwise interactions."""

    def __init__(
        self,
        gamma: float = 5.0 / 3.0,
        eta: float = 1.2,
        alpha_av: float = 1.0,
        beta_av: float = 2.0,
        av_eps: float = 0.01,
        block_size: int = 128,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.alpha_av = alpha_av
        self.beta_av = beta_av
        self.av_eps = av_eps
        self.block_size = block_size

    def update_smoothing_lengths(self, state: ParticleStateMatrix) -> None:
        """Adaptive smoothing length h_i = eta * (m_i / rho_i)^(1/3)."""
        rho_safe = np.maximum(state.rho, _CUBIC_SPLINE_FLOOR)
        np.power(state.mass / rho_safe, 1.0 / 3.0, out=state.h)
        state.h *= self.eta

    def compute_density(self, state: ParticleStateMatrix) -> None:
        """SPH density estimate rho_i = sum_j m_j W_ij."""
        n = state.N
        rho = np.zeros(n, dtype=np.float64)
        mass = state.mass
        h = state.h
        pos = state.pos

        for i0 in range(0, n, self.block_size):
            i1 = min(i0 + self.block_size, n)
            dr = pos[i0:i1, np.newaxis, :] - pos[np.newaxis, :, :]
            r = np.linalg.norm(dr, axis=2)
            h_ij = 0.5 * (h[i0:i1, np.newaxis] + h[np.newaxis, :])
            q = r / np.maximum(h_ij, _CUBIC_SPLINE_FLOOR)
            active = q <= 2.0
            w = cubic_spline_kernel(r, h_ij) * active
            rho[i0:i1] = np.sum(mass[np.newaxis, :] * w, axis=1)

        np.maximum(rho, _CUBIC_SPLINE_FLOOR, out=state.rho)

    def sync_hydro_state(self, state: ParticleStateMatrix) -> None:
        """Refresh h, rho, and pressure from current positions."""
        self.update_smoothing_lengths(state)
        self.compute_density(state)
        state.compute_eos_ideal_gas(gamma=self.gamma)

    def compute_forces(self, state: ParticleStateMatrix) -> HydroForces:
        """Compute SPH pressure-gradient accelerations and du/dt."""
        n = state.N
        acc = np.zeros((n, 3), dtype=np.float64, order="C")
        du_dt = np.zeros(n, dtype=np.float64)

        mass = state.mass
        pos = state.pos
        vel = state.vel
        h = state.h
        rho = np.maximum(state.rho, _CUBIC_SPLINE_FLOOR)
        p = state.p
        sound = np.sqrt(np.maximum(self.gamma * p / rho, 0.0))

        p_over_rho2 = p / (rho**2)
        mass_row = mass[np.newaxis, :]

        for i0 in range(0, n, self.block_size):
            i1 = min(i0 + self.block_size, n)
            dr = pos[i0:i1, np.newaxis, :] - pos[np.newaxis, :, :]
            r = np.linalg.norm(dr, axis=2)
            h_ij = 0.5 * (h[i0:i1, np.newaxis] + h[np.newaxis, :])
            q = r / np.maximum(h_ij, _CUBIC_SPLINE_FLOOR)
            active = q <= 2.0

            grad_w = cubic_spline_kernel_gradient(dr, r, h_ij)
            grad_w *= active[..., np.newaxis]

            v_i = vel[i0:i1, np.newaxis, :]
            v_j = vel[np.newaxis, :, :]
            v_ij = v_i - v_j
            v_dot_r = np.sum(v_ij * dr, axis=2)

            rho_ij = 0.5 * (rho[i0:i1, np.newaxis] + rho[np.newaxis, :])
            c_ij = 0.5 * (sound[i0:i1, np.newaxis] + sound[np.newaxis, :])
            mu_ij = h_ij * v_dot_r / (r**2 + self.av_eps * h_ij**2)
            pi_ij = np.where(
                v_dot_r < 0.0,
                -self.alpha_av * c_ij * mu_ij + self.beta_av * mu_ij**2 / np.maximum(rho_ij, _CUBIC_SPLINE_FLOOR),
                0.0,
            )

            p_i = p_over_rho2[i0:i1, np.newaxis]
            p_j = p_over_rho2[np.newaxis, :]
            pressure_term = p_i + p_j + pi_ij
            pressure_term *= active

            acc[i0:i1] = -np.sum(
                mass_row[..., np.newaxis] * pressure_term[..., np.newaxis] * grad_w,
                axis=1,
            )

            v_dot_grad_w = np.sum(v_ij * grad_w, axis=2)
            du_dt[i0:i1] = 0.5 * np.sum(
                mass_row * pressure_term * v_dot_grad_w,
                axis=1,
            )

        return HydroForces(acceleration=acc, du_dt=du_dt)


def to_amuse_particles(state: ParticleStateMatrix):
    """Export ParticleStateMatrix to an AMUSE Particles set."""
    from amuse.datamodel import Particles
    from amuse.units import units

    particles = Particles(state.N)
    particles.x = state.pos[:, 0] | units.m
    particles.y = state.pos[:, 1] | units.m
    particles.z = state.pos[:, 2] | units.m
    particles.vx = state.vel[:, 0] | units.m / units.s
    particles.vy = state.vel[:, 1] | units.m / units.s
    particles.vz = state.vel[:, 2] | units.m / units.s
    particles.mass = state.mass | units.kg
    particles.u = state.u | units.J / units.kg
    particles.rho = state.rho | units.kg / units.m**3
    particles.h_smooth = state.h | units.m
    return particles


def from_amuse_particles(particles, state: ParticleStateMatrix) -> None:
    """Synchronize AMUSE Particles back into ParticleStateMatrix (SI units)."""
    from amuse.units import units

    state.pos[:, 0] = particles.x.value_in(units.m)
    state.pos[:, 1] = particles.y.value_in(units.m)
    state.pos[:, 2] = particles.z.value_in(units.m)
    state.vel[:, 0] = particles.vx.value_in(units.m / units.s)
    state.vel[:, 1] = particles.vy.value_in(units.m / units.s)
    state.vel[:, 2] = particles.vz.value_in(units.m / units.s)
    state.mass[:] = particles.mass.value_in(units.kg)
    state.u[:] = particles.u.value_in(units.J / units.kg)
    state.rho[:] = particles.rho.value_in(units.kg / units.m**3)
    state.h[:] = particles.h_smooth.value_in(units.m)


class AMUSEBridge:
    """AMUSE Fi/Gadget2 worker harness with graceful native fallback."""

    def __init__(
        self,
        state: ParticleStateMatrix,
        gamma: float = 5.0 / 3.0,
        prefer: Literal["fi", "gadget2"] = "fi",
    ) -> None:
        from amuse.units import units  # noqa: F401 - availability probe

        self.state = state
        self.gamma = gamma
        self.available = False
        self.backend_name: HydroBackend = "native"
        self._worker = None
        self._particles = None

        worker_errors: list[str] = []
        for name in (prefer, "gadget2" if prefer == "fi" else "fi"):
            try:
                if name == "fi":
                    from amuse.community.fi.interface import Fi

                    worker = Fi()
                    self.backend_name = "amuse-fi"
                else:
                    from amuse.community.gadget2.interface import Gadget2

                    worker = Gadget2()
                    self.backend_name = "amuse-gadget2"

                worker.parameters.set_defaults_for_current_epoch()
                worker.gamma = gamma
                self._worker = worker
                self._particles = to_amuse_particles(state)
                worker.gas_particles.add_particles(self._particles)
                worker.commit_particles()
                self.available = True
                logger.info("[HVIT ENGINE] AMUSE hydro backend active: %s", self.backend_name)
                return
            except Exception as exc:  # noqa: BLE001 - aggregate import/worker failures
                worker_errors.append(f"{name}: {exc}")

        logger.warning(
            "[HVIT ENGINE] AMUSE workers unavailable (%s); falling back to native NumPy SPH.",
            "; ".join(worker_errors),
        )

    def sync_to_worker(self) -> None:
        if not self.available or self._worker is None or self._particles is None:
            return
        from_amuse_particles(self._particles, self.state)
        self._worker.particles.remove_particles(self._worker.particles)
        self._particles = to_amuse_particles(self.state)
        self._worker.gas_particles.add_particles(self._particles)
        self._worker.commit_particles()

    def sync_from_worker(self) -> None:
        if not self.available or self._worker is None or self._particles is None:
            return
        self._particles = self._worker.gas_particles.copy()
        from_amuse_particles(self._particles, self.state)

    def step(self, dt: float) -> None:
        if not self.available or self._worker is None:
            raise RuntimeError("AMUSEBridge.step called while backend is unavailable.")
        from amuse.units import units

        self.sync_to_worker()
        self._worker.evolve_model(dt | units.s)
        self.sync_from_worker()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()


class HVITSolver:
    def __init__(
        self,
        state: ParticleStateMatrix,
        smbh_mass_msun: float,
        spin_a: float = 0.9,
        gamma: float = 5 / 3,
        eta: float = 1.2,
        alpha_av: float = 1.0,
        beta_av: float = 2.0,
        prefer_amuse: Literal["fi", "gadget2", "none"] = "fi",
        block_size: int = 128,
    ):
        self.state = state
        self.gamma = gamma
        self.relativity = RelativisticPotential(
            M_smbh=smbh_mass_msun * MSUN,
            spin_a=spin_a,
        )
        self._initial_total_energy: float | None = None

        self.native = NativeSPHHydro(
            gamma=gamma,
            eta=eta,
            alpha_av=alpha_av,
            beta_av=beta_av,
            block_size=block_size,
        )

        self.amuse: AMUSEBridge | None = None
        self.hydro_backend: HydroBackend = "native"
        if prefer_amuse != "none":
            self.amuse = AMUSEBridge(state, gamma=gamma, prefer=prefer_amuse)
            if self.amuse.available:
                self.hydro_backend = self.amuse.backend_name
            else:
                logger.warning(
                    "[HVIT ENGINE] Using native NumPy SPH hydrodynamics (eta=%.2f, alpha=%.1f).",
                    eta,
                    alpha_av,
                )
        else:
            logger.info("[HVIT ENGINE] AMUSE disabled by config; using native NumPy SPH.")

        self.initialize_hydro()

    def initialize_hydro(self) -> None:
        """Build consistent SPH density/pressure fields from current positions."""
        self.native.sync_hydro_state(self.state)

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

    def _compute_hydro_forces(self) -> HydroForces:
        return self.native.compute_forces(self.state)

    def _kick_hydro_amuse(self, dt: float) -> None:
        if self.amuse is not None and self.amuse.available:
            self.amuse.step(dt)
            self.native.sync_hydro_state(self.state)
            return
        raise RuntimeError("AMUSE hydro path invoked without an active worker.")

    def step_hydro_and_relativity(self, dt: float) -> None:
        """
        Symplectic Kick-Drift-Kick step integrating SPH fluid dynamics
        and relativistic post-Newtonian acceleration fields.
        """
        if self.hydro_backend != "native":
            acc_rel = self.relativity.compute_acceleration(self.state.pos, self.state.vel)
            self.state.vel += 0.5 * acc_rel * dt
            self.state.pos += self.state.vel * dt
            self._kick_hydro_amuse(dt)
            acc_rel_next = self.relativity.compute_acceleration(self.state.pos, self.state.vel)
            self.state.vel += 0.5 * acc_rel_next * dt
            return

        hydro = self._compute_hydro_forces()
        acc_rel = self.relativity.compute_acceleration(self.state.pos, self.state.vel)
        acc_total = hydro.acceleration + acc_rel

        self.state.vel += 0.5 * acc_total * dt
        self.state.u += 0.5 * hydro.du_dt * dt

        self.state.pos += self.state.vel * dt
        self.native.sync_hydro_state(self.state)

        hydro_next = self._compute_hydro_forces()
        acc_rel_next = self.relativity.compute_acceleration(self.state.pos, self.state.vel)
        acc_total_next = hydro_next.acceleration + acc_rel_next

        self.state.vel += 0.5 * acc_total_next * dt
        self.state.u += 0.5 * hydro_next.du_dt * dt

    def to_amuse_particles(self):
        """Export current state to AMUSE Particles."""
        return to_amuse_particles(self.state)

    def from_amuse_particles(self, particles) -> None:
        """Import AMUSE Particles into the state matrix."""
        from amuse.units import units  # noqa: F401 - ensure units registry side effects

        from_amuse_particles(particles, self.state)
        self.native.sync_hydro_state(self.state)

    def shutdown(self) -> None:
        if self.amuse is not None:
            self.amuse.stop()
