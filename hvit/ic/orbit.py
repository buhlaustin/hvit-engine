"""
Relativistic binary impact orbit injection for polytropic stellar initial conditions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hvit.core.relativity import C, G, MSUN
from hvit.core.state import ParticleStateMatrix
from hvit.ic.polytrope import PolytropeSPH, generate_polytrope_sph


def _hyperbolic_orbit_vectors(
    mu: float,
    r0: float,
    r_p: float,
    eccentricity: float,
    v_inf: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute inbound center-of-mass position and velocity at radius r0 for a Keplerian
    hyperbolic orbit with periapsis r_p, eccentricity e, and speed at infinity v_inf.

    Orbit lies in the x-y plane with periapsis on +x and inbound approach from -x.
    """
    if eccentricity <= 1.0:
        raise ValueError(f"Hyperbolic orbit requires eccentricity > 1, got {eccentricity}.")
    if r0 <= r_p:
        raise ValueError(f"Initial radius r0={r0} must exceed periapsis r_p={r_p}.")
    if v_inf <= 0.0:
        raise ValueError("v_inf must be positive.")

    # Specific angular momentum fixed by periapsis: h² = GM r_p (1 + e)
    h_mag = np.sqrt(max(0.0, mu * r_p * (1.0 + eccentricity)))
    v0 = np.sqrt(max(0.0, v_inf**2 + 2.0 * mu / r0))
    v_t = h_mag / r0
    v_r_sq = v0**2 - v_t**2
    if v_r_sq < 0.0:
        raise ValueError(
            f"Inconsistent orbit: v_t={v_t:.3e} exceeds v_0={v0:.3e} at r_0={r0:.3e}."
        )
    v_r = np.sqrt(v_r_sq)

    # COM on -x axis, inbound toward SMBH at origin; +y is prograde tangential direction.
    pos = np.array([-r0, 0.0, 0.0], dtype=np.float64)
    vel = np.array([v_r, v_t, 0.0], dtype=np.float64)
    return pos, vel


def _write_star_slice(
    state: ParticleStateMatrix,
    start: int,
    end: int,
    star: PolytropeSPH,
    offset: npt.NDArray[np.float64],
    bulk_velocity: npt.NDArray[np.float64],
    id_offset: int,
) -> None:
    """Copy polytrope arrays into a contiguous slice of the state matrix."""
    n = end - start
    if star.num_particles != n:
        raise ValueError(f"Expected {n} particles, polytrope has {star.num_particles}.")

    state.pos[start:end] = star.pos + offset
    state.vel[start:end] = star.vel + bulk_velocity
    state.mass[start:end] = star.mass
    state.rho[start:end] = star.rho
    state.u[start:end] = star.u
    state.h[start:end] = star.h
    state.ids[start:end] = np.arange(id_offset, id_offset + n, dtype=np.int64)


def inject_binary_collision(
    state: ParticleStateMatrix,
    M_smbh_msun: float,
    r_p_rsch: float,
    eccentricity: float,
    v_inf_c: float,
    m1_msun: float = 1.0,
    r1_rsun: float = 1.0,
    m2_msun: float = 0.8,
    r2_rsun: float = 0.85,
    n1: float = 1.5,
    n2: float = 1.5,
    gamma: float = 5.0 / 3.0,
    r0_factor: float = 50.0,
    separation_factor: float = 2.5,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Populate ``state`` with two polytropic stars on a hyperbolic plunge orbit around an SMBH.

    Particles are split 50/50 between primary and secondary. Periapsis distance is
    specified in Schwarzschild radii: r_s = 2GM/c², r_p = r_p_rsch · r_s.
    Initial center-of-mass radius is r_0 = r0_factor · r_p (default 50 r_p).

    Returns orbital metadata dict (r_s, r_p, r_0, v_inf [m/s]).
    """
    if state.N < 2:
        raise ValueError("Binary collision requires at least 2 particles.")

    n1_particles = state.N // 2
    n2_particles = state.N - n1_particles

    M_smbh = M_smbh_msun * MSUN
    r_s = 2.0 * G * M_smbh / (C**2)
    r_p = r_p_rsch * r_s
    r_0 = r0_factor * r_p
    v_inf = v_inf_c * C

    rng = np.random.default_rng(seed)
    star1 = generate_polytrope_sph(
        n1_particles,
        mass_msun=m1_msun,
        radius_rsun=r1_rsun,
        n=n1,
        gamma=gamma,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    star2 = generate_polytrope_sph(
        n2_particles,
        mass_msun=m2_msun,
        radius_rsun=r2_rsun,
        n=n2,
        gamma=gamma,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    com_pos, com_vel = _hyperbolic_orbit_vectors(
        mu=G * M_smbh,
        r0=r_0,
        r_p=r_p,
        eccentricity=eccentricity,
        v_inf=v_inf,
    )

    separation = separation_factor * (star1.radius + star2.radius)
    offset1 = np.array(
        [0.0, -m2_msun / (m1_msun + m2_msun) * separation, 0.0],
        dtype=np.float64,
    )
    offset2 = np.array(
        [0.0, m1_msun / (m1_msun + m2_msun) * separation, 0.0],
        dtype=np.float64,
    )

    _write_star_slice(state, 0, n1_particles, star1, com_pos + offset1, com_vel, id_offset=0)
    _write_star_slice(
        state,
        n1_particles,
        state.N,
        star2,
        com_pos + offset2,
        com_vel,
        id_offset=n1_particles,
    )

    state.compute_eos_ideal_gas(gamma=gamma)

    return {
        "r_schwarzschild_m": float(r_s),
        "r_periapsis_m": float(r_p),
        "r_initial_m": float(r_0),
        "v_infinity_m_s": float(v_inf),
        "eccentricity": float(eccentricity),
    }
