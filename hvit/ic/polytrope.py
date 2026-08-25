"""
Polytropic stellar profile generation via Lane-Emden solutions and 3D rejection sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import integrate
from scipy.interpolate import interp1d

from hvit.core.relativity import G, MSUN

RSUN: float = 6.957e8


@dataclass(slots=True)
class LaneEmdenSolution:
    """Dimensionless Lane-Emden polytropic structure."""

    n: float
    xi: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    dtheta_dxi: npt.NDArray[np.float64]
    xi_surface: float
    mass_integral: float  # ∫₀^ξ₁ ξ² θ^n dξ


@dataclass(slots=True)
class PolytropeSPH:
    """SPH particle snapshot for a single polytropic sphere in SI units."""

    num_particles: int
    pos: npt.NDArray[np.float64]  # (N, 3) [m]
    vel: npt.NDArray[np.float64]  # (N, 3) [m/s]
    mass: npt.NDArray[np.float64]  # (N,) [kg]
    rho: npt.NDArray[np.float64]  # (N,) [kg/m³]
    u: npt.NDArray[np.float64]  # (N,) [J/kg]
    h: npt.NDArray[np.float64]  # (N,) [m]
    rho_c: float  # central density [kg/m³]
    radius: float  # stellar radius [m]
    K: float  # polytropic constant [Pa·(m³/kg)^Γ]


def solve_lane_emden(n: float = 1.5, xi_max: float = 20.0) -> LaneEmdenSolution:
    """
    Solve the Lane-Emden equation for polytropic index n:

        (1/ξ²) d/dξ (ξ² dθ/dξ) = -θ^n

    with θ(0) = 1 and dθ/dξ(0) = 0. Integration terminates at the first zero crossing.
    """
    if n <= 0.0 or n > 4.999:
        raise ValueError(f"Polytropic index n={n} outside supported range (0, 5).")

    eps = 1.0e-8

    def ode(_xi: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        theta_val, dtheta = y
        theta_safe = max(theta_val, 0.0)
        xi_safe = max(_xi, eps)
        d2theta = -theta_safe**n - (2.0 / xi_safe) * dtheta
        return np.array([dtheta, d2theta], dtype=np.float64)

    y0 = np.array([1.0 - eps**2 / 6.0, -eps / 3.0], dtype=np.float64)

    def surface_event(_xi: float, y: npt.NDArray[np.float64]) -> float:
        return y[0]

    surface_event.terminal = True  # type: ignore[attr-defined]
    surface_event.direction = -1  # type: ignore[attr-defined]

    sol = integrate.solve_ivp(
        ode,
        (eps, xi_max),
        y0,
        method="RK45",
        events=surface_event,
        dense_output=False,
        max_step=0.01,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    if not sol.success or len(sol.t) < 2:
        raise RuntimeError(f"Lane-Emden integration failed for n={n}: {sol.message}")

    xi = sol.t
    theta = np.maximum(sol.y[0], 0.0)
    dtheta_dxi = sol.y[1]
    xi_surface = float(xi[-1])
    mass_integral = float(np.trapezoid(xi**2 * theta**n, xi))

    return LaneEmdenSolution(
        n=n,
        xi=xi,
        theta=theta,
        dtheta_dxi=dtheta_dxi,
        xi_surface=xi_surface,
        mass_integral=mass_integral,
    )


def _polytropic_constant(rho_c: float, alpha: float, n: float) -> float:
    """K from ρ_c and scale radius α for a polytrope of index n."""
    gamma = 1.0 + 1.0 / n
    return (4.0 * np.pi * G * alpha**2 * rho_c) / (n + 1.0) * gamma


def _sample_radius_from_profile(
    num_samples: int,
    xi: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
    n: float,
    alpha: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Inverse-CDF sampling of radial coordinate from mass shell distribution ∝ ξ² θ^n."""
    shell_mass = xi**2 * theta**n
    cumulative = np.cumsum(0.5 * (shell_mass[1:] + shell_mass[:-1]) * np.diff(xi))
    cumulative = np.concatenate(([0.0], cumulative))
    cumulative /= cumulative[-1]
    u = rng.random(num_samples)
    xi_sampled = np.interp(u, cumulative, xi)
    return alpha * xi_sampled


def generate_polytrope_sph(
    num_particles: int,
    mass_msun: float,
    radius_rsun: float,
    n: float = 1.5,
    gamma: float = 5.0 / 3.0,
    seed: int | None = None,
) -> PolytropeSPH:
    """
    Generate SPH particles for a polytropic sphere via 3D rejection sampling.

    Density profile: ρ(r) = ρ_c θ(r/α)^n from the Lane-Emden solution.
    Internal energy: u = P / ((γ - 1) ρ) with P = K ρ^Γ and Γ = 1 + 1/n.
    """
    if num_particles < 1:
        raise ValueError("num_particles must be >= 1.")
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 for ideal-gas closure.")

    le = solve_lane_emden(n)
    if le.mass_integral <= 0.0:
        raise RuntimeError("Lane-Emden mass integral is non-positive.")

    total_mass = mass_msun * MSUN
    radius = radius_rsun * RSUN

    alpha = radius / le.xi_surface
    rho_c = total_mass / (4.0 * np.pi * alpha**3 * le.mass_integral)
    K = _polytropic_constant(rho_c, alpha, n)
    gamma_poly = 1.0 + 1.0 / n

    theta_interp = interp1d(
        le.xi,
        le.theta,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    rng = np.random.default_rng(seed)
    pos = np.zeros((num_particles, 3), dtype=np.float64, order="C")
    accepted = 0
    batch = max(num_particles * 4, 1024)

    while accepted < num_particles:
        r_samp = _sample_radius_from_profile(batch, le.xi, le.theta, n, alpha, rng)
        cos_theta = rng.uniform(-1.0, 1.0, size=batch)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=batch)

        x = r_samp * sin_theta * np.cos(phi)
        y = r_samp * sin_theta * np.sin(phi)
        z = r_samp * cos_theta

        r_mag = np.sqrt(x**2 + y**2 + z**2)
        xi_local = np.clip(r_mag / alpha, 0.0, le.xi_surface)
        theta_local = theta_interp(xi_local)
        rho_local = rho_c * np.power(np.maximum(theta_local, 0.0), n)

        u_rand = rng.random(batch)
        accept = u_rand * rho_c <= rho_local
        n_accept = int(np.count_nonzero(accept))
        if n_accept == 0:
            continue

        end = min(accepted + n_accept, num_particles)
        n_write = end - accepted
        idx = np.flatnonzero(accept)[:n_write]
        pos[accepted:end, 0] = x[idx]
        pos[accepted:end, 1] = y[idx]
        pos[accepted:end, 2] = z[idx]
        accepted = end

    r_mag = np.linalg.norm(pos, axis=1)
    xi_part = np.clip(r_mag / alpha, 0.0, le.xi_surface)
    theta_part = theta_interp(xi_part)
    rho = rho_c * np.power(np.maximum(theta_part, 0.0), n)
    p = K * np.power(rho, gamma_poly)
    u = p / ((gamma - 1.0) * np.maximum(rho, np.finfo(np.float64).tiny))

    # Equal-mass particles: m_i = M / N; smoothing length from local number density.
    mass = np.full(num_particles, total_mass / num_particles, dtype=np.float64)
    h = 2.0 * np.cbrt(3.0 * mass / (4.0 * np.pi * np.maximum(rho, np.finfo(np.float64).tiny)))

    vel = np.zeros((num_particles, 3), dtype=np.float64, order="C")

    return PolytropeSPH(
        num_particles=num_particles,
        pos=pos,
        vel=vel,
        mass=mass,
        rho=rho,
        u=u,
        h=h,
        rho_c=float(rho_c),
        radius=float(radius),
        K=float(K),
    )
