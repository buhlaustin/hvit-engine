"""HVIT engine pipeline entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import time
from typing import Any

import yaml

from hvit.core.state import ParticleStateMatrix
from hvit.ic.orbit import inject_binary_collision
from hvit.io.hdf5_sink import AsyncHDF5Sink
from hvit.solvers.sph_engine import HVITSolver


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def initialize_state(cfg: dict[str, Any]) -> tuple[ParticleStateMatrix, dict[str, float]]:
    sim = cfg["simulation"]
    smbh = cfg["smbh"]
    primary = cfg["stellar_primary"]
    secondary = cfg["stellar_secondary"]
    orbit = cfg["orbit"]
    gamma = float(cfg.get("eos", {}).get("gamma", 5 / 3))

    n = int(sim["num_particles"])
    state = ParticleStateMatrix.allocate(n)

    orbit_meta = inject_binary_collision(
        state,
        M_smbh_msun=float(smbh["mass_msun"]),
        r_p_rsch=float(orbit["r_p_rsch"]),
        eccentricity=float(orbit["eccentricity"]),
        v_inf_c=float(orbit["v_infinity_c"]),
        m1_msun=float(primary["mass_msun"]),
        r1_rsun=float(primary["radius_rsun"]),
        m2_msun=float(secondary["mass_msun"]),
        r2_rsun=float(secondary["radius_rsun"]),
        n1=float(primary.get("polytropic_index", 1.5)),
        n2=float(secondary.get("polytropic_index", 1.5)),
        gamma=gamma,
        r0_factor=float(orbit.get("r0_factor", 50.0)),
        separation_factor=float(orbit.get("separation_factor", 2.5)),
        seed=orbit.get("seed"),
    )
    return state, orbit_meta


async def run_simulation(config_path: pathlib.Path) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = load_config(config_path)
    sim = cfg["simulation"]
    smbh = cfg["smbh"]
    io_cfg = cfg.get("io", {})
    sph_cfg = cfg.get("sph", {})

    n = int(sim["num_particles"])
    dt_max = float(sim["dt"])
    total_steps = int(sim["total_steps"])
    write_interval = int(sim["write_interval"])
    end_time = total_steps * dt_max
    gamma = float(cfg.get("eos", {}).get("gamma", 5 / 3))
    prefer_amuse = sph_cfg.get("prefer_amuse", "fi")
    if prefer_amuse is not None and str(prefer_amuse).lower() == "none":
        prefer_amuse = "none"

    print(f"[HVIT ENGINE] Initializing {n:,} fluid particle state matrix...")
    state, orbit_meta = initialize_state(cfg)
    print(
        f"[HVIT ENGINE] Orbit: r_p={orbit_meta['r_periapsis_m']:.3e} m "
        f"({cfg['orbit']['r_p_rsch']} r_s), "
        f"r_0={orbit_meta['r_initial_m']:.3e} m, "
        f"v_inf={orbit_meta['v_infinity_m_s']:.3e} m/s"
    )

    solver = HVITSolver(
        state,
        smbh_mass_msun=float(smbh["mass_msun"]),
        spin_a=float(smbh.get("spin_a", 0.9)),
        gamma=gamma,
        eta=float(sph_cfg.get("eta", 1.2)),
        alpha_av=float(sph_cfg.get("alpha_av", 1.0)),
        beta_av=float(sph_cfg.get("beta_av", 2.0)),
        c_cfl=float(sph_cfg.get("c_cfl", 0.2)),
        dt_min=float(sph_cfg.get("dt_min", 1.0e-6)),
        prefer_amuse=prefer_amuse,  # type: ignore[arg-type]
    )
    print(f"[HVIT ENGINE] Hydro backend: {solver.hydro_backend}")

    sink = AsyncHDF5Sink(output_dir=str(io_cfg.get("output_dir", "scratch/snapshots")))

    init_diag = solver.compute_diagnostics(step=0, time=0.0)
    init_cfl = solver.compute_cfl_timestep(dt_max=dt_max)
    print(
        f"[HVIT ENGINE] E0 = {init_diag.total_energy:.6e} J | "
        f"|P| = {init_diag.momentum_magnitude:.6e} kg·m/s"
    )
    print(
        f"[HVIT ENGINE] CFL init: dt={init_cfl.dt:.3e} s, "
        f"h_min={init_cfl.h_min:.3e} m, rho_max={init_cfl.rho_max:.3e} kg/m³"
    )

    print(f"[HVIT ENGINE] Execution started (target end_time={end_time:.3e} s, dt_max={dt_max:.3e} s).")
    t_start = time.perf_counter()

    current_time = 0.0
    step = 0

    try:
        while current_time < end_time:
            cfl = solver.compute_cfl_timestep(dt_max=dt_max)
            dt_step = min(cfl.dt, end_time - current_time)

            solver.step_hydro_and_relativity(dt_step)
            current_time += dt_step
            step += 1

            if step % write_interval == 0:
                diag = solver.compute_diagnostics(step=step, time=current_time)
                sink.schedule_snapshot(state, step, current_time, diagnostics=diag)
                rel_drift = (
                    (diag.total_energy - init_diag.total_energy) / abs(init_diag.total_energy)
                    if init_diag.total_energy != 0.0
                    else 0.0
                )
                print(
                    f"[HVIT ENGINE] step={step:5d} t={current_time:.4e}s dt={dt_step:.3e}s "
                    f"h_min={cfl.h_min:.3e}m rho_max={cfl.rho_max:.3e}kg/m³ "
                    f"E={diag.total_energy:.6e} J dE/E0={rel_drift:+.6e} "
                    f"|P|={diag.momentum_magnitude:.6e}"
                )
    finally:
        solver.shutdown()

    written = await sink.flush()
    sink.shutdown()

    t_end = time.perf_counter()
    final_diag = solver.compute_diagnostics(step=step, time=current_time)
    rel_drift = (
        (final_diag.total_energy - init_diag.total_energy) / abs(init_diag.total_energy)
        if init_diag.total_energy != 0.0
        else 0.0
    )

    print(f"[HVIT ENGINE] Finished {step} sub-steps in {t_end - t_start:.3f}s (t={current_time:.4e} s)")
    print(f"[HVIT ENGINE] Final dE/E0 = {rel_drift:+.6e}")
    print(f"[HVIT ENGINE] Wrote {len(written)} snapshot(s) to {sink.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HVIT minimal AMUSE SPH loop")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("config/default_sim.yaml"),
        help="Path to simulation YAML config",
    )
    args = parser.parse_args()
    asyncio.run(run_simulation(args.config))


if __name__ == "__main__":
    main()
