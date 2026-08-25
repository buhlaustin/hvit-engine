"""HVIT engine pipeline entry point."""

from __future__ import annotations

import argparse
import asyncio
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
    cfg = load_config(config_path)
    sim = cfg["simulation"]
    smbh = cfg["smbh"]
    io_cfg = cfg.get("io", {})

    n = int(sim["num_particles"])
    dt = float(sim["dt"])
    total_steps = int(sim["total_steps"])
    write_interval = int(sim["write_interval"])
    gamma = float(cfg.get("eos", {}).get("gamma", 5 / 3))

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
    )
    sink = AsyncHDF5Sink(output_dir=str(io_cfg.get("output_dir", "scratch/snapshots")))

    init_diag = solver.compute_diagnostics(step=0, time=0.0)
    print(
        f"[HVIT ENGINE] E0 = {init_diag.total_energy:.6e} J | "
        f"|P| = {init_diag.momentum_magnitude:.6e} kg·m/s"
    )

    print("[HVIT ENGINE] Execution started.")
    t_start = time.perf_counter()

    for step in range(total_steps):
        solver.step_hydro_and_relativity(dt)
        current_time = (step + 1) * dt

        if step % write_interval == 0:
            diag = solver.compute_diagnostics(step=step, time=current_time)
            sink.schedule_snapshot(state, step, current_time, diagnostics=diag)
            rel_drift = (
                (diag.total_energy - init_diag.total_energy) / abs(init_diag.total_energy)
                if init_diag.total_energy != 0.0
                else 0.0
            )
            print(
                f"[HVIT ENGINE] step={step:5d} t={current_time:.4e}s "
                f"E={diag.total_energy:.6e} J dE/E0={rel_drift:+.6e} "
                f"|P|={diag.momentum_magnitude:.6e}"
            )

    written = await sink.flush()
    sink.shutdown()

    t_end = time.perf_counter()
    final_diag = solver.compute_diagnostics(step=total_steps - 1, time=total_steps * dt)
    rel_drift = (
        (final_diag.total_energy - init_diag.total_energy) / abs(init_diag.total_energy)
        if init_diag.total_energy != 0.0
        else 0.0
    )

    print(f"[HVIT ENGINE] Finished {total_steps} steps in {t_end - t_start:.3f}s")
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
