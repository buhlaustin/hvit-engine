"""HVIT engine pipeline entry point."""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import time
from typing import Any

import numpy as np
import yaml

from hvit.core.state import ParticleStateMatrix
from hvit.io.hdf5_sink import AsyncHDF5Sink
from hvit.solvers.sph_engine import HVITSolver


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def initialize_state(cfg: dict[str, Any]) -> ParticleStateMatrix:
    sim = cfg["simulation"]
    ic = cfg["initial_conditions"]
    n = int(sim["num_particles"])

    state = ParticleStateMatrix.allocate(n)
    half = n // 2

    state.pos[:half, 0] = float(ic["star1_pos_x"])
    state.pos[half:, 0] = float(ic["star2_pos_x"])
    state.vel[half:, 0] = float(ic["star2_vel_x"])

    state.mass[:] = float(ic["particle_mass"])
    state.rho[:] = float(ic["rho"])
    state.u[:] = float(ic["u"])
    state.h[:] = np.cbrt(state.mass / state.rho) * 2.0

    gamma = float(cfg.get("eos", {}).get("gamma", 5 / 3))
    state.compute_eos_ideal_gas(gamma=gamma)
    return state


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
    state = initialize_state(cfg)

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
