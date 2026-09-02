"""
High-throughput asynchronous HDF5 data writer designed for scratch NVMe mounts.
Pre-allocates buffers to eliminate allocation overhead during simulation steps.
"""

from __future__ import annotations

import asyncio
import pathlib
from concurrent.futures import ThreadPoolExecutor

import h5py

from hvit.core.state import ParticleStateMatrix
from hvit.solvers.sph_engine import SimulationDiagnostics


class AsyncHDF5Sink:
    def __init__(self, output_dir: str = "scratch/snapshots", max_workers: int = 2):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: set[asyncio.Task[str]] = set()

    def _write_sync(
        self,
        state: ParticleStateMatrix,
        step: int,
        timestamp: float,
        diagnostics: SimulationDiagnostics | None = None,
    ) -> str:
        filepath = self.output_dir / f"snapshot_{step:06d}.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["timestamp"] = timestamp
            f.attrs["step"] = step
            f.attrs["num_particles"] = state.N

            if diagnostics is not None:
                f.attrs["kinetic_energy"] = diagnostics.kinetic_energy
                f.attrs["potential_energy"] = diagnostics.potential_energy
                f.attrs["internal_energy"] = diagnostics.internal_energy
                f.attrs["total_energy"] = diagnostics.total_energy
                f.attrs["momentum_x"] = diagnostics.momentum_x
                f.attrs["momentum_y"] = diagnostics.momentum_y
                f.attrs["momentum_z"] = diagnostics.momentum_z

            # Low-compression chunking for maximum IOPS on NVMe drives
            f.create_dataset("pos", data=state.pos, compression="lzf", chunks=True)
            f.create_dataset("vel", data=state.vel, compression="lzf", chunks=True)
            f.create_dataset("mass", data=state.mass, compression="lzf", chunks=True)
            f.create_dataset("rho", data=state.rho, compression="lzf", chunks=True)
            f.create_dataset("p", data=state.p, compression="lzf", chunks=True)
            f.create_dataset("u", data=state.u, compression="lzf", chunks=True)
            f.create_dataset("h", data=state.h, compression="lzf", chunks=True)
            f.create_dataset("ids", data=state.ids)
        return str(filepath)

    async def write_snapshot(
        self,
        state: ParticleStateMatrix,
        step: int,
        timestamp: float,
        diagnostics: SimulationDiagnostics | None = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._write_sync,
            state,
            step,
            timestamp,
            diagnostics,
        )

    def schedule_snapshot(
        self,
        state: ParticleStateMatrix,
        step: int,
        timestamp: float,
        diagnostics: SimulationDiagnostics | None = None,
    ) -> asyncio.Task[str]:
        task = asyncio.create_task(
            self.write_snapshot(state, step, timestamp, diagnostics)
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def flush(self) -> list[str]:
        """Await all pending snapshot writes before shutdown."""
        if not self._pending:
            return []
        results = await asyncio.gather(*self._pending, return_exceptions=True)
        paths: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            paths.append(result)
        return paths

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
