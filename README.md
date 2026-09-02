# hvit-engine

Open-source particle-mesh simulation framework developed by the Smoky Mountain Institute for Relativistic Physics (SMIRP). Models hydrodynamics of relativistic stellar collisions near black holes to isolate kinetic energy signatures of hyper-velocity impact transients (HVIT).

Built with Python, AMUSE, and CUDA.

## Layout

```
hvit-engine/
├── config/default_sim.yaml   # Run parameters (YAML)
├── hvit/
│   ├── core/                 # ParticleStateMatrix, RelativisticPotential
│   ├── ic/                   # Polytropic profiles, binary orbit injection
│   ├── solvers/              # HVITSolver (leapfrog + PN gravity)
│   └── io/                   # AsyncHDF5Sink → scratch/snapshots/
├── scratch/                  # gitignored runtime output
├── main.py                   # Pipeline entry point
└── requirements.txt
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # AMUSE requires local build toolchain
PYTHONPATH=. python3 main.py
PYTHONPATH=. python3 main.py --config config/default_sim.yaml
```

Default config uses `10_000` particles for dev iteration. Bump `simulation.num_particles` in `config/default_sim.yaml` for production runs (e.g. `1_000_000`).

## Current Loop (v0.1)

- **ICs:** Lane–Emden polytropes (`n=1.5`) via 3D rejection sampling; hyperbolic plunge orbits with `r_p` in Schwarzschild radii
- **Gravity:** Paczynski–Wiita pseudo-Newtonian + Lense–Thirring frame dragging
- **Hydro:** Ideal-gas EOS closure (`P = (γ−1)ρu`); full AMUSE SPH coupling next
- **Monitors:** Total energy (KE + PW potential + internal) and linear momentum per snapshot
- **I/O:** Async LZF-compressed HDF5 dumps to `scratch/snapshots/`

## AMUSE Note

`amuse-framework` must be compiled against your local MPI/CUDA stack. The minimal loop runs without AMUSE at runtime; `HVITSolver.to_amuse_particles()` exports state for downstream Gadget2/Fi coupling.
