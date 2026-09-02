#!/usr/bin/env python3
"""
Snapshot diagnostic and visualization for hvit-engine HDF5 dumps.

Usage:
    python3 scripts/plot_snapshot.py -i scratch/snapshots/snapshot_000010.h5 --mode 2d_slice
    python3 scripts/plot_snapshot.py -i scratch/snapshots --mode render_all
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

# Allow standalone execution without PYTHONPATH=.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hvit.ic.polytrope import RSUN, solve_lane_emden

PlotMode = Literal["2d_slice", "density_profile", "energy_conservation", "render_all"]
ColorField = Literal["rho", "u"]

_SNAPSHOT_PATTERN = re.compile(r"snapshot_(\d+)\.h5$")


@dataclass(slots=True)
class SnapshotData:
    path: pathlib.Path
    step: int
    timestamp: float
    num_particles: int
    pos: npt.NDArray[np.float64]
    vel: npt.NDArray[np.float64]
    mass: npt.NDArray[np.float64]
    rho: npt.NDArray[np.float64]
    u: npt.NDArray[np.float64]
    p: npt.NDArray[np.float64]
    h: npt.NDArray[np.float64]
    ids: npt.NDArray[np.int64]
    kinetic_energy: float | None = None
    potential_energy: float | None = None
    internal_energy: float | None = None
    total_energy: float | None = None
    momentum_x: float | None = None
    momentum_y: float | None = None
    momentum_z: float | None = None


@dataclass(slots=True)
class EnergySeries:
    steps: npt.NDArray[np.int64]
    times: npt.NDArray[np.float64]
    total_energy: npt.NDArray[np.float64]
    energy_drift: npt.NDArray[np.float64]
    momentum: npt.NDArray[np.float64]


def apply_publication_style() -> None:
    """Configure Matplotlib for publication-ready figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
            "figure.constrained_layout.use": True,
        }
    )


def resolve_snapshots(input_path: pathlib.Path) -> list[pathlib.Path]:
    """Resolve a snapshot file or directory to a sorted list of HDF5 paths."""
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = sorted(input_path.glob("snapshot_*.h5"), key=_snapshot_sort_key)
    if not files:
        raise FileNotFoundError(f"No snapshot_*.h5 files in {input_path}")
    return files


def _snapshot_sort_key(path: pathlib.Path) -> tuple[int, str]:
    match = _SNAPSHOT_PATTERN.search(path.name)
    if match is None:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def load_snapshot(path: pathlib.Path) -> SnapshotData:
    """Load particle arrays and diagnostic attrs from one HDF5 snapshot."""
    with h5py.File(path, "r") as handle:
        attrs = handle.attrs
        return SnapshotData(
            path=path,
            step=int(attrs.get("step", 0)),
            timestamp=float(attrs.get("timestamp", 0.0)),
            num_particles=int(attrs.get("num_particles", handle["pos"].shape[0])),
            pos=np.asarray(handle["pos"], dtype=np.float64),
            vel=np.asarray(handle["vel"], dtype=np.float64),
            mass=np.asarray(handle["mass"], dtype=np.float64),
            rho=np.asarray(handle["rho"], dtype=np.float64),
            u=np.asarray(handle["u"], dtype=np.float64),
            p=np.asarray(handle["p"], dtype=np.float64),
            h=np.asarray(handle["h"], dtype=np.float64),
            ids=np.asarray(handle["ids"], dtype=np.int64),
            kinetic_energy=_optional_attr(attrs, "kinetic_energy"),
            potential_energy=_optional_attr(attrs, "potential_energy"),
            internal_energy=_optional_attr(attrs, "internal_energy"),
            total_energy=_optional_attr(attrs, "total_energy"),
            momentum_x=_optional_attr(attrs, "momentum_x"),
            momentum_y=_optional_attr(attrs, "momentum_y"),
            momentum_z=_optional_attr(attrs, "momentum_z"),
        )


def _optional_attr(attrs: h5py.AttributeManager, key: str) -> float | None:
    if key not in attrs:
        return None
    return float(attrs[key])


def primary_star_mask(snapshot: SnapshotData) -> npt.NDArray[np.bool_]:
    """
    Identify primary-star particles.

    Matches inject_binary_collision convention: lower half of particle IDs.
    """
    split_id = snapshot.num_particles // 2
    return snapshot.ids < split_id


def plot_2d_slice(
    snapshot: SnapshotData,
    output_dir: pathlib.Path,
    field: ColorField = "rho",
) -> pathlib.Path:
    """Orbital-plane scatter colored by log-density or specific internal energy."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    x = snapshot.pos[:, 0]
    y = snapshot.pos[:, 1]

    if field == "rho":
        values = np.maximum(snapshot.rho, np.finfo(np.float64).tiny)
        color_values = values
        norm = LogNorm(vmin=np.min(values), vmax=np.max(values))
        cbar_label = r"$\rho\ \mathrm{[kg\,m^{-3}]}$"
        title_field = r"$\log_{10}\rho$"
    else:
        values = np.maximum(snapshot.u, np.finfo(np.float64).tiny)
        color_values = values
        norm = LogNorm(vmin=np.min(values), vmax=np.max(values))
        cbar_label = r"$u\ \mathrm{[J\,kg^{-1}]}$"
        title_field = r"$\log_{10} u$"

    scatter = ax.scatter(
        x,
        y,
        c=color_values,
        cmap=cm.inferno,
        norm=norm,
        s=2.0,
        alpha=0.75,
        linewidths=0.0,
        rasterized=True,
    )

    ax.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=160,
        c="cyan",
        linewidths=1.8,
        label="SMBH",
        zorder=5,
    )

    ax.set_xlabel(r"$x\ \mathrm{[m]}$")
    ax.set_ylabel(r"$y\ \mathrm{[m]}$")
    ax.set_title(
        rf"Orbital slice (step {snapshot.step}, $t={snapshot.timestamp:.3e}\,\mathrm{{s}}$)"
        "\n"
        rf"colored by {title_field}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", framealpha=0.9)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"2d_slice_step{snapshot.step:06d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _analytic_polytrope_profile(
    radii: npt.NDArray[np.float64],
    rho_c: float,
    radius: float,
    n: float = 1.5,
) -> npt.NDArray[np.float64]:
    le = solve_lane_emden(n)
    alpha = radius / le.xi_surface
    xi = np.clip(radii / alpha, 0.0, le.xi_surface)
    theta_interp = interp1d(le.xi, le.theta, kind="linear", bounds_error=False, fill_value=0.0)
    theta = theta_interp(xi)
    return rho_c * np.power(np.maximum(theta, 0.0), n)


def plot_density_profile(
    snapshot: SnapshotData,
    output_dir: pathlib.Path,
    n_poly: float = 1.5,
) -> pathlib.Path:
    """Radial density profile of the primary star vs analytic n=1.5 polytrope."""
    apply_publication_style()

    mask = primary_star_mask(snapshot)
    pos = snapshot.pos[mask]
    rho = snapshot.rho[mask]

    com = np.average(pos, axis=0, weights=snapshot.mass[mask])
    radii = np.linalg.norm(pos - com, axis=1)

    rho_c_est = float(np.max(rho))
    radius_est = float(np.percentile(radii, 99.0))
    if radius_est <= 0.0:
        radius_est = float(np.max(radii))

    bins = np.linspace(0.0, radius_est, 40)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(radii, bins) - 1

    profile_r: list[float] = []
    profile_rho: list[float] = []
    for idx in range(len(bin_centers)):
        in_bin = digitized == idx
        if np.count_nonzero(in_bin) < 3:
            continue
        profile_r.append(bin_centers[idx])
        profile_rho.append(float(np.median(rho[in_bin])))

    profile_r_arr = np.asarray(profile_r, dtype=np.float64)
    profile_rho_arr = np.asarray(profile_rho, dtype=np.float64)

    r_model = np.linspace(0.0, radius_est, 300)
    rho_model = _analytic_polytrope_profile(r_model, rho_c=rho_c_est, radius=radius_est, n=n_poly)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.scatter(
        radii / RSUN,
        rho,
        s=4,
        alpha=0.15,
        color="0.55",
        label="SPH particles (primary)",
        rasterized=True,
    )
    if profile_r_arr.size > 0:
        ax.plot(
            profile_r_arr / RSUN,
            profile_rho_arr,
            "o-",
            color="tab:orange",
            linewidth=1.5,
            markersize=4,
            label="SPH median profile",
        )
    ax.plot(
        r_model / RSUN,
        rho_model,
        "-",
        color="tab:blue",
        linewidth=2.0,
        label=rf"Analytic polytrope ($n={n_poly:.1f}$)",
    )

    ax.set_xlabel(r"$r\ \mathrm{[R_\odot]}$")
    ax.set_ylabel(r"$\rho\ \mathrm{[kg\,m^{-3}]}$")
    ax.set_yscale("log")
    ax.set_title(
        rf"Primary density profile (step {snapshot.step}, $t={snapshot.timestamp:.3e}\,\mathrm{{s}}$)"
    )
    ax.legend(loc="upper right", framealpha=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"density_profile_step{snapshot.step:06d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_energy_series(snapshot_paths: list[pathlib.Path]) -> EnergySeries:
    """Extract energy and momentum time series from an ordered snapshot sequence."""
    steps: list[int] = []
    times: list[float] = []
    totals: list[float] = []
    momenta: list[float] = []

    e0: float | None = None
    drifts: list[float] = []

    for path in snapshot_paths:
        snap = load_snapshot(path)
        if snap.total_energy is None:
            raise ValueError(f"Snapshot missing total_energy attrs: {path}")

        if e0 is None:
            e0 = snap.total_energy

        rel_drift = (snap.total_energy - e0) / abs(e0) if e0 != 0.0 else 0.0
        px = snap.momentum_x or 0.0
        py = snap.momentum_y or 0.0
        pz = snap.momentum_z or 0.0
        p_mag = float(np.sqrt(px**2 + py**2 + pz**2))

        steps.append(snap.step)
        times.append(snap.timestamp)
        totals.append(snap.total_energy)
        drifts.append(rel_drift)
        momenta.append(p_mag)

    return EnergySeries(
        steps=np.asarray(steps, dtype=np.int64),
        times=np.asarray(times, dtype=np.float64),
        total_energy=np.asarray(totals, dtype=np.float64),
        energy_drift=np.asarray(drifts, dtype=np.float64),
        momentum=np.asarray(momenta, dtype=np.float64),
    )


def plot_energy_conservation(
    snapshot_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Plot fractional energy drift and momentum magnitude vs time."""
    apply_publication_style()
    series = load_energy_series(snapshot_paths)

    fig, (ax_e, ax_p) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

    ax_e.plot(
        series.times,
        series.energy_drift,
        "o-",
        color="tab:red",
        linewidth=1.5,
        markersize=4,
        label=r"$\Delta E / E_0$",
    )
    ax_e.axhline(0.0, color="0.3", linewidth=1.0, linestyle=":")
    ax_e.set_ylabel(r"$\Delta E / E_0$")
    ax_e.set_title("Energy conservation diagnostics")
    ax_e.legend(loc="best", framealpha=0.9)

    ax_p.plot(
        series.times,
        series.momentum,
        "s-",
        color="tab:green",
        linewidth=1.5,
        markersize=4,
        label=r"$|\mathbf{P}|$",
    )
    ax_p.set_xlabel(r"$t\ \mathrm{[s]}$")
    ax_p.set_ylabel(r"$|\mathbf{P}|\ \mathrm{[kg\,m\,s^{-1}]}$")
    ax_p.legend(loc="best", framealpha=0.9)

    fig.suptitle(f"Snapshot sequence ({len(snapshot_paths)} dumps)", fontsize=13)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "energy_conservation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all(
    snapshot: SnapshotData,
    snapshot_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    field: ColorField,
) -> list[pathlib.Path]:
    """Generate all diagnostic figures."""
    outputs = [
        plot_2d_slice(snapshot, output_dir, field=field),
        plot_density_profile(snapshot, output_dir),
    ]
    if len(snapshot_paths) >= 2:
        outputs.append(plot_energy_conservation(snapshot_paths, output_dir))
    else:
        print("[plot_snapshot] Skipping energy_conservation (need >= 2 snapshots in sequence).")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and plot hvit-engine HDF5 snapshot diagnostics.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="Path to a snapshot .h5 file or directory of snapshots",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="2d_slice",
        choices=["2d_slice", "density_profile", "energy_conservation", "render_all"],
        help="Visualization mode",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("scratch/plots"),
        help="Output directory for PNG figures",
    )
    parser.add_argument(
        "--field",
        choices=["rho", "u"],
        default="rho",
        help="Scalar field for 2d_slice coloring",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    snapshot_paths = resolve_snapshots(args.input)
    primary_path = snapshot_paths[-1] if len(snapshot_paths) == 1 else snapshot_paths[-1]
    snapshot = load_snapshot(primary_path)
    output_dir = args.output

    outputs: list[pathlib.Path] = []
    mode: PlotMode = args.mode

    if mode == "2d_slice":
        outputs.append(plot_2d_slice(snapshot, output_dir, field=args.field))
    elif mode == "density_profile":
        outputs.append(plot_density_profile(snapshot, output_dir))
    elif mode == "energy_conservation":
        outputs.append(plot_energy_conservation(snapshot_paths, output_dir))
    elif mode == "render_all":
        outputs.extend(render_all(snapshot, snapshot_paths, output_dir, field=args.field))

    for path in outputs:
        print(f"[plot_snapshot] Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
