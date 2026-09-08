# Core Thesis: Hyper-Velocity Impact Transients

**Author:** Austin Buhl  
**Project:** hvit-engine

---

## 1. Statement of Thesis

We reject the **single-body Tidal Disruption Event (TDE) default model** as a universal explanation for all macro-scale flares observed near supermassive black holes (SMBHs).

Macro-scale electromagnetic flares—particularly those associated with high-eccentricity stellar encounters, stellar collisions, and deeply plunging orbits—cannot be reduced, in general, to the disruption of a rigid or passively responding point mass. Instead, we treat stars as **highly compressible, self-gravitating fluid plasmas** whose macroscopic observables are set by **hydrodynamic stretching, compression, and high-energy kinetic shocks** at the moment of intersection with the tidal field and with other stellar material.

The **Hyper-Velocity Impact Transient (HVIT)** framework posits that a distinct class of flare signatures arises when bulk kinetic energy is converted into thermal and internal energy through **resolved shock dissipation** in a compressible fluid, rather than through the quasi-adiabatic mass stripping assumed in canonical TDE light-curve models.

---

## 2. The Default TDE Model and Its Limits

### 2.1 Canonical single-body TDE picture

The standard TDE narrative assumes:

1. A single star on a mildly eccentric or parabolic orbit approaches an SMBH.
2. Tidal forces exceed self-gravity at the pericenter; the star is **disrupted** into a stream of debris.
3. Roughly half the debris remains bound; accretion onto the SMBH powers a luminous flare with a characteristic rise and decline set by fallback timescales.

In this picture, the star is often treated as a **rigid or weakly deformable body** until the formal disruption radius $r_t \sim R_\star (M_\bullet / M_\star)^{1/3}$. Hydrodynamic detail—shock structure, compressibility, collisional intersections—is subsumed into a mass-return rate $\dot M(t)$.

### 2.2 Where the default model breaks down

The single-body TDE default is appropriate when:

- One star disrupts at moderate penetration ($r_p \sim r_t$).
- The encounter is slow compared to the sound-crossing time of the stellar envelope.
- Binary or collisional geometry is absent.

It is **not** appropriate when:

- **Binary or multi-body encounters** bring two stellar fluids into direct hydrodynamic contact before or during tidal stripping.
- **Hyperbolic, high-eccentricity plunges** ($e \gg 1$, $r_p \ll r_t$) drive supersonic compression and shock heating across the stellar interior.
- **Macro-scale flare energetics** require accounting for **bulk kinetic energy deposition** ($\tfrac{1}{2} \rho v^2$) that is dissipated through artificial and physical viscosity, not merely gravitational unbinding.

In these regimes, the flare is not a passive fallback event; it is an **impact transient**—a HVIT.

---

## 3. The HVIT Alternative: Stars as Compressible Fluids

### 3.1 Physical object

A star near an SMBH is modeled as a **self-gravitating, barotropic or ideal-gas fluid** in polytropic hydrostatic equilibrium at initialization. Each fluid element carries:

- mass density $\rho$,
- pressure $P$,
- specific internal energy $u$,
- velocity $\mathbf{v}$ in the SMBH rest frame.

The plasma is **highly compressible**: tidal compression, ram pressure from orbital motion, and collisional overlap can raise $\rho$ and $u$ by orders of magnitude over dynamical timescales. This is fundamentally different from a rigid-body tide model, in which strain energy is stored elastically and dissipation is parametrized rather than resolved.

### 3.2 Hydrodynamic mechanisms

Three coupled processes define HVIT phenomenology:

| Mechanism | Description |
|-----------|-------------|
| **Tidal stretching** | The SMBH potential imposes a position-dependent tidal tensor that elongates the stellar fluid along the radial direction and compresses it azimuthally. For $r_p \ll r_t$, stretching is accompanied by **transverse compression** and rising internal pressure. |
| **Bulk compression** | Hyperbolic inbound trajectories ($v_\infty \sim 0.05\,c$) supply large specific orbital kinetic energy. As the fluid converges near pericenter, **PdV work** $-P\, \nabla\cdot\mathbf{v}$ raises $u$ even before formal disruption. |
| **Kinetic shocks** | When two stellar fluids intersect—or when a single star undergoes self-shocking in converging flow—**Rankine–Hugoniot-like discontinuities** form on scales resolved by SPH. Dissipation is captured by Monaghan artificial viscosity $\Pi_{ij}$, converting kinetic energy into thermal energy. |

The observable consequence is a flare whose **early-time luminosity and spectral state** may be dominated by shock-heated material at $T \sim 10^4$–$10^6\,\mathrm{K}$, not by the fallback of loosely bound debris streams predicted by TDE templates.

### 3.3 Self-gravity and compressibility

Self-gravity of the stellar fluid is retained at initialization through the Lane–Emden polytropic profile ($n = 1.5$), which encodes the balance between pressure support and gravitational binding. During the encounter, SMBH tides dominate over stellar self-gravity in the plunge phase; the fluid response is **hydrodynamic**, not ballistic. Compressibility—encoded in the ideal-gas closure $P = (\gamma - 1)\rho u$ with $\gamma = 5/3$—permits density contrasts and shock jumps that a rigid-body treatment cannot represent.

---

## 4. Distinguishing HVIT from Classical TDE

| Aspect | Classical single-body TDE | HVIT (this work) |
|--------|---------------------------|------------------|
| Stellar model | Disruption radius; debris stream | Compressible SPH fluid plasma |
| Geometry | One star, parabolic orbit | Binary/collisional hyperbolic plunge |
| Energy channel | Gravitational unbinding → fallback accretion | Kinetic impact → shock dissipation → thermal flare |
| Timescale driver | Fallback time $t_{\mathrm{fb}} \sim (r_p^3 / GM_\bullet)^{1/2}$ | Sound-crossing and shock-crossing times at pericenter |
| Observational signature | Smooth rise/decay template | Impulsive heating; impact-dependent kinetic signatures |

HVIT does not deny that classical TDEs occur. It asserts that **assigning every macro-scale SMBH flare to the single-body TDE template is physically incomplete** when compressible hydrodynamics and collisional geometry are active.

---

## 5. Numerical Realization in hvit-engine

The hvit-engine code is the computational instrument for this thesis. Its design choices follow directly from the physical claims above:

1. **SPH hydrodynamics** resolves stretching, compression, and shocks as continuous fluid motion with explicit $\rho$, $u$, and $P$ fields.
2. **Paczyński–Wiita + Lense–Thirring gravity** places the fluid in a relativistic SMBH potential appropriate for deeply plunging orbits ($r_p \sim 10\,r_s$).
3. **Hyperbolic binary injection** initializes two polytropic stars on collisional trajectories, not a single parabolic disruptor.
4. **Energy diagnostics** track kinetic, internal, and gravitational energy separately, enabling direct measurement of shock-driven heating versus orbital energy exchange.

Mathematical details of the governing equations, closures, and integrators are given in [THEORY.md](THEORY.md).

---

## 6. Scientific Program

The HVIT research program asks:

1. Under what orbital parameters ($r_p$, $e$, $v_\infty$, binary separation) does shock heating dominate over tidal stripping?
2. What **kinetic energy signatures**—in internal-energy growth rate, momentum flux, and eventual light-curve morphology—distinguish HVIT from TDE?
3. Can time-domain surveys (optical, UV, X-ray) be re-analyzed with a fluid-impact prior rather than a single-body TDE prior?

hvit-engine provides the forward-modeling infrastructure to answer these questions through controlled numerical experiments.

---

## 7. Summary

> **Core thesis:** Macro-scale flares near SMBHs are not universally described by single-body TDE physics. Stars are compressible, self-gravitating fluid plasmas. High-eccentricity encounters produce **Hyper-Velocity Impact Transients** through hydrodynamic stretching, compression, and kinetic shock dissipation—phenomena that require resolved fluid dynamics, not rigid-body disruption templates.

---

*See also: [THEORY.md](THEORY.md) for the full mathematical formulation.*
