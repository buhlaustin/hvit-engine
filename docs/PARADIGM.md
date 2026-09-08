# Paradigm Comparison: Standard TDE vs. HVIT

**Author:** Austin Buhl  
**Project:** hvit-engine

This document states the four core tenets of the canonical **Tidal Disruption Event (TDE)** paradigm (Rees 1988 and successors), contrasts each with the **Hyper-Velocity Impact Transient (HVIT)** thesis, and catalogs the principal **failure cases** of the standard model that motivate a fluid-impact alternative.

*See also: [THESIS.md](THESIS.md) (core thesis statement) · [THEORY.md](THEORY.md) (governing equations)*

---

## 1. The Standard TDE Paradigm (Rees 1988)

The default model for macro-scale flares near supermassive black holes (SMBHs) rests on four interlocking assumptions:

1. A **single isolated star** disrupts on a parabolic orbit.
2. Post-disruption debris is **ballistic**; gas pressure is negligible.
3. Mass fallback follows a universal **$t^{-5/3}$ power law**, setting the flare decay.
4. Peak luminosity arises from **late-time stream circularization** and viscous accretion-disk formation.

This framework has been enormously successful for well-selected events. HVIT does not dispute that classical TDEs occur. It rejects their use as the **universal prior** for every nuclear flare.

---

## 2. Core Tenets: Standard TDE vs. HVIT

| # | Standard TDE Paradigm (Rees 1988) | HVIT Core Thesis |
|---|-----------------------------------|------------------|
| **1** | **Single isolated body.** A lone star on a parabolic orbit ($e = 1$) is torn apart when tidal forces exceed self-gravity at pericenter ($R_p \le R_t$). Disruption radius $R_t \approx R_\star (M_\bullet / M_\star)^{1/3}$. | **Hyper-velocity collisions.** Rejects the single-body default. Macro flares are driven by high-velocity intersections ($v \sim 0.01\,c$–$0.1\,c$) between two stars in dense nuclear clusters, where collisional geometry—not isolated tidal stripping—sets the energy budget. |
| **2** | **Frozen-in approximation.** Stellar gas is treated as non-interacting, ballistic test particles after tidal disruption. Pressure forces are assumed negligible; the debris stream evolves under gravity alone. | **Compressible plasma shocks.** Stars are highly compressible, self-gravitating fluid plasmas. Ram pressure, tidal compression, and **Monaghan artificial viscosity** ($\Pi_{ij}$) dominate energy transfer. Pressure gradients and shock dissipation are resolved, not discarded. |
| **3** | **The $t^{-5/3}$ fallback power law.** Debris falls back toward the SMBH at $\dot M \propto t^{-5/3}$, dictating the light-curve decay tail. This scaling is treated as a robust, near-universal signature of TDEs. | **Dual-phase non-power-law evolution.** Shocks generate a **rapid prompt kinetic–thermal flare** at impact, followed by a **secondary, multi-peak fallback accretion** curve that generically **violates** $t^{-5/3}$. Early-time luminosity is shock-heated, not fallback-driven. |
| **4** | **Circularization and accretion disk.** Luminosity is generated primarily through late-time viscous dissipation as returning debris streams self-intersect, circularize, and form an accretion disk around the SMBH. | **Direct shock thermalization.** Massive kinetic energy $E_k = \tfrac{1}{2} m v^2$ is converted directly into thermal radiation (specific internal energy $u$) **at the moment of impact**, long before disk circularization can occur. |

### 2.1 Tenet 1 — Geometry: one star vs. colliding fluids

The Rees picture begins with one star, one SMBH, and one pericenter passage. Eccentricity is fixed at $e = 1$ (parabolic) or slightly super-parabolic; binary companions, cluster multiplicity, and direct stellar collisions are out of scope.

HVIT places **binary and collisional geometry** at center stage. In nuclear star clusters, stellar number density is high enough that hyperbolic encounters between *pairs* of stars plunging jointly toward the SMBH are not rare exotica—they are an expected channel for macro-scale energy release. The engine injects two Lane–Emden polytropes on a shared hyperbolic orbit with configurable separation, eccentricity $e > 1$, and $v_\infty \sim 0.05\,c$.

### 2.2 Tenet 2 — Dynamics: ballistic debris vs. compressible shocks

After disruption, standard TDE theory tracks the ballistic trajectories of gas parcels. Internal energy is frozen in; the "frozen-in approximation" treats the stream as a collisionless ensemble until late-time stream crossing.

HVIT solves the **compressible Euler equations** via SPH. The momentum equation includes symmetrized pressure gradients and the Monaghan viscous pressure

$$
\Pi_{ij} =
\begin{cases}
\dfrac{-\alpha_{\mathrm{AV}}\, c_{s,ij}\, \mu_{ij} + \beta_{\mathrm{AV}}\, \mu_{ij}^2}{\bar{\rho}_{ij}},
& \mathbf{v}_{ij}\cdot\mathbf{r}_{ij} < 0, \\[8pt]
0, & \text{otherwise},
\end{cases}
$$

which captures shock heating when fluid elements converge. Ram pressure $\rho v^2$ at impact velocities $v \sim 0.01$–$0.1\,c$ can exceed thermal pressure in the stellar envelope by orders of magnitude.

### 2.3 Tenet 3 — Light curves: universal $t^{-5/3}$ vs. dual-phase structure

The $t^{-5/3}$ fallback law arises from the assumption that debris energy is spread over a range of binding energies with a specific phase-space density, producing a mass-return rate $\dot M(t) \propto t^{-5/3}$ at late times (Lodato et al. 2009; Coughlin & Begelman 2014).

HVIT predicts a **two-component** light curve:

1. **Prompt phase:** Impulsive rise from shock thermalization as $E_k \to u$ on dynamical timescales ($t \sim R_\star / c_s$).
2. **Secondary phase:** Multi-peak fallback accretion from partially bound debris, with structure set by collision geometry rather than a single-stream $t^{-5/3}$ template.

Survey pipelines that fit only $t^{-5/3}$ decay will systematically mis-characterize HVIT events.

### 2.4 Tenet 4 — Emission mechanism: disk viscous heating vs. impact thermalization

In the standard picture, the peak bolometric luminosity is delayed until debris streams return, cross, and circularize—viscous dissipation in a geometrically thin disk powers the flare.

HVIT identifies a **pre-circularization** channel: kinetic energy deposited at stellar intersection is thermalized immediately via shocks. Observable radiation can precede disk formation by many dynamical timescales. The engine tracks this channel directly through the internal-energy field $u_i$ and the diagnostic sum $E_{\mathrm{int}} = \sum_i m_i u_i$.

---

## 3. Failure Cases of the Standard Model

The standard TDE paradigm encounters systematic difficulties in regimes where HVIT provides a natural alternative.

### 3.1 The Hills mass limit

For an SMBH of mass $M_\bullet$, the Schwarzschild radius and tidal radius scale as

$$
R_s = \frac{2 G M_\bullet}{c^2}, \qquad
R_t \approx R_\star \left(\frac{M_\bullet}{M_\star}\right)^{1/3}.
$$

The Hills mass limit $M_{\mathrm{Hills}} \sim 10^8\,M_\odot$ marks the crossover where $R_s > R_t$: the event horizon swallows solar-type stars whole without producing a classical tidal disruption flare. Standard TDE theory therefore **predicts a suppression** of nuclear flares in the most massive galaxies.

Hyper-velocity stellar **collisions** in the nuclear cluster do not require $R_p \le R_t$ for a single star. Two stars can intersect and dissipate kinetic energy at radii $r \gg R_s$, producing a macro-scale thermal flare **outside the horizon**, independent of the Hills limit. HVIT events are not gated by the $R_s > R_t$ condition that silences canonical TDEs.

### 3.2 Light-curve anomalies in time-domain surveys

Time-domain surveys—**ZTF**, **Pan-STARRS**, **LSST**—have cataloged dozens of nuclear flares whose light curves deviate from the canonical TDE template:

| Anomaly | Standard TDE expectation | HVIT interpretation |
|---------|--------------------------|----------------------|
| Rapid early rise ($t_{\mathrm{rise}} \ll t_{\mathrm{fb}}$) | Rise set by fallback diffusion timescale | Prompt shock heating at impact |
| Secondary re-brightening peaks | Single smooth peak + $t^{-5/3}$ tail | Multi-stream collision geometry; repeated fallback episodes |
| Thermal spectra inconsistent with $t^{-5/3}$ fit | Forced fit by adjusting $M_\bullet$, $R_\star$ | Dual-phase structure; impact $u$ dominates early spectrum |

Fitting software routinely forces these events into the standard template by twisting free parameters (black hole mass, stellar radius, disruption factor $\beta$). HVIT provides a **structurally different prior**: a prompt shock-powered component plus a non–$t^{-5/3}$ secondary fallback, without requiring unphysical parameter excursions.

### 3.3 Dense nuclear-cluster environments

The single-body TDE rate scales with stellar disruption probability per galaxy. In dense cores ($\rho_\star \sim 10^3$–$10^6\,M_\odot\,\mathrm{pc}^{-3}$), **direct stellar collisions** and **binary-SMBH three-body encounters** become competitive with isolated disruptions. The standard model has no native channel for these events; HVIT is built for them.

---

## 4. What hvit-engine Tests

The simulation framework is designed to falsify or support the HVIT thesis against the standard paradigm on each tenet:

| Tenet | Engine capability |
|-------|-------------------|
| Collision geometry | Binary polytrope injection on hyperbolic orbits ($e > 1$, $v_\infty \sim 0.05\,c$) |
| Compressible shocks | SPH with ideal-gas EOS ($\gamma = 5/3$) and Monaghan $\Pi_{ij}$ |
| Non–$t^{-5/3}$ light curves | Time-resolved $E_{\mathrm{kin}}$, $E_{\mathrm{int}}$, $E_{\mathrm{grav}}$ diagnostics |
| Impact thermalization | Specific internal energy $u_i$ evolved via PdV work and viscous dissipation |

Forward models produced by hvit-engine can be compared directly against survey light curves, testing whether shock-dominated, dual-phase signatures provide better fits than single-body $t^{-5/3}$ templates.

---

## 5. Summary

The standard TDE paradigm (Rees 1988) assumes a single parabolic disruptor, ballistic debris, universal $t^{-5/3}$ fallback, and disk-powered luminosity. HVIT rejects each tenet as a **default** explanation for macro-scale nuclear flares:

1. **Collisions, not isolated disruption** — two-body hyper-velocity intersections in dense clusters.
2. **Compressible shocks, not frozen-in debris** — ram pressure and $\Pi_{ij}$ dominate energy transfer.
3. **Dual-phase non-power-law light curves** — prompt shock flare plus multi-peak fallback.
4. **Direct thermalization, not late-time disk circularization** — $E_k \to u$ at impact.

The standard model fails at the Hills mass limit, for anomalous survey light curves, and in dense cluster environments. hvit-engine exists to compute the fluid dynamics those failure cases demand.

---

*See also: [THESIS.md](THESIS.md) · [THEORY.md](THEORY.md)*
