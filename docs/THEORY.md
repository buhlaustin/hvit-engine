# Theoretical Foundations of the HVIT Engine

This document presents a self-contained mathematical formulation of the physics and numerics implemented in **hvit-engine**: a smoothed-particle hydrodynamics (SPH) code for simulating **Hyper-Velocity Impact Transients (HVIT)** arising from high-eccentricity stellar collisions and tidal disruptions in the gravitational field of a central supermassive black hole (SMBH). All quantities are expressed in SI units unless otherwise noted.

---

## 1. Theoretical Foundations and Problem Definition

### 1.1 Physical Domain

We consider a collisionless-plus-fluid system comprising:

- A **central point mass** $M_\bullet$ representing an SMBH with dimensionless Kerr spin parameter $a \in [0, 1)$.
- Two **gaseous stellar components** modeled as compressible fluids, each initially in hydrostatic polytropic equilibrium, approaching on a hyperbolic orbit with periapsis distance $r_p$ of order tens of Schwarzschild radii and asymptotic speed $v_\infty \ll c$.

The scientific objective is to resolve the coupled **orbital dynamics**, **tidal deformation**, and **shock-driven dissipative heating** during the plunge phase, when kinetic energy deposition and internal-energy growth may produce observably distinct impact transients.

Each SPH particle $i$ carries mass $m_i$, position $\mathbf{r}_i$, velocity $\mathbf{v}_i$, specific internal energy $u_i$, mass density $\rho_i$, pressure $P_i$, and smoothing length $h_i$.

### 1.2 Background Spacetime Regime

Full general-relativistic hydrodynamics in dynamical spacetimes is not required for the parameter space of interest ($r_p \gtrsim 10\,r_s$, $v_\infty \sim 0.05\,c$). The engine adopts a **test-fluid approximation** on a fixed background:

1. **Monopole sector:** Paczyński–Wiita (PW) pseudo-Newtonian potential, which reproduces the Schwarzschild marginally bound circular orbit and captures the qualitative strengthening of gravity near $r \sim 2 r_g$.
2. **Spin sector:** Leading-order **Lense–Thirring** frame-dragging acceleration from the SMBH angular momentum $\mathbf{J}$, aligned with the $+z$ axis.

The gravitational radius is

$$r_g \equiv \frac{G M_\bullet}{c^2}, \qquad r_s = 2 r_g = \frac{2 G M_\bullet}{c^2}.$$

The dimensionless spin vector magnitude is encoded via

$$\mathbf{J} = a\, M_\bullet\, r_g\, c\,\hat{\mathbf{z}}.$$

This regime is consistent with a **1.5PN** treatment of two-body dynamics: relativistic corrections enter at $\mathcal{O}(v^2/c^2)$ and $\mathcal{O}(G M_\bullet/(r c^2))$, while hydrodynamic self-gravity of the stellar fluid is neglected relative to the SMBH tide (monopole plus spin).

### 1.3 Post-Newtonian Equations of Motion

#### 1.3.1 Harmonic-gauge 1.5PN acceleration

In the standard post-Newtonian expansion (harmonic coordinates, point mass $M_\bullet$ at the origin), the acceleration of a test particle with velocity $\mathbf{v}$ at position $\mathbf{r}$ ($r = |\mathbf{r}|$, $\hat{\mathbf{r}} = \mathbf{r}/r$) through 1.5PN order is

$$
\begin{aligned}
\mathbf{a}^{\mathrm{PN}} &= -\frac{G M_\bullet}{r^2}\Bigg[
\Big(1 - \frac{4 G M_\bullet}{r c^2} + \frac{v^2}{c^2} - \frac{4(\mathbf{r}\cdot\mathbf{v})^2}{r^2 c^2}\Big)\hat{\mathbf{r}}
+ \frac{4}{c^2}\frac{G M_\bullet}{r^3}(\mathbf{r}\cdot\mathbf{v})\mathbf{v}
\Bigg] \\
&\quad + \frac{G M_\bullet}{r^3}\mathbf{r}\times\frac{\mathbf{v}}{c^2}\times\mathbf{v}
\;+\; \mathcal{O}\!\left(c^{-4}\right).
\end{aligned}
$$

The final term is the **1.5PN precession** contribution (sometimes grouped with spin-orbit effects in Kerr extensions). For a Kerr background, the gravitomagnetic sector is dominated by the **Lense–Thirring** coupling.

#### 1.3.2 Lense–Thirring frame dragging

The gravitomagnetic acceleration due to spin angular momentum $\mathbf{J}$ is

$$
\mathbf{a}^{\mathrm{LT}}
= \frac{2 G}{c^2 r^3}
\left[
\frac{3}{r^2}\big(\mathbf{J}\cdot(\mathbf{r}\times\mathbf{v})\big)\mathbf{r}
+ \mathbf{v}\times\mathbf{J}
\right].
$$

This term induces nodal precession of the orbital plane and couples radial and azimuthal motion near the SMBH.

#### 1.3.3 Paczyński–Wiita closure (implemented monopole)

The engine replaces the weak-field $1/r$ Newtonian potential with the PW form

$$
\Phi_{\mathrm{PW}}(r) = -\frac{G M_\bullet}{r - 2 r_g},
$$

with regularization $r_{\mathrm{eff}} = \max(r - 2 r_g,\, \varepsilon r_g)$ to avoid singular evaluation at the photon sphere. The corresponding radial acceleration is

$$
\mathbf{a}^{\mathrm{PW}} = -\frac{G M_\bullet}{r_{\mathrm{eff}}^{\,2}\, r}\,\mathbf{r}.
$$

The total external gravitational acceleration applied to each particle is

$$
\mathbf{a}^{\mathrm{grav}} = \mathbf{a}^{\mathrm{PW}} + \mathbf{a}^{\mathrm{LT}}.
$$

This hybrid scheme preserves the strong-field circular-orbit behavior of Schwarzschild while retaining the explicit velocity-dependent Lense–Thirring coupling at 1.5PN order in the spin sector.

### 1.4 Total Particle Acceleration

The complete Lagrangian acceleration of particle $i$ is

$$
\frac{d\mathbf{v}_i}{dt} = \mathbf{a}_i^{\mathrm{hydro}} + \mathbf{a}_i^{\mathrm{grav}},
$$

where $\mathbf{a}_i^{\mathrm{hydro}}$ is the SPH pressure-gradient and artificial-viscosity force (Section 2) and $\mathbf{a}_i^{\mathrm{grav}} = \mathbf{a}^{\mathrm{grav}}(\mathbf{r}_i, \mathbf{v}_i)$.

---

## 2. Hydrodynamic Formulation (Smoothed Particle Hydrodynamics)

### 2.1 Continuum Interpolants

A continuum field $A(\mathbf{r})$ is approximated by the SPH summation over particles $j$:

$$
A(\mathbf{r}) \approx \sum_j m_j\, \frac{A_j}{\rho_j}\, W(\mathbf{r} - \mathbf{r}_j,\, h),
$$

where $W$ is a compactly supported kernel of smoothing length $h$. The particle estimate at the location of particle $i$ is

$$
A_i = \sum_j m_j\, \frac{A_j}{\rho_j}\, W_{ij}, \qquad W_{ij} \equiv W(|\mathbf{r}_{ij}|,\, h_{ij}),
$$

with $\mathbf{r}_{ij} \equiv \mathbf{r}_i - \mathbf{r}_j$ and pairwise smoothing length

$$
h_{ij} \equiv \tfrac{1}{2}(h_i + h_j).
$$

Neighbor searches are restricted to pairs with $|\mathbf{r}_{ij}| \le 2 h_{ij}$.

### 2.2 Kernel Functions

Define the dimensionless radius $q \equiv r/h$. Normalization in three dimensions uses $\sigma_h = 1/(\pi h^3)$.

#### 2.2.1 $M_4$ cubic spline (Monaghan 1992)

The $C^2$ cubic spline kernel implemented in the native hydro solver is

$$
W_{\mathrm{M4}}(r,h) = \sigma_h
\begin{cases}
1 - \dfrac{3}{2}q^2 + \dfrac{3}{4}q^3, & 0 \le q \le 1, \\[6pt]
\dfrac{1}{4}(2 - q)^3, & 1 < q \le 2, \\[6pt]
0, & q > 2.
\end{cases}
$$

Its radial derivative yields the gradient with respect to the first index:

$$
\nabla_i W_{ij} = \frac{dW}{dr}\bigg|_{r = r_{ij}}\, \frac{\mathbf{r}_{ij}}{r_{ij}}.
$$

Explicitly, with $q = r_{ij}/h_{ij}$,

$$
\frac{dW}{dq} = \sigma_h
\begin{cases}
-3q + \dfrac{9}{4}q^2, & 0 < q \le 1, \\[6pt]
-\dfrac{3}{4}(2-q)^2, & 1 < q \le 2,
\end{cases}
\qquad
\frac{dW}{dr} = \frac{1}{h_{ij}}\frac{dW}{dq}.
$$

#### 2.2.2 Wendland $C^2$ kernel (reference)

For completeness, the Wendland $C^2$ kernel (frequently employed in modern SPH shock-capture studies) is

$$
W_{\mathrm{Wend}}(r,h) = \frac{21}{16\pi h^3}
(1 - q/2)^4 (2q + 1),
\qquad 0 \le q \le 2,
$$

and vanishes for $q > 2$. Although not active in the current native backend, it defines an alternative resolution-stability closure within the same SPH formalism.

### 2.3 Adaptive Smoothing Length

Particle smoothing lengths evolve with the local mean interparticle spacing via

$$
h_i = \eta \left(\frac{m_i}{\rho_i}\right)^{1/3},
$$

where $\eta$ is a dimensionless resolution parameter (default $\eta = 1.2$). After each drift step, $\rho_i$ is recomputed from the kernel sum and $h_i$ is updated before force evaluation.

### 2.4 Density Estimate

The SPH density of particle $i$ is

$$
\rho_i = \sum_j m_j\, W_{ij}.
$$

### 2.5 Euler Momentum Equation

The symmetrized pressure-gradient acceleration is

$$
\mathbf{a}_i^{\mathrm{hydro}}
= -\sum_j m_j \left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} + \Pi_{ij} \right) \nabla_i W_{ij},
$$

where $\Pi_{ij}$ is the **Monaghan artificial viscosity** tensor (scalar form) activated only for converging flow.

### 2.6 Monaghan Artificial Viscosity

Define $\mathbf{v}_{ij} \equiv \mathbf{v}_i - \mathbf{v}_j$ and the standard signal

$$
\mu_{ij} = \frac{h_{ij}\, (\mathbf{v}_{ij}\cdot\mathbf{r}_{ij})}{r_{ij}^2 + \eta_{\mathrm{AV}}^2 h_{ij}^2},
$$

where $\eta_{\mathrm{AV}}$ is a small softening parameter (implementation default $\eta_{\mathrm{AV}} = 0.01$). Let $c_{s,ij} = \tfrac{1}{2}(c_{s,i} + c_{s,j})$ denote the pairwise sound speed and $\bar{\rho}_{ij} = \tfrac{1}{2}(\rho_i + \rho_j)$. Then

$$
\Pi_{ij} =
\begin{cases}
\dfrac{-\alpha_{\mathrm{AV}}\, c_{s,ij}\, \mu_{ij} + \beta_{\mathrm{AV}}\, \mu_{ij}^2}{\bar{\rho}_{ij}},
& \mathbf{v}_{ij}\cdot\mathbf{r}_{ij} < 0, \\[8pt]
0, & \mathbf{v}_{ij}\cdot\mathbf{r}_{ij} \ge 0.
\end{cases}
$$

The switch on $\mathbf{v}_{ij}\cdot\mathbf{r}_{ij}$ restricts dissipation to approaching particle pairs, preventing spurious heating in expanding regions. Default coefficients are $\alpha_{\mathrm{AV}} = 1.0$, $\beta_{\mathrm{AV}} = 2.0$.

The sound speed follows from the ideal-gas closure (Section 3):

$$
c_{s,i} = \sqrt{\gamma \frac{P_i}{\rho_i}}.
$$

---

## 3. Thermodynamic Closures and Equation of State

### 3.1 Internal Energy Evolution

The Lagrangian rate of change of specific internal energy for particle $i$ is driven by pressure-volume (PdV) work and viscous dissipation:

$$
\frac{du_i}{dt}
= \frac{1}{2}\sum_j m_j
\left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} + \Pi_{ij} \right)
\mathbf{v}_{ij}\cdot\nabla_i W_{ij}.
$$

This form is consistent with the symmetrized momentum equation and conserves total energy (kinetic plus internal plus external potential) to truncation order when coupled with the symplectic integrator of Section 5.

### 3.2 Ideal Gas Equation of State

During dynamical evolution, pressure is recovered from $(\rho_i, u_i)$ via the ideal gas law

$$
P_i = (\gamma - 1)\,\rho_i\, u_i,
$$

with adiabatic index $\gamma = 5/3$ appropriate for a non-relativistic, monatomic fully ionized plasma. The internal energy density is therefore $e_i = \rho_i u_i$.

### 3.3 Polytropic Initial Stellar Profiles

#### 3.3.1 Lane–Emden equation

Initial stellar models are polytropes of index $n$ satisfying

$$
P = K \rho^{1 + 1/n} \equiv K \rho^{\Gamma_p},
\qquad \Gamma_p = 1 + \frac{1}{n}.
$$

Hydrostatic equilibrium reduces to the Lane–Emden equation in dimensionless radius $\xi \equiv r/\alpha$:

$$
\frac{1}{\xi^2}\frac{d}{d\xi}\!\left(\xi^2 \frac{d\theta}{d\xi}\right) = -\theta^n,
$$

with central conditions $\theta(0) = 1$ and $d\theta/d\xi|_{\xi=0} = 0$. The surface $\xi_1$ is defined by $\theta(\xi_1) = 0$.

The physical density profile is

$$
\rho(r) = \rho_c\, \theta^n\!\left(\frac{r}{\alpha}\right),
$$

where $\rho_c$ is the central density and $\alpha$ the polytropic scale radius.

#### 3.3.2 Mass normalization

The total mass constraint

$$
M = 4\pi \rho_c \alpha^3 \int_0^{\xi_1} \xi^2 \theta^n(\xi)\, d\xi
$$

determines $\rho_c$ once $\alpha$ is fixed by the stellar radius $R_\star = \alpha \xi_1$.

#### 3.3.3 Polytropic constant

The polytropic constant follows from the structure integral:

$$
K = \frac{4\pi G \alpha^2 \rho_c}{n+1}\,\Gamma_p.
$$

For $n = 1.5$ (fully convective, degenerate-matter proxy), $\Gamma_p = 5/3$, consistent with the dynamical ideal-gas closure.

#### 3.3.4 Initial thermal state

At initialization, each particle receives

$$
P_i = K \rho_i^{\Gamma_p}, \qquad
u_i = \frac{P_i}{(\gamma - 1)\rho_i}.
$$

---

## 4. Orbital Geometry and Initial Conditions

### 4.1 Lane–Emden to Cartesian SPH Mapping

Particle positions are drawn from the three-dimensional probability density $\rho(\mathbf{r}) \propto \rho(r)$ using a two-stage procedure:

1. **Radial inverse-CDF sampling.** The cumulative mass shell distribution

   $$
   M(<\xi) \propto \int_0^{\xi} s^2 \theta^n(s)\, ds
   $$

   is inverted to sample $\xi$, then $r = \alpha \xi$.

2. **Angular isotropy.** Directions are drawn uniformly on the unit sphere via $(\cos\theta, \phi)$, yielding $\mathbf{r} = r\,(\sin\theta\cos\phi,\, \sin\theta\sin\phi,\, \cos\theta)$.

3. **Rejection refinement (optional).** Accept points with probability $\rho(\mathbf{r})/\rho_c$ to correct discretization bias when required.

Equal-mass particles $m_i = M/N$ are assigned. Initial smoothing lengths follow a mean-density estimate $h_i \sim 2 (3 m_i / 4\pi \rho_i)^{1/3}$.

### 4.2 Hyperbolic Binary Orbit

The stellar binary center of mass follows a Keplerian hyperbola ($e > 1$) in the Paczyński–Wiita field, parameterized by:

| Parameter | Definition |
|-----------|------------|
| Schwarzschild radius | $r_s = 2 G M_\bullet / c^2$ |
| Periapsis | $r_p = r_{p,\mathrm{rsch}}\, r_s$ |
| Eccentricity | $e > 1$ |
| Speed at infinity | $v_\infty$ |

The specific orbital angular momentum at periapsis is

$$
h = \sqrt{G M_\bullet\, r_p\, (1 + e)}.
$$

At injection radius $r_0$ (default $r_0 = 50\, r_p$), the hyperbolic excess speed gives

$$
v_0 = \sqrt{v_\infty^2 + \frac{2 G M_\bullet}{r_0}},
\qquad
v_{t,0} = \frac{h}{r_0},
\qquad
v_{r,0} = \sqrt{v_0^2 - v_{t,0}^2}.
$$

The orbit lies in the $xy$-plane with periapsis on $+x$; the center of mass is placed at $\mathbf{r}_{\mathrm{COM}} = (-r_0, 0, 0)$ with velocity $\mathbf{v}_{\mathrm{COM}} = (v_{r,0}, v_{t,0}, 0)$, inbound toward the SMBH at the origin.

### 4.3 Binary Separation

Two polytropic spheres (primary mass $M_1$, secondary $M_2$) are offset transverse to the orbital velocity by a separation

$$
d = f_{\mathrm{sep}}\,(R_1 + R_2),
$$

with center-of-mass offsets $\mathbf{r}_1 = \mathbf{r}_{\mathrm{COM}} - (M_2/M_{\mathrm{tot}})\, d\,\hat{\mathbf{y}}$ and $\mathbf{r}_2 = \mathbf{r}_{\mathrm{COM}} + (M_1/M_{\mathrm{tot}})\, d\,\hat{\mathbf{y}}$. Each star's SPH particles inherit the bulk COM velocity.

---

## 5. Symplectic Integrator and CFL Dynamic Timestepping

### 5.1 Kick–Drift–Kick (Leapfrog) Scheme

Let $\mathbf{a}_i^n$ denote the total acceleration and $\dot{u}_i^n$ the specific internal-energy rate at step $n$. The velocity-Verlet (KDK) update over timestep $\Delta t$ is:

**Half-kick (velocity and internal energy):**

$$
\mathbf{v}_i^{n+1/2} = \mathbf{v}_i^{n} + \tfrac{1}{2}\,\mathbf{a}_i^{n}\,\Delta t,
\qquad
u_i^{n+1/2} = u_i^{n} + \tfrac{1}{2}\,\dot{u}_i^{n}\,\Delta t.
$$

**Drift (positions):**

$$
\mathbf{r}_i^{n+1} = \mathbf{r}_i^{n} + \mathbf{v}_i^{n+1/2}\,\Delta t.
$$

**Hydro sync:** recompute $h_i$, $\rho_i$, and $P_i$ from updated positions.

**Second half-kick:**

$$
\mathbf{v}_i^{n+1} = \mathbf{v}_i^{n+1/2} + \tfrac{1}{2}\,\mathbf{a}_i^{n+1}\,\Delta t,
\qquad
u_i^{n+1} = u_i^{n+1/2} + \tfrac{1}{2}\,\dot{u}_i^{n+1}\,\Delta t.
$$

Gravitational and hydrodynamic accelerations are evaluated coherently at the endpoints of each substep. The scheme is symplectic in the Hamiltonian sector (gravity) and second-order accurate in $\Delta t$ for the coupled hydro-gravity system.

### 5.2 Courant–Friedrichs–Lewy Condition

Define the pairwise signal speed for approaching neighbors ($\mathbf{v}_{ij}\cdot\mathbf{r}_{ij} < 0$):

$$
v_{\mathrm{sig},ij} = c_{s,i} + c_{s,j} - 3\,\mu_{ij}.
$$

For each particle $i$, let

$$
v_{\mathrm{sig},i} = \max_j\, v_{\mathrm{sig},ij}
$$

over all neighbors $j$ within $2 h_{ij}$. The adaptive timestep is

$$
\Delta t_{\mathrm{CFL}}
= C_{\mathrm{CFL}}\,
\min_i \left(
\frac{h_i}{c_{s,i} + \alpha_{\mathrm{AV}}\, v_{\mathrm{sig},i}}
\right),
$$

where $C_{\mathrm{CFL}}$ is a safety factor (default $C_{\mathrm{CFL}} = 0.2$). The operational timestep is

$$
\Delta t = \min\!\big(\Delta t_{\mathrm{CFL}},\; \Delta t_{\max}\big),
\qquad
\Delta t \ge \Delta t_{\min}.
$$

This bounds acoustic and shock propagation across the smallest smoothing lengths, which is essential for stability during hyper-velocity impact events.

### 5.3 Conserved Quantities and Diagnostics

The engine monitors:

- **Total energy:** $E = E_{\mathrm{kin}} + E_{\mathrm{int}} + E_{\mathrm{grav}}$, with $E_{\mathrm{grav}} = \sum_i m_i \Phi_{\mathrm{PW}}(r_i)$.
- **Linear momentum:** $\mathbf{P} = \sum_i m_i \mathbf{v}_i$.

Drift in $E$ and $\mathbf{P}$ quantifies cumulative timestep truncation, artificial viscosity dissipation, and PW regularization effects.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $G$ | Newtonian gravitational constant |
| $c$ | Speed of light |
| $M_\bullet$ | SMBH mass |
| $a$ | Dimensionless Kerr spin parameter |
| $r_g$, $r_s$ | Gravitational and Schwarzschild radii |
| $h_i$, $\eta$ | Smoothing length and resolution ratio |
| $\gamma$ | Ideal-gas adiabatic index ($5/3$) |
| $n$, $\Gamma_p$ | Polytropic index and exponent |
| $\alpha_{\mathrm{AV}}$, $\beta_{\mathrm{AV}}$ | Artificial viscosity coefficients |
| $C_{\mathrm{CFL}}$ | CFL safety factor |
| $e$, $r_p$, $v_\infty$ | Orbital eccentricity, periapsis, asymptotic speed |

---

*hvit-engine v0.1 — native NumPy SPH with Paczyński–Wiita + Lense–Thirring gravity.*
