"""Calibrate the Newton cloth's stiffness against standard fabric tests of 12 oz denim.

    python scripts/fabric_calibrate.py                  # at the [newton] spacing
    python scripts/fabric_calibrate.py --spacing 0.005  # another resolution (the stiffness depends on it)

VBD's bending and stretch stiffness are per hinge and per triangle, so the numbers that make a
strip of cloth behave like denim depend on the mesh. This runs three textile tests on a strip
or disc of the fabric at the given spacing, with the engine's own solver settings, and finds the
stiffness that reproduces them:

  * bending, ASTM D1388 cantilever: a strip 25 mm wide hangs over an edge. Its bending length c
    is half the overhang at which the tip droops to 41.5 degrees. Rigid 12-14 oz denim measures
    c ~ 3-4.5 cm along the warp; target 3.5 cm (flexural rigidity G = w c^3 ~ 1.7e-4 N m).
  * stretch, Kawabata EMT style: a strip 50 mm wide hangs from its top edge with a load of
    500 gf/cm (49 N/m) on the bottom edge. A tester reads ~2-4 % for rigid denim (warp), but most
    of that is the crimp of the yarns straightening at almost no load; past it denim locks. VBD's
    stretch is linear, and fitted to 3 % the jeans stretched 16 % (p99) in a scripted fold and
    stayed so, so the target is the locked fabric's ~1 % (--emt-strain).
  * drape, Cusick (BS 5058) as a check, not fitted: a 30 cm disc on an 18 cm pedestal.
    Drape coefficient = share of the overhanging ring's area that stays spread out; denim
    ~0.6-0.85 (stiff), a cotton shirting ~0.4, a silk ~0.2.

The areal density is that of 12 oz/yd^2 denim, 0.407 kg/m^2 (one layer). The fitted edge_ke /
tri_ke go into config/cloth.toml's [newton] section by hand; the script prints the lines.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import warp as wp  # noqa: E402

import newton  # noqa: E402

from vrteleop import cloth as C  # noqa: E402
from vrteleop import cloth_config as CC  # noqa: E402

wp.config.quiet = True

DENIM_DENSITY = 0.407          # kg/m^2: 12 oz per square yard
G = 9.81
CANTILEVER_C = 0.035           # target bending length [m]
CANTILEVER_ANGLE = np.radians(41.5)
EMT_LOAD = 49.0                # N/m (500 gf/cm)
EMT_STRAIN = 0.01


@wp.kernel
def _damp(qd: wp.array[wp.vec3], f: float):
    """Quasi-static tests: bleed the kinetic energy so the strip settles in a few seconds."""
    i = wp.tid()
    qd[i] = qd[i] * f


def grid(nx: int, ny: int, h: float):
    """(nx+1)(ny+1) points in the x-y plane, h apart, and triangles with alternating diagonals."""
    X = np.array([(i * h, j * h, 0.0) for i in range(nx + 1) for j in range(ny + 1)])
    F = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = i * (ny + 1) + j, i * (ny + 1) + j + 1, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1
            F += [(a, c, b), (b, c, d)] if (i + j) % 2 == 0 else [(a, c, d), (a, d, b)]
    return X, np.array(F, np.int32)


def simulate(X, F, fixed, p: CC.NewtonSettings, tri_ke, edge_ke, seconds, extra_mass=None,
             self_contact=False, device="cuda"):
    """The fabric at rest positions X under gravity (-z) with points `fixed` held, after
    `seconds`. Same solver and substeps as the jeans."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                     vertices=[wp.vec3(*map(float, v)) for v in X], indices=F.ravel().tolist(),
                     density=DENIM_DENSITY, tri_ke=tri_ke, tri_ka=tri_ke * p.tri_ka / p.tri_ke,
                     tri_kd=p.tri_kd, edge_ke=edge_ke, edge_kd=p.edge_kd, particle_radius=p.particle_radius)
    b.color()
    m = b.finalize(device=device)
    mass = m.particle_mass.numpy()
    if extra_mass is not None:
        mass = mass + extra_mass
    mass = np.where(fixed, 0.0, mass)                       # zero mass: VBD leaves the point where it is
    inv = np.where(fixed, 0.0, 1.0 / np.maximum(mass, 1e-12))
    m.particle_mass.assign(mass.astype(np.float32))
    m.particle_inv_mass.assign(inv.astype(np.float32))
    solver = newton.solvers.SolverVBD(m, iterations=p.iterations, particle_enable_self_contact=self_contact,
                                      particle_self_contact_margin=p.self_contact_margin,
                                      particle_self_contact_gap=p.self_contact_gap)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    dt = 1.0 / p.fps / p.substeps
    for _ in range(int(seconds * p.fps)):
        if self_contact:
            solver.rebuild_bvh(s0)
        for _ in range(p.substeps):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, None, dt)
            wp.launch(_damp, dim=m.particle_count, inputs=[s1.particle_qd, 0.97], device=device)
            s0, s1 = s1, s0
    return s0.particle_q.numpy()


def cantilever_angle(p, h, tri_ke, edge_ke, overhang=2 * CANTILEVER_C, clamp=0.03, width=0.025):
    """Tip droop [rad] of a clamped strip `overhang` long beyond the edge."""
    nc = int(round(clamp / h))
    nx, ny = nc + int(round(overhang / h)), max(2, int(round(width / h)))
    X, F = grid(nx, ny, h)
    X[:, 0] -= nc * h                                          # a row of points on the edge
    fixed = X[:, 0] <= 1e-9
    Q = simulate(X, F, fixed, p, tri_ke, edge_ke, seconds=5.0)
    tip = Q[X[:, 0] > X[:, 0].max() - 1e-9].mean(axis=0)
    return float(np.arctan2(-tip[2], tip[0]))


def emt_strain(p, h, tri_ke, edge_ke, length=0.20, width=0.05):
    """Warp strain of a strip hung from its top edge with 49 N/m on its bottom edge."""
    nx, ny = int(round(length / h)), max(2, int(round(width / h)))
    X, F = grid(nx, ny, h)
    X = np.c_[X[:, 1], np.zeros(len(X)), -X[:, 0]]           # hang along -z
    top, bottom = X[:, 2] >= -1e-9, X[:, 2] <= X[:, 2].min() + 1e-9
    extra = np.where(bottom, EMT_LOAD * width / G / bottom.sum(), 0.0)
    Q = simulate(X, F, top, p, tri_ke, edge_ke, seconds=3.0, extra_mass=extra)
    # the middle half of the strip (clear of the clamp and the loaded edge)
    rows = np.unique(np.round(X[:, 2], 6))
    r0, r1 = rows[len(rows) // 4], rows[3 * len(rows) // 4]
    z0 = Q[np.isclose(X[:, 2], r0), 2].mean()
    z1 = Q[np.isclose(X[:, 2], r1), 2].mean()
    return float(abs(z1 - z0) / abs(r1 - r0) - 1.0)


def drape_coefficient(p, h, tri_ke, edge_ke, R=0.15, r=0.09):
    """Cusick drape coefficient of a disc of radius R on a pedestal of radius r."""
    n = int(round(2 * R / h))
    X, F = grid(n, n, h)
    X[:, :2] -= R
    inside = np.linalg.norm(X[:, :2], axis=1) <= R + 1e-9
    F = F[inside[F].all(axis=1)]
    used = np.unique(F)
    remap = -np.ones(len(X), int)
    remap[used] = np.arange(len(used))
    X, F = X[used], remap[F].astype(np.int32)
    fixed = np.linalg.norm(X[:, :2], axis=1) <= r
    Q = simulate(X, F, fixed, p, tri_ke, edge_ke, seconds=4.0, self_contact=True)
    a = C.footprint_area(Q[:, :2], F, res=0.002)
    disc = C.footprint_area(X[:, :2], F, res=0.002)
    ped = np.pi * r * r
    return float((a - ped) / (disc - ped))


def fit(f, lo, hi, target, increasing, tol=0.01, steps=14):
    """Log-space bisection of a monotonic f(x) = target."""
    for _ in range(steps):
        x = np.sqrt(lo * hi)
        y = f(x)
        print(f"    {x:10.4g} -> {y:.4f}", flush=True)
        if abs(y - target) <= tol * abs(target):
            return x, y
        if (y < target) == increasing:
            lo = x
        else:
            hi = x
    return x, y


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--spacing", type=float, default=None)
    a.add_argument("--skip-drape", action="store_true")
    a.add_argument("--emt-strain", type=float, default=EMT_STRAIN, help="target strain at 49 N/m")
    a.add_argument("--iterations", type=int, default=None, help="VBD iterations (default: [newton])")
    args = a.parse_args()
    p = CC.load().newton
    h = args.spacing or p.spacing
    if args.iterations:
        p.iterations = args.iterations
    t0 = time.perf_counter()
    print(f"spacing {h * 1000:.1f} mm, {p.substeps} substeps x {p.iterations} iterations, "
          f"{DENIM_DENSITY} kg/m^2")
    print(f"  now: edge_ke {p.edge_ke:g} -> cantilever droop "
          f"{np.degrees(cantilever_angle(p, h, p.tri_ke, p.edge_ke)):.1f} deg (target 41.5), "
          f"tri_ke {p.tri_ke:g} -> EMT strain {100 * emt_strain(p, h, p.tri_ke, p.edge_ke):.2f} % "
          f"(target {100 * args.emt_strain:g})")
    # stretch first (a hanging strip hardly bends), then the bending with that stretch stiffness
    print("  stretch (EMT strain at 49 N/m):")
    tk, eps = fit(lambda k: emt_strain(p, h, k, p.edge_ke), 1e1, 1e6, args.emt_strain, increasing=False, tol=0.03)
    print("  bending (cantilever droop at 2c = 7 cm, degrees):")
    ke, ang = fit(lambda k: np.degrees(cantilever_angle(p, h, tk, k)), 1e-3, 1e3,
                  np.degrees(CANTILEVER_ANGLE), increasing=False)
    # VBD stops converging for stiff hinges on fine meshes: the droop then hardly depends on the
    # stiffness, and the fitted value is not a measurement of anything
    stiff = np.degrees(cantilever_angle(p, h, tk, 10 * ke))
    if np.degrees(CANTILEVER_ANGLE) - stiff < 10:
        print(f"  WARNING: 10x the bending stiffness still droops {stiff:.0f} deg: the solver does not "
              f"converge for stiff bending at this spacing; the fit is unreliable")
    print(f"\nfitted at {h * 1000:.1f} mm: edge_ke = {ke:.4g} (droop {ang:.1f} deg), "
          f"tri_ke = tri_ka = {tk:.4g} (strain {100 * eps:.2f} %)")
    if not args.skip_drape:
        dc_old = drape_coefficient(p, h, p.tri_ke, p.edge_ke)
        dc = drape_coefficient(p, h, tk, ke)
        print(f"Cusick drape coefficient: {dc:.2f} fitted, {dc_old:.2f} before (denim ~0.6-0.85)")
    print(f"[{time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
