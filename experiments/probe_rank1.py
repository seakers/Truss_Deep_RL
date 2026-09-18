"""Is a single-bar edit a rank-1 update to the stiffness matrix?

If it is, the exact objectives of every candidate single-bar mutation are available in
closed form from ONE factorisation, and the delta-HV oracle that round 10 measured at
+0.0453 (free) / -0.0428 (charged) stops needing the lookahead evaluations that made it
unaffordable.

The element stiffness is T^T k_local T with k_local = [[1,-1],[-1,1]] = v v^T, so
k_e = (E A / L) b b^T with b = T^T v = [d, -d]. K is therefore LINEAR in the areas and each
bar contributes a rank-1 term, which means Sherman-Morrison gives u' exactly. Mass is
exactly linear too. Only the eigenvalue needs an approximation, and there the standard one
is first-order Rayleigh using the eigenvector the solver already computes and the current
code discards.

This script checks all of that against full re-solves.
"""
import os, sys
import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))

from evaluation.truss_problem import TrussProblem

prob = TrussProblem()
NV, AL = prob.n_elements, len(prob.area_map)
areas_lut = prob.area_lut
free = prob.free_dofs
rng = np.random.default_rng(0)


def b_vectors():
    """The rank-1 direction for each element, restricted to free DOFs."""
    B = np.zeros((NV, prob.n_nodes * 3))
    for e in range(NV):
        dofs = prob.dof_map[e]
        B[e, dofs[:3]] += prob.dirs[e]
        B[e, dofs[3:]] -= prob.dirs[e]
    return B[:, free]


B_FREE = b_vectors()


def exact(design):
    r = prob.solve(design)
    return r


def predict(design, bar, new_level):
    """Closed-form prediction of the new design's objectives from one factorisation."""
    areas = areas_lut[np.asarray(design)]
    K, M_diag = prob._assemble(areas)
    Kr = K[np.ix_(free, free)]
    Mr = M_diag[free]

    lu, piv = lu_factor(Kr)
    u = lu_solve((lu, piv), prob.forces_free)

    lam, phi = eigh(Kr, np.diag(Mr), subset_by_index=[0, 0])
    lam = float(lam[0]); phi = phi[:, 0]                 # M-orthonormal: phi^T M phi = 1

    dA = areas_lut[new_level] - areas[bar]
    alpha = prob.E * dA / prob.lengths[bar]
    w = B_FREE[bar]

    # ---- displacement: exact, by Sherman-Morrison ----
    Kinv_w = lu_solve((lu, piv), w)
    denom = 1.0 + alpha * float(w @ Kinv_w)
    u_new = u - alpha * Kinv_w * float(w @ u) / denom

    u_full = np.zeros(prob.n_nodes * 3)
    u_full[free] = u_new
    max_disp = float(np.max(np.abs(u_full)))

    # ---- stress: exact, from the updated displacement field ----
    du = u_full[prob.dof_map[:, 3:]] - u_full[prob.dof_map[:, :3]]
    delta_l = np.einsum('ij,ij->i', du, prob.dirs)
    stress = prob.E * delta_l / prob.lengths

    # ---- mass: exact, analytic ----
    new_areas = areas.copy(); new_areas[bar] = areas_lut[new_level]
    mass = float(np.sum(prob.rho * new_areas * prob.lengths))

    # ---- eigenvalue: first-order Rayleigh (the only approximation) ----
    dM = np.zeros_like(Mr)
    dofs = prob.dof_map[bar]
    keep = np.isin(dofs, free)
    idx = np.searchsorted(free, dofs[keep])
    np.add.at(dM, idx, prob.rho * dA * prob.lengths[bar] / 2.0)
    dK_phi = alpha * w * float(w @ phi)
    lam_new = lam + float(phi @ dK_phi) - lam * float(np.sum(dM * phi * phi))
    freq = float(np.sqrt(max(lam_new, 0.0)) / (2 * np.pi))

    return dict(mass=mass, max_disp=max_disp, frequency=freq, stress=stress)


def main():
    n_designs, n_edits = 40, 25
    err = {k: [] for k in ('mass', 'max_disp', 'frequency', 'stress_absmax')}
    for _ in range(n_designs):
        design = list(rng.integers(1, AL, NV))
        for _ in range(n_edits):
            bar = int(rng.integers(0, NV))
            lvl = int(rng.integers(1, AL))
            d2 = list(design); d2[bar] = lvl
            t = exact(d2)
            if not isinstance(t, dict) or 'mass' not in t:
                continue
            p = predict(design, bar, lvl)

            def rel(a, b):
                return abs(a - b) / max(abs(b), 1e-12)

            err['mass'].append(rel(p['mass'], t['mass']))
            # t['max_disp'] is stored transformed as x/(x+1); invert it
            tmd = t['max_disp'] / (1.0 - t['max_disp'])
            err['max_disp'].append(rel(p['max_disp'], tmd))
            err['frequency'].append(rel(p['frequency'], t['frequency']))
            ts = np.abs(t['utilisation']) * prob.stress_limit
            err['stress_absmax'].append(
                float(np.max(np.abs(np.abs(p['stress']) - ts)) / max(prob.stress_limit, 1e-12)))

    print(f"{n_designs} designs x {n_edits} single-bar edits = {len(err['mass'])} comparisons\n")
    print(f"{'quantity':<16}{'median rel err':>16}{'90th pct':>12}{'max':>12}")
    for k, v in err.items():
        v = np.asarray(v)
        print(f"{k:<16}{np.median(v):>16.3e}{np.percentile(v,90):>12.3e}{v.max():>12.3e}")


if __name__ == '__main__':
    main()
