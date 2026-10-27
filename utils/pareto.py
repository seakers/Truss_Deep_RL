"""Custom Pareto-front and hypervolume utilities (no pymoo dependency).

Drop-in replacement for the is_pareto_efficient + HV(ref_point) pattern used
throughout the reference optimizers. Inputs to `hypervolume` are expected to be
normalized to minimization against a unit reference point, matching the
existing convention.
"""
from copy import deepcopy
import numpy as np
from bisect import bisect_left, insort


def is_pareto_efficient(points, return_mask=True):
    """Find Pareto-efficient points (minimization). Matches reference usage.

    Args:
        points: (n, m) array of objective vectors (lower is better).
        return_mask: if True, return a boolean mask; else return indices.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    is_efficient = np.arange(n)
    next_point_index = 0
    pts = points.copy()

    while next_point_index < len(pts):
        nondominated = np.any(pts < pts[next_point_index], axis=1)
        nondominated[next_point_index] = True
        is_efficient = is_efficient[nondominated]
        pts = pts[nondominated]
        next_point_index = np.sum(nondominated[:next_point_index]) + 1

    if return_mask:
        mask = np.zeros(n, dtype=bool)
        mask[is_efficient] = True
        return mask
    return is_efficient


def _hv2d(points, ref_point):
    pts = points[np.all(points < ref_point, axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    order = np.argsort(pts[:, 0])[::-1]   # objective 0 DESCENDING
    pts = pts[order]
    area = 0.0
    prev_x = ref_point[0]
    best_y = ref_point[1]
    for x, y in pts:
        if y < best_y:
            area += (prev_x - x) * (ref_point[1] - y)
            prev_x = x
            best_y = y
    return float(area)

# def _hv2d(points, ref_point):
#     """Exact 2D hypervolume (minimization) via a single sorted sweep. O(n log n).

#     Sort by objective 0 ascending, sweep objective 1. Accumulate rectangle
#     area only when a point improves the second axis (i.e., lies on the
#     staircase). Handles arbitrary (non-Pareto-filtered) input correctly.
#     """
#     pts = points[np.all(points < ref_point, axis=1)]
#     if pts.shape[0] == 0:
#         return 0.0
#     order = np.argsort(pts[:, 0])
#     pts = pts[order]
#     area = 0.0
#     prev_x = ref_point[0]
#     best_y = ref_point[1]
#     for x, y in pts:
#         if y < best_y:
#             area += (prev_x - x) * (ref_point[1] - y)
#             prev_x = x
#             best_y = y
#     return float(area)


def _hv3d(points, ref_point):
    """Exact 3D hypervolume (minimization) via an O(n log n) dimension sweep.

    Sweep along objective 0 (z), from best (lowest) to worst. Maintain the 2D
    dominated area of the projection onto objectives (1, 2) incrementally in a
    sorted staircase. Between consecutive z-values we add a slab whose height
    is the z-interval and whose base is the *current* dominated 2D area.

    The staircase is a plain Python list of (x, y) kept sorted by x ascending,
    with y strictly decreasing along it. Sentinels bound the structure so
    neighbor lookups never fall off the ends. Uses only the stdlib `bisect`
    module (no third-party dependency).
    """
    pts = points[np.all(points < ref_point, axis=1)]
    if pts.shape[0] == 0:
        return 0.0

    x_ref, y_ref, z_ref = ref_point[0], ref_point[1], ref_point[2]

    # Sweep by the first objective (call it z) ascending: best z first.
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    # 2D staircase in the (objective1=x, objective2=y) plane, minimization.
    # Sorted by x ascending; on the front, y strictly decreases as x increases.
    # Sentinels (-inf, y_ref) and (x_ref, -inf) bound neighbor lookups.
    front = [(-np.inf, y_ref), (x_ref, -np.inf)]
    area = 0.0
    volume = 0.0
    prev_z = pts[0, 0]

    for z, x, y in pts:
        # Slab from prev_z up to current z uses the area accumulated so far.
        volume += area * (z - prev_z)
        prev_z = z

        # Position of x in the staircase.
        i = bisect_left(front, (x, -np.inf))

        # If the left neighbor already has y' <= y, the new (x, y) is dominated
        # in the projected plane and contributes no new area.
        if front[i - 1][1] <= y:
            continue

        # Remove points strictly dominated by (x, y): those with x' >= x and
        # y' >= y sit immediately to the right. Subtract their area strips.
        while front[i][1] >= y:
            rx, ry = front[i]
            left_y = front[i - 1][1]
            right_x = front[i + 1][0]
            area -= (right_x - rx) * (left_y - ry)
            del front[i]

        # Insert (x, y) and add its exposed rectangle: width to the right
        # neighbor, height up to the left neighbor's y.
        left_y = front[i - 1][1]
        right_x = front[i][0]
        area += (right_x - x) * (left_y - y)
        insort(front, (x, y))

    # Final slab from the last (largest) z out to the reference.
    volume += area * (z_ref - prev_z)
    return float(volume)


def _hv_recursive(points, ref_point):
    """Exact hypervolume for minimization via dimension sweep.

    Fast/exact specializations for 1-3 objectives (the range used here);
    recurses along the first axis for m >= 4 if ever needed.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0.0

    # Keep only points that dominate the reference (strictly below it).
    points = points[np.all(points < ref_point, axis=1)]
    if points.size == 0:
        return 0.0

    m = points.shape[1]

    if m == 1:
        return float(ref_point[0] - np.min(points[:, 0]))

    if m == 2:
        return _hv2d(points, ref_point)

    if m == 3:
        return _hv3d(points, ref_point)

    # General case (m >= 4): recurse by slicing along the first axis.
    order = np.argsort(points[:, 0])
    pts = points[order]
    volume = 0.0
    prev_x = pts[0, 0]
    for i in range(len(pts)):
        x = pts[i, 0]
        volume += _hv_recursive(pts[: i + 1, 1:], ref_point[1:]) * (x - prev_x)
        prev_x = x
    # Final slab to the reference along the first axis.
    volume += _hv_recursive(pts[:, 1:], ref_point[1:]) * (ref_point[0] - prev_x)
    return float(volume)


def hypervolume(front_points, ref_point):
    """Exact hypervolume of a Pareto front (minimization, unit ref by default).

    Args:
        front_points: (k, m) array already normalized to minimization.
        ref_point: (m,) reference point (typically np.ones(m)).
    """
    front_points = np.asarray(front_points, dtype=float)
    if front_points.ndim == 1:
        front_points = front_points.reshape(1, -1)
    if front_points.size == 0:
        return 0.0
    return _hv_recursive(front_points, np.asarray(ref_point, dtype=float))


def normalize_objectives(all_obj, objective_min_max, max_values=None):
    """Normalize + flip-to-minimization, matching the reference convention.

    Divides by max_values and applies (1 - norm) for 'max' objectives so the
    result is minimization against a unit reference point.

    Args:
        all_obj: (n, m) raw objective array.
        objective_min_max: list of 'min'/'max' per objective.
        max_values: optional (m,) normalizer. If None, derived from the data
                    as max * 1.1 + 1e-6.

    Returns:
        (hv_obj, max_values) where hv_obj is minimization-normalized.
    """
    all_obj = np.asarray(all_obj, dtype=float)
    if max_values is None:
        max_values = np.max(all_obj, axis=0) * 1.1 + 1e-6  # avoid div-by-zero

    norm_obj = all_obj / max_values
    hv_obj = deepcopy(norm_obj)
    for i, mm in enumerate(objective_min_max):
        if mm.lower() == "max":
            hv_obj[:, i] = 1.0 - norm_obj[:, i]  # flip maximization
    hv_obj = np.clip(hv_obj, 0.0, None)          # <-- round-6 safety net [22][17]
    return hv_obj, max_values


def pareto_progress(all_obj, all_constraints, objective_min_max,
                    max_values=None, ref_point=None):
    """Compute Pareto front + hypervolume history over evaluations (NFE).

    Mirrors calculate_pareto_progress / the post-processing loop in the
    reference optimizers, including infeasible-design handling and carrying
    the last hypervolume forward when no improvement occurs.
    """
    all_obj = np.asarray(all_obj, dtype=float).copy()
    all_constraints = np.asarray(all_constraints, dtype=bool)
    num_objectives = all_obj.shape[1]
    min_mask = np.array([mm.lower() == "min" for mm in objective_min_max])

    if ref_point is None:
        ref_point = np.ones(num_objectives)

    # Map infeasible designs to the reference point (max for min-objs, 0 for max-objs).
    hv_obj_tmp, max_values = normalize_objectives(
        np.where(all_constraints[:, None], 0.0, all_obj), objective_min_max, max_values
    )
    ref_values = max_values * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values

    hv_obj, _ = normalize_objectives(all_obj, objective_min_max, max_values)

    hypervolumes = []
    pareto_front_obj = []
    for i in range(len(all_obj)):
        feasible = ~all_constraints[: i + 1]
        sub = hv_obj[: i + 1][feasible]

        if sub.shape[0] == 0:
            hypervolumes.append(0.0)
            pareto_front_obj.append([])
            continue

        mask = is_pareto_efficient(sub, return_mask=True)
        front = sub[mask]
        hv = hypervolume(front, ref_point)

        # numerical issues sometimes cause a small drop, so only append if it's a strict improvement
        if not hypervolumes or hv > hypervolumes[-1] + 1e-12:
            hypervolumes.append(hv)
            pareto_front_obj.append(all_obj[: i + 1][feasible][mask])
        else:
            hypervolumes.append(hypervolumes[-1])
            pareto_front_obj.append(pareto_front_obj[-1])

    return pareto_front_obj, hypervolumes


def pareto_last(all_obj, all_constraints, objective_min_max, max_values=None, ref_point=None):
    """Compute the Pareto front and hypervolume of the last evaluation.

    Mirrors the final post-processing step in the reference optimizers.
    """
    all_obj = np.asarray(all_obj, dtype=float).copy()
    all_constraints = np.asarray(all_constraints, dtype=bool)
    num_objectives = all_obj.shape[1]
    min_mask = np.array([mm.lower() == "min" for mm in objective_min_max])

    if ref_point is None:
        ref_point = np.ones(num_objectives)

    # Map infeasible designs to the reference point (max for min-objs, 0 for max-objs).
    hv_obj_tmp, max_values = normalize_objectives(
        np.where(all_constraints[:, None], 0.0, all_obj), objective_min_max, max_values
    )
    ref_values = max_values * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values

    hv_obj, _ = normalize_objectives(all_obj, objective_min_max, max_values)

    feasible = ~all_constraints
    sub = hv_obj[feasible]

    if sub.shape[0] == 0:
        return [], 0.0

    mask = is_pareto_efficient(sub, return_mask=True)
    front = sub[mask]
    hv = hypervolume(front, ref_point)

    return all_obj[feasible][mask], hv



def postprocess_run(all_des, all_obj, all_constraints, all_constraint_vals,
                    objective_min_max, max_values, ref_point=None):
    """
    Shared replacement for the ~20-line post-processing block duplicated across
    every optimizer (random_search, genetic_algorithm, intelligent_mutation_ga,
    ppo_rand_weights_state, warm_start_ga, warm_start_intelligent_ga,
    design_repair, and both transfer_learning files).

    Uses the custom pareto.py (is_pareto_efficient, hypervolume). Produces a
    monotonic HV curve and tracks the design-space Pareto front in parallel so
    main.py's store_results / visualize_configurations keep working unchanged.

    Returns: (all_des, all_obj, pareto_front_des, pareto_front_obj,
              hypervolumes, NFE)
    """
    all_des = np.asarray(all_des)
    all_obj = np.asarray(all_obj, dtype=float).copy()
    all_constraints = np.asarray(all_constraints, dtype=bool)
    NFE = len(all_obj)

    num_objectives = len(all_obj[0])
    objective_min_max = list(objective_min_max)[:num_objectives]
    min_mask = np.array([mm.lower() == "min" for mm in objective_min_max])
    max_objectives = np.asarray(max_values[:num_objectives], dtype=float)

    if ref_point is None:
        ref_point = np.ones(num_objectives)

    # Pin infeasible designs to the reference (nadir on min, 0 on flipped max),
    # matching pareto.py's own infeasible handling.
    ref_values = max_objectives * ref_point
    ref_values[~min_mask] = 0.0
    all_obj[all_constraints] = ref_values[:num_objectives]

    # Normalize + flip-to-min via the custom util (single source of truth).
    hv_obj, _ = normalize_objectives(all_obj, objective_min_max, max_objectives)

    pareto_front_des = []
    pareto_front_obj = []
    hypervolumes = []

    for i in range(NFE):
        feasible = ~all_constraints[: i + 1]
        sub = hv_obj[: i + 1][feasible]

        if sub.shape[0] == 0:
            hypervolumes.append(0.0)
            pareto_front_obj.append([])
            pareto_front_des.append([])
            continue

        mask = is_pareto_efficient(sub, return_mask=True)
        hv = hypervolume(sub[mask], ref_point)

        if not hypervolumes or hv > hypervolumes[-1] + 1e-12:
            hypervolumes.append(hv)
            pareto_front_obj.append(hv_obj[: i + 1][feasible][mask])
            pareto_front_des.append(all_des[: i + 1][feasible][mask])
        else:
            hypervolumes.append(hypervolumes[-1])
            pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
            pareto_front_des.append(pareto_front_des[-1] if pareto_front_des else [])

    print(f"Number of valid designs (False in all_constraints): "
          f"{np.sum(all_constraints == False)}")

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE