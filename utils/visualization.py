import os
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ────────────────────────────────────────────────────────────
# Shared helpers (identical to the spacecraft project)
# ────────────────────────────────────────────────────────────
def _rgba(color, alpha):
    """Accepts 'rgb(r,g,b)', '#rrggbb', or a matplotlib color name/tuple and
    returns an 'rgba(...)' string. method.color is used throughout."""
    import matplotlib.colors as mcolors
    if isinstance(color, str) and color.startswith('rgb('):
        r, g, b = color[4:-1].split(',')
        return f"rgba({r.strip()},{g.strip()},{b.strip()},{alpha})"
    r, g, b = (int(255 * c) for c in mcolors.to_rgb(color))
    return f"rgba({r},{g},{b},{alpha})"


def _rgb(color):
    """Same conversion as _rgba but opaque — used for marker/line colors."""
    import matplotlib.colors as mcolors
    if isinstance(color, str) and color.startswith('rgb('):
        return color
    r, g, b = (int(255 * c) for c in mcolors.to_rgb(color))
    return f"rgb({r},{g},{b})"


def calculate_statistics(hypervolumes_array):
    """Calculate statistics for hypervolumes"""
    return {
        'max':    np.max(hypervolumes_array, axis=0),
        'min':    np.min(hypervolumes_array, axis=0),
        'median': np.median(hypervolumes_array, axis=0),
        'q25':    np.percentile(hypervolumes_array, 25, axis=0),
        'q75':    np.percentile(hypervolumes_array, 75, axis=0),
    }


# ────────────────────────────────────────────────────────────
# 1) Hypervolume comparison  (HTML / Plotly)
# ────────────────────────────────────────────────────────────
def plot_hypervolumes(storage, methods, params):
    """Interactive HV comparison. Median line + shaded IQR band + dashed
    max / dotted min, one group per method. Legend toggles methods on/off."""
    fig = go.Figure()
    any_data = False

    for method in methods:
        key = method.name.replace(" ", "_").lower()
        if key not in storage or not storage[key]['all_runs_hypervolumes']:
            continue
        hv = np.array(storage[key]['all_runs_hypervolumes'])
        if hv.size == 0:
            continue
        any_data = True
        s = calculate_statistics(hv)
        x = list(range(len(s['median'])))
        c = method.color
        grp = method.name

        # IQR band (q25..q75) as a filled region
        fig.add_trace(go.Scatter(x=x + x[::-1],
                                 y=list(s['q75']) + list(s['q25'])[::-1],
                                 fill='toself', fillcolor=_rgba(c, 0.12),
                                 line=dict(width=0), hoverinfo='skip',
                                 legendgroup=grp, showlegend=False,
                                 name=f'{grp} IQR'))
        fig.add_trace(go.Scatter(x=x, y=s['median'], mode='lines',
                                 line=dict(color=_rgb(c), width=2.5), name=grp,
                                 legendgroup=grp,
                                 hovertemplate='iter %{x}<br>median HV %{y:.4f}<extra>' + grp + '</extra>'))
        # fig.add_trace(go.Scatter(x=x, y=s['max'], mode='lines',
        #                          line=dict(color=_rgb(c), width=1, dash='dash'),
        #                          name=f'{grp} max', legendgroup=grp, showlegend=False,
        #                          hoverinfo='skip'))
        # fig.add_trace(go.Scatter(x=x, y=s['min'], mode='lines',
        #                          line=dict(color=_rgb(c), width=1, dash='dot'),
        #                          name=f'{grp} min', legendgroup=grp, showlegend=False,
        #                          hoverinfo='skip'))

    if not any_data:
        print("  [viz] no hypervolume data to plot")
        return

    fig.update_layout(
        title='Hypervolume Comparison Across All Methods',
        xaxis_title='Iterations',
        yaxis_title='Hypervolume',
        hovermode='x unified',
        legend=dict(title='Method (click to toggle)'),
        template='plotly_white',
    )

    folder = f"results/{params['date_str']}"
    os.makedirs(folder, exist_ok=True)
    out = f"{folder}/hypervolume_comparison.html"
    plotly_plot(fig, filename=out, auto_open=False,
                include_plotlyjs='cdn',
                config={'responsive': True, 'displaylogo': False})
    print(f"  [viz] wrote {out}")


# ────────────────────────────────────────────────────────────
# 2) Pareto front (3D)  (HTML / Plotly)
# ────────────────────────────────────────────────────────────
def plot_pareto_front_3d(storage, methods, params):
    """Interactive 3D scatter of final Pareto fronts in objective space.
    Same data path as the matplotlib version: last run -> final front ->
    first three objective columns (Mass, Compliance, Neg Frequency)."""
    fig = go.Figure()
    plotted = False

    for method in methods:
        key = method.name.replace(" ", "_").lower()
        objs_runs = storage[key]['all_runs_pareto_front_obj']
        if not objs_runs or len(objs_runs) == 0:
            continue
        last_run = objs_runs[-1]
        if last_run is None:
            continue
        final_pfront = last_run[-1]
        if final_pfront is None:
            continue
        arr = np.array(final_pfront)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] >= 3:
            pts = arr[:, :3]
            z = pts[:, 2]
        elif arr.shape[1] == 2:
            pts = arr[:, :2]
            z = np.zeros(arr.shape[0])
        else:
            continue

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=z,
            mode='markers',
            name=method.name,
            marker=dict(color=_rgb(method.color), size=5,
                        line=dict(color='black', width=1), opacity=0.9),
            hovertemplate=('<b>' + method.name + '</b><br>'
                           'Mass: %{x:.4f}<br>'
                           'Compliance: %{y:.4f}<br>'
                           'Neg Freq: %{z:.4f}<extra></extra>'),
        ))
        plotted = True

    if not plotted:
        print("  [viz] no Pareto data to plot")
        return

    fig.update_layout(
        title='Pareto Fronts: Truss Optimization (3D)',
        scene=dict(
            xaxis_title='Mass (lbs)',
            yaxis_title='Compliance',
            zaxis_title='Neg Frequency (Hz)',
        ),
        legend=dict(title='Method (click to toggle)'),
        template='plotly_white',
        margin=dict(l=0, r=0, t=50, b=0),
    )

    folder = f"results/{params['date_str']}"
    os.makedirs(folder, exist_ok=True)
    out = f"{folder}/pareto_front_3d.html"
    plotly_plot(fig, filename=out, auto_open=False,
                include_plotlyjs='cdn',
                config={'responsive': True, 'displaylogo': False})
    print(f"  [viz] wrote {out}")


# ────────────────────────────────────────────────────────────
# 3) Truss configuration renderer (3D)  (HTML / Plotly)
# ────────────────────────────────────────────────────────────
def render_truss(problem, design, save_name, deformation_scale=100.0,
                 show_constraints=True):
    """Interactive 3D truss viewer exported to HTML.

    Mirrors TrussProblem.render: decodes the design, re-runs the static solve
    for deformation + element stresses, then draws each active member colored
    by |stress| (jet) with width proportional to the chosen area action.
    """
    discrete_actions = problem._decode(design)
    res = problem.solve(design)

    if not res['valid']:
        print(f"Cannot visualize {save_name}: Structure Invalid.")
        return

    # --- Re-run the static solve for deformation vectors (as in render) ---
    areas = np.array([problem.area_map[a] for a in discrete_actions])
    n_dof = problem.n_nodes * 3
    K_global = np.zeros((n_dof, n_dof))
    for idx, (n1, n2) in enumerate(problem.elements):
        p1, p2 = problem.nodes[n1], problem.nodes[n2]
        L = np.linalg.norm(p2 - p1)
        cx, cy, cz = (p2 - p1) / L
        A = areas[idx]
        k_val = (problem.E * A / L)
        T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
        k_local = np.array([[1, -1], [-1, 1]]) * k_val
        k_elem = T.T @ k_local @ T
        indices = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        for r in range(6):
            for c in range(6):
                K_global[indices[r], indices[c]] += k_elem[r, c]

    fixed_dofs = []
    for n in problem.fixed_nodes:
        fixed_dofs.extend([n*3, n*3+1, n*3+2])
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]

    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = problem.forces[free_dofs]
    u_reduced = spsolve(coo_matrix(K_reduced), F_reduced)
    u_full = np.zeros(n_dof)
    u_full[free_dofs] = u_reduced

    element_stresses = []
    for idx, (n1, n2) in enumerate(problem.elements):
        p1, p2 = problem.nodes[n1], problem.nodes[n2]
        L = np.linalg.norm(p2 - p1)
        u_el = np.concatenate([u_full[n1*3:n1*3+3], u_full[n2*3:n2*3+3]])
        vec = (p2 - p1) / L
        delta_l = np.dot(u_el[3:] - u_el[:3], vec)
        stress = (problem.E * delta_l) / L
        element_stresses.append(stress)
    element_stresses = np.array(element_stresses)

    # --- Build Plotly traces: one line per active member, colored by |stress| ---
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    cmap = cm.get_cmap('jet')
    norm = colors.Normalize(vmin=0, vmax=problem.stress_limit)

    deformed_nodes = problem.nodes + u_full.reshape(-1, 3) * deformation_scale

    fig = go.Figure()
    drew_colorbar = False

    for idx, (n1, n2) in enumerate(problem.elements):
        action = discrete_actions[idx]
        if action == 0:
            continue

        d1 = u_full[n1*3:n1*3+3] * deformation_scale
        d2 = u_full[n2*3:n2*3+3] * deformation_scale
        coord_1 = problem.nodes[n1] + d1
        coord_2 = problem.nodes[n2] + d2

        stress_val = abs(element_stresses[idx])
        rgba = cmap(norm(stress_val))
        line_color = f"rgb({int(255*rgba[0])},{int(255*rgba[1])},{int(255*rgba[2])})"
        linewidth = max(1.0, action * 0.1)

        fig.add_trace(go.Scatter3d(
            x=[coord_1[0], coord_2[0]],
            y=[coord_1[1], coord_2[1]],
            z=[coord_1[2], coord_2[2]],
            mode='lines',
            line=dict(color=line_color, width=linewidth * 3),
            opacity=0.9,
            showlegend=False,
            hovertemplate=(f'<b>Member {idx}</b><br>'
                           f'Stress: {element_stresses[idx]:.1f} psi<br>'
                           f'Area idx: {action}<extra></extra>'),
        ))

    # Nodes (deformed)
    fig.add_trace(go.Scatter3d(
        x=deformed_nodes[:, 0], y=deformed_nodes[:, 1], z=deformed_nodes[:, 2],
        mode='markers',
        marker=dict(color='black', size=3),
        name='Nodes', showlegend=False,
        hovertemplate='Node<br>x %{x:.1f}<br>y %{y:.1f}<br>z %{z:.1f}<extra></extra>',
    ))

    # Invisible marker trace solely to render the stress colorbar (jet)
    fig.add_trace(go.Scatter3d(
        x=[deformed_nodes[0, 0]], y=[deformed_nodes[0, 1]], z=[deformed_nodes[0, 2]],
        mode='markers',
        marker=dict(
            size=0.1,
            color=[0, problem.stress_limit],
            colorscale='Jet',
            cmin=0, cmax=problem.stress_limit,
            colorbar=dict(title='Stress (psi)', len=0.5),
            showscale=True,
        ),
        hoverinfo='skip', showlegend=False,
    ))

    # --- Equal-aspect scaling, mirroring the matplotlib set_xlim/ylim/zlim ---
    max_range = np.array([
        problem.nodes[:, 0].max() - problem.nodes[:, 0].min(),
        problem.nodes[:, 1].max() - problem.nodes[:, 1].min(),
        problem.nodes[:, 2].max() - problem.nodes[:, 2].min(),
    ]).max() / 2.0
    mid_x = (problem.nodes[:, 0].max() + problem.nodes[:, 0].min()) * 0.5
    mid_y = (problem.nodes[:, 1].max() + problem.nodes[:, 1].min()) * 0.5
    mid_z = (problem.nodes[:, 2].max() + problem.nodes[:, 2].min()) * 0.5

    title = (f'{os.path.basename(save_name)}<br>'
             f'Mass: {res["mass"]:.1f} kg | Disp: {res["max_disp"]:.1f} m | Freq: {res["frequency"]:.1f} Hz')

    # equal-aspect scaling based on the DEFORMED geometry so nothing clips
    pad = 0.1  # 10% breathing room
    span = deformed_nodes.max(axis=0) - deformed_nodes.min(axis=0)
    max_range = span.max() / 2.0
    max_range *= (1 + pad)

    mid = (deformed_nodes.max(axis=0) + deformed_nodes.min(axis=0)) * 0.5
    mid_x, mid_y, mid_z = mid

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(title='Y', range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(title='Z', range=[mid_z - max_range, mid_z + max_range]),
            aspectmode='cube',
        ),
        template='plotly_white',
        margin=dict(l=0, r=0, t=60, b=0),
    )

    out = save_name + ".html"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plotly_plot(fig, filename=out, auto_open=False,
                include_plotlyjs='cdn',
                config={'responsive': True, 'displaylogo': False})
    print(f"  [viz] wrote {out}")


# ────────────────────────────────────────────────────────────
# 4) Batch configuration visualization
# ────────────────────────────────────────────────────────────
def visualize_configurations(storage, methods, params, max_designs=5):
    """Render final Pareto-front designs for each method to HTML.
    Same signature/behavior as the matplotlib version, so the main.py
    call site is unchanged."""
    from evaluation.truss_problem import TrussProblem

    vis_problem = TrussProblem(
        overlap_constraint=params.get('overlap_constraint', False))

    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        method_safe_name = method_key

        pareto_designs = storage[method_key]['all_runs_pareto_front_des']

        if (pareto_designs and len(pareto_designs) > 0 and
                pareto_designs[-1] and len(pareto_designs[-1]) > 0 and
                len(pareto_designs[-1][-1]) > 0):

            final_pareto_designs = pareto_designs[-1][-1]
            print(f"\nVisualizing {min(len(final_pareto_designs), max_designs)} "
                  f"designs for {method.name}...")

            for i, design in enumerate(final_pareto_designs):
                if isinstance(design, np.ndarray):
                    design = design.tolist()

                filename = f"results/{params['date_str']}/{method_safe_name}_design_{i}"

                try:
                    render_truss(vis_problem, design, save_name=filename)
                except Exception as e:
                    print(f"Error visualizing {method.name} design {i}: {e}")

                if i >= max_designs - 1:
                    break