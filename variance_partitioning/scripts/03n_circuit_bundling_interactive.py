"""
03n_circuit_bundling_interactive.py  (lives in results/)

Generates a Radial Root plot to visualize which AST classes own which 
Attention Heads. The entire capacity of the model's Attention Heads 
(all layers and indices) forms the fixed outer unit circle. 
When a head is owned by an AST Class, a root grows inwards from the 
perimeter towards the class node in the center.

Extended with:
- SciPy Dendrogram Matrix Seriation (visually similar heads grouped)
- Dynamic light-up head labels 
- Dual legends (Layers top horizontal, ASTs right vertical)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# High-Contrast Categorical Palette
# ─────────────────────────────────────────────────────────────────────────────

SOFT_COLOURS = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7cb94"
]

def get_polar_route(r0, th0, r_highway, r3, th3, num_points=100):
    th0 = th0 % (2 * np.pi)
    th3 = th3 % (2 * np.pi)
    diff = th3 - th0
    if diff > np.pi:
        th3 -= 2 * np.pi
    elif diff < -np.pi:
        th3 += 2 * np.pi
        
    t = np.linspace(0, 1, num_points)
    r_t = (1-t)**3 * r0 + 3*(1-t)**2*t * r_highway + 3*(1-t)*t**2 * r_highway + t**3 * r3
    th_t = (1-t)**3 * th0 + 3*(1-t)**2*t * th0 + 3*(1-t)*t**2 * th3 + t**3 * th3
    
    return r_t * np.cos(th_t), r_t * np.sin(th_t)

def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

def get_polar_route(r0, th0, r_highway, r3, th3, num_points=100):
    """Bezier curve route from (r0,th0) through highway to (r3,th3)."""
    th0 = th0 % (2 * np.pi)
    th3 = th3 % (2 * np.pi)
    diff = th3 - th0
    if diff > np.pi:
        th3 -= 2 * np.pi
    elif diff < -np.pi:
        th3 += 2 * np.pi
    t = np.linspace(0, 1, num_points)
    r_t  = (1-t)**3*r0 + 3*(1-t)**2*t*r_highway + 3*(1-t)*t**2*r_highway + t**3*r3
    th_t = (1-t)**3*th0 + 3*(1-t)**2*t*th0 + 3*(1-t)*t**2*th3 + t**3*th3
    return r_t * np.cos(th_t), r_t * np.sin(th_t)

def make_layered_route(r_root, th_root, r_arc, node_thetas, node_radii_list, n_arc=40):
    """
    Per-node route:
      1. Radial arm from (r_root, th_root) to (r_arc, th_root)  — drawn once
      2. For each node: short arc at r_arc from th_root to th_node (shorter angular
         path, ≤ π), then radial spur outward to the node at r_node.
    """
    xs = [r_root * np.cos(th_root), r_arc * np.cos(th_root)]
    ys = [r_root * np.sin(th_root), r_arc * np.sin(th_root)]

    for th_n, r_n in zip(node_thetas, node_radii_list):
        # Shorter angular path: delta in (-π, π]
        delta = (th_n - th_root + np.pi) % (2 * np.pi) - np.pi
        arc_end = th_root + delta
        n_pts = max(3, int(n_arc * abs(delta) / np.pi))
        arc_ts = np.linspace(th_root, arc_end, n_pts)

        # Arc from root angle to node angle, then spur outward
        xs += [None] + list(r_arc * np.cos(arc_ts)) + [r_n * np.cos(arc_end)]
        ys += [None] + list(r_arc * np.sin(arc_ts)) + [r_n * np.sin(arc_end)]

    return xs, ys


def get_text_position(theta, inward=False):
    deg = np.degrees(theta) % 360
    if inward:
        if 22.5 <= deg < 67.5: return 'bottom left'
        elif 67.5 <= deg < 112.5: return 'bottom center'
        elif 112.5 <= deg < 157.5: return 'bottom right'
        elif 157.5 <= deg < 202.5: return 'middle right'
        elif 202.5 <= deg < 247.5: return 'top right'
        elif 247.5 <= deg < 292.5: return 'top center'
        elif 292.5 <= deg < 337.5: return 'top left'
        else: return 'middle left'
    else:
        if 22.5 <= deg < 67.5: return 'top right'
        elif 67.5 <= deg < 112.5: return 'top center'
        elif 112.5 <= deg < 157.5: return 'top left'
        elif 157.5 <= deg < 202.5: return 'middle left'
        elif 202.5 <= deg < 247.5: return 'bottom left'
        elif 247.5 <= deg < 292.5: return 'bottom center'
        elif 292.5 <= deg < 337.5: return 'bottom right'
        else: return 'middle right'

def main():
    parser = argparse.ArgumentParser(description="Interactive Radial Roots: Inward Edge Bundling")
    parser.add_argument("--stem", default="contrastive_stubs")
    parser.add_argument("--component", default="all", choices=["heads", "mlp", "residual", "all"], help="Which component to visualize")
    parser.add_argument("--in_dir", default="../results/output")
    parser.add_argument("--top_k", type=int, default=10, help="Top items per class/layer")
    parser.add_argument("--format", default="html", help="Output format (html recommended for interactive Plotly)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    in_dir = Path(__file__).parent / args.in_dir
    
    comps = ["heads", "mlp", "residual"] if args.component == "all" else [args.component]
    for component in comps:
        print(f"\n{'='*50}\nGenerating Radial Root plot for: {component.upper()}\n{'='*50}")
        args.component = component
        run_bundling(args, in_dir)

def run_bundling(args, in_dir):
    head_grid = []
    head_radii = {}
    head_owners = {}
    
    if args.component == "heads":
        total_layers = 8
        items_per_layer = 128
        # Load raw selectivity arrays so we can apply the threshold to ALL classes
        # independently — a head can belong to multiple classes if sel > thresh for each.
        # This avoids the argmax "single owner" artefact: a head with sel=0.5 for
        # both 'For' and 'While' will appear in both groups here.
        sel_path = in_dir / f"02l_{args.stem}_head_sel.npz"
        if not sel_path.exists():
            print(f"Missing {sel_path}"); return
        sel_data   = np.load(sel_path)
        sel        = sel_data["sel"]        # (C, L, H)
        ast_pure   = sel_data["ast_pure"].astype(bool)  # (L, H)
        # derive class list from ownership JSON (same order as 02l)
        own_path = in_dir / f"02l_{args.stem}_head_ownership.json"
        if not own_path.exists():
            print(f"Missing {own_path}"); return
        with open(own_path) as f:
            own_data = json.load(f)
        classes    = sorted(own_data.keys())
        SEL_THRESH = 0.4
        C_count, L_count, H_count = sel.shape
        for l in range(L_count):
            for h in range(H_count):
                if not ast_pure[l, h]:
                    continue
                hid = f"L{l}I{h}"
                for ci, cls in enumerate(classes):
                    if ci >= C_count:
                        continue
                    if float(sel[ci, l, h]) > SEL_THRESH:
                        if hid not in head_owners:
                            head_owners[hid] = []
                        if cls not in head_owners[hid]:
                            head_owners[hid].append(cls)
        n_multi = sum(1 for v in head_owners.values() if len(v) > 1)
        print(f"  Heads with multiple class owners: {n_multi} / {len(head_owners)}")
    else:
        # Load raw selectivity arrays (same structure as 02l for heads) and apply
        # SEL_THRESH independently to ALL classes — avoids the argmax artefact.
        npz_key  = "neurons" if args.component == "mlp" else "residual"
        sel_path = in_dir / f"02g_{args.stem}_selectivity_{npz_key}.npz"
        if not sel_path.exists():
            print(f"Missing {sel_path}"); return
        sel_data = np.load(sel_path)
        sel      = sel_data["sel"].astype(float)       # (C, L, N)
        classes  = list(sel_data["classes"])
        C_count, L_count, N_count = sel.shape
        total_layers    = L_count
        items_per_layer = N_count   # 8192 for mlp, 2048 for residual
        # "core" is empty for this model; use class_neurons.any(axis=0) as filter
        # — includes any neuron that is owned by at least one class
        class_neurons = sel_data["class_neurons"].astype(bool)  # (C, L, N)
        any_owned = class_neurons.any(axis=0)                   # (L, N)
        SEL_THRESH = 0.4
        for l in range(L_count):
            for n in range(N_count):
                if not any_owned[l, n]:
                    continue
                hid = f"L{l}I{n}"
                for ci, cls in enumerate(classes):
                    if float(sel[ci, l, n]) > SEL_THRESH:
                        if hid not in head_owners:
                            head_owners[hid] = []
                        if cls not in head_owners[hid]:
                            head_owners[hid].append(cls)
        n_multi = sum(1 for v in head_owners.values() if len(v) > 1)
        print(f"  {args.component.upper()} neurons with multiple class owners: "
              f"{n_multi} / {len(head_owners)}")
                            
    # 1. Establish the fixed Grid
    layer_radii = np.linspace(0.55, 1.60, total_layers)
    for l in range(total_layers):
        r = layer_radii[l]
        for i in range(items_per_layer):
            hid = f"L{l}I{i}"
            head_grid.append(hid)
            head_radii[hid] = r

    # ─────────────────────────────────────────────────────────────────────────
    # Dendrogram Bipartite Sorting (Group Visually Similar Indices & ASTs)
    # ─────────────────────────────────────────────────────────────────────────
    active_h_indices = sorted(list(set(int(hid.split("I")[1]) for hid in head_owners.keys())))
    if items_per_layer > 256:
        # For MLP and Residual, only space out the active indices around the circle
        plotted_indices = active_h_indices
    else:
        # For Attention Heads, keep the full grid so inactive ones can be drawn
        plotted_indices = list(range(items_per_layer))
        
    num_plotted = len(plotted_indices)

    # Build binary feature matrix: Rows = Plotted Indices, Cols = AST Classes
    head_idx_features = np.zeros((num_plotted, len(classes)))
    for hid, owners in head_owners.items():
        h_idx = int(hid.split("I")[1])
        if h_idx in plotted_indices:
            row_idx = plotted_indices.index(h_idx)
            for ast_cls in owners:
                if ast_cls in classes:
                    c_idx = classes.index(ast_cls)
                    head_idx_features[row_idx, c_idx] = 1

    # Hierarchical clustering (SciPy)
    if num_plotted > 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            distances = pdist(head_idx_features, metric='jaccard')
            distances = np.nan_to_num(distances, nan=1.0)
        Z = linkage(distances, method='ward')
        optimal_leaf_order = leaves_list(Z)
    else:
        optimal_leaf_order = np.arange(num_plotted)

    # Sort AST Classes (Inner Circle) so similar concepts are bundled together
    valid_c_idx = [i for i in range(len(classes)) if head_idx_features[:, i].sum() > 0]
    if len(valid_c_idx) > 2:
        valid_classes = [classes[i] for i in valid_c_idx]
        valid_ast_f = head_idx_features[:, valid_c_idx].T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast_dist = pdist(valid_ast_f, metric='jaccard')
            ast_dist = np.nan_to_num(ast_dist, nan=1.0)
        Z_ast = linkage(ast_dist, method='ward')
        ast_leaf_order = leaves_list(Z_ast)
        sorted_valid = [valid_classes[i] for i in ast_leaf_order]
        inactive_cls = [cls for i, cls in enumerate(classes) if i not in valid_c_idx]
        classes = sorted_valid + inactive_cls

        # Generate and save a Dendrogram Plot for the AST Classes
        try:
            fig_dendro, ax_dendro = plt.subplots(figsize=(15, 6))
            dendrogram(Z_ast, labels=valid_classes, leaf_rotation=90, leaf_font_size=10)
            ax_dendro.set_title(f"Dendrogram of AST Classes ({args.component.upper()} Overlap)")
            ax_dendro.set_ylabel("Jaccard Distance (Ward)")
            dendro_out = in_dir / "figures" / "head_neuron_bundling" / f"{args.component}_ast_dendrogram.pdf"
            dendro_out.parent.mkdir(parents=True, exist_ok=True)
            fig_dendro.tight_layout()
            fig_dendro.savefig(dendro_out)
            plt.close(fig_dendro)
            print(f"Saved AST Dendrogram to {dendro_out.resolve()}")
        except Exception as e:
            print(f"Could not generate dendrogram: {e}")

    # Use this order to reassign the active head indices to Theta positions
    theta_grid = np.linspace(0, 2 * np.pi, num_plotted, endpoint=False)
    head_index_to_theta = {plotted_indices[optimal_leaf_order[i]]: theta_grid[i] for i in range(num_plotted)}
    
    head_thetas = {}
    for hid in head_grid:
        h_idx = int(hid.split("I")[1])
        if h_idx in head_index_to_theta:
            head_thetas[hid] = head_index_to_theta[h_idx]

    # 3. Inner Circle for AST Classes
    theta_classes = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
    cls_coords = {cls: polar_to_cartesian(0.3, th) for cls, th in zip(classes, theta_classes)}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plotly Interactive Setup
    # ─────────────────────────────────────────────────────────────────────────
    fig = go.Figure()
    
    base_trace_indices = []
    layer_trace_indices = {l: [] for l in range(total_layers)}
    
    # 4. Draw the base concentric rings
    t_circle = np.linspace(0, 2 * np.pi, 200)
    
    # Inner boundary circle
    fig.add_trace(go.Scatter(
        x=0.3 * np.cos(t_circle), y=0.3 * np.sin(t_circle),
        mode='lines', line=dict(color='#dddddd', width=1, dash='dot'),
        hoverinfo='skip', showlegend=False
    ))
    base_trace_indices.append(len(fig.data) - 1)

    # Layer rings
    for l_idx, r in enumerate(layer_radii):
        fig.add_trace(go.Scatter(
            x=r * np.cos(t_circle), y=r * np.sin(t_circle),
            mode='lines', line=dict(color='#eeeeee', width=0.8),
            hoverinfo='skip', showlegend=False
        ))
        layer_trace_indices[l_idx].append(len(fig.data) - 1)
    
    # Inactive items (hoverable background grid) - skipped for MLP/Residual to avoid massive SVG overhead
    if items_per_layer <= 256:
        for l_idx in range(total_layers):
            bg_x, bg_y, bg_text = [], [], []
            for i in range(items_per_layer):
                hid = f"L{l_idx}I{i}"
                if hid not in head_owners:
                    r = head_radii[hid]
                    th = head_thetas[hid]
                    bg_x.append(r * np.cos(th))
                    bg_y.append(r * np.sin(th))
                    lbl = hid.replace('I', 'H') if args.component == 'heads' else hid
                    bg_text.append(f"{lbl} (Inactive)")
            
            if bg_x:
                fig.add_trace(go.Scatter(
                    x=bg_x, y=bg_y,
                    mode='markers', marker=dict(color='#e8e8e8', size=3),
                        text=bg_text, hoverinfo='text', showlegend=False
                ))
                layer_trace_indices[l_idx].append(len(fig.data) - 1)
            
    # 5. Draw Interactive Classes and Roots
    class_node_traces = []
    ast_annotations = []

    # Add dummy trace for 'Show/Hide All' ASTs
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='#cccccc', size=10, symbol='square'),
        name="<b>[ 🔄 Dbl-Click: Show/Hide All ]</b>",
        legendgroup="___SelectAll___",
        showlegend=True
    ))
    base_trace_indices.append(len(fig.data) - 1)

    # Build per-class metadata (label, colour, node dot) — no route drawing yet
    for cls_idx, cls in enumerate(classes):
        th_c = theta_classes[cls_idx]
        cx, cy = cls_coords[cls]
        c_color = SOFT_COLOURS[cls_idx % len(SOFT_COLOURS)]

        # Rotated spoke label
        angle_deg = np.degrees(th_c) % 360
        if 90 < angle_deg <= 270:
            textangle = 180 - angle_deg
        else:
            textangle = -angle_deg
        text_len = len(cls) * 0.0135
        r_center = 0.27 - (text_len / 2)
        lx = r_center * np.cos(th_c)
        ly = r_center * np.sin(th_c)
        ast_annotations.append(dict(
            x=lx, y=ly, text=f"<b>{cls}</b>", showarrow=False,
            textangle=textangle, xanchor='center', yanchor='middle',
            font=dict(color=c_color, size=11, family='Arial'),
            xref='x', yref='y'
        ))

        # Class Node dot — deferred to draw ON TOP of routes
        class_node_traces.append(go.Scatter(
            x=[cx], y=[cy], mode='markers',
            marker=dict(color=c_color, size=14, line=dict(color='white', width=1.5)),
            hoverinfo='text', hovertext=[cls], name=cls, legendgroup=cls, showlegend=True
        ))

    # 5b. Draw routes: outermost layer first so inner layers render on top.
    # For MLP/Residual, default to showing only Layer 0 to keep the page responsive.
    default_l0    = args.component in ('mlp', 'residual')
    natural_op    = {}   # trace_idx → its "full" opacity (for All-Layers button)
    arc_route_idx  = []   # indices of route traces (drawn as arcs by default)
    arc_data_store = {}   # trace_idx → [arc_xs, arc_ys]  (for toggle-back)
    crv_data_store = {}   # trace_idx → [crv_xs, crv_ys]  (swapped in on "Curves")

    # highway_radii used by Bezier curve routing (one per class, linearly spaced)
    highway_radii = np.linspace(0.35, 0.65, len(classes))

    GAP_BEFORE = 0.05
    ARC_STEP   = 0.008

    for l_idx in range(total_layers - 1, -1, -1):   # outermost first
        r_layer  = layer_radii[l_idx]
        r_prev   = layer_radii[l_idx - 1] if l_idx > 0 else 0.30
        visible  = (not default_l0) or (l_idx == 0)   # only L0 shown initially for mlp/residual

        asts_at_layer = []
        for cls_idx, cls in enumerate(classes):
            owned = [hid for hid, owners in head_owners.items()
                     if cls in owners and hid.startswith(f"L{l_idx}I")]
            if owned:
                asts_at_layer.append((cls_idx, cls, owned))
        asts_at_layer.sort(key=lambda x: theta_classes[x[0]] % (2 * np.pi))

        for rank, (cls_idx, cls, owned_heads_l) in enumerate(asts_at_layer):
            r_arc    = max(r_prev + 0.01, r_layer - GAP_BEFORE - rank * ARC_STEP)
            th_root  = theta_classes[cls_idx]
            c_color  = SOFT_COLOURS[cls_idx % len(SOFT_COLOURS)]
            r_hw     = highway_radii[cls_idx % len(highway_radii)]

            node_thetas, node_radii_list = [], []
            active_x, active_y, active_text, tpos_list = [], [], [], []
            valid_hids = []

            for hid in owned_heads_l:
                if hid not in head_thetas or hid not in head_radii:
                    continue
                th_h = head_thetas[hid]
                r_h  = head_radii[hid]
                node_thetas.append(th_h)
                node_radii_list.append(r_h)
                active_x.append(r_h * np.cos(th_h))
                active_y.append(r_h * np.sin(th_h))
                lbl = hid.replace('I', 'H') if args.component == 'heads' else hid
                owners_str = ", ".join(head_owners[hid])
                active_text.append(f"<b>{lbl}</b><br>ASTs: {owners_str}")
                tpos_list.append(get_text_position(th_h, inward=False))
                valid_hids.append(hid)

            if not node_thetas:
                continue

            node_symbols = ['star' if len(head_owners.get(h, [])) == 1 else 'circle'
                            for h in valid_hids]
            node_labels  = [h.replace('I', 'H') if args.component == 'heads' else h
                            for h in valid_hids]

            # ── Route trace (arcs by default; data swapped on toggle) ──────
            arc_x, arc_y = make_layered_route(0.3, th_root, r_arc, node_thetas, node_radii_list)
            init_op_arc  = 0.6 if visible else 0.0
            fig.add_trace(go.Scatter(
                x=arc_x, y=arc_y, mode='lines',
                line=dict(color=c_color, width=1.5),
                opacity=init_op_arc, hoverinfo='skip',
                name=cls, legendgroup=cls, showlegend=False
            ))
            ti = len(fig.data) - 1
            layer_trace_indices[l_idx].append(ti)
            arc_route_idx.append(ti)
            natural_op[ti] = 0.6

            # Pre-compute curve coordinates (stored in JS, NOT added as a trace)
            crv_x, crv_y = [], []
            for th_h, r_h in zip(node_thetas, node_radii_list):
                rx, ry = get_polar_route(r_h, th_h, r_hw, 0.3, th_root)
                crv_x.extend(rx); crv_x.append(None)
                crv_y.extend(ry); crv_y.append(None)
            arc_data_store[ti] = [list(arc_x), list(arc_y)]
            crv_data_store[ti] = [crv_x, crv_y]

            # ── Node markers (shared by both routing styles) ────────────────
            init_op_node = 1.0 if visible else 0.0
            fig.add_trace(go.Scatter(
                x=active_x, y=active_y,
                mode='markers+text',
                marker=dict(
                    color=c_color, size=8,
                    symbol=node_symbols,
                    line=dict(color='white', width=0.5)
                ),
                text=node_labels,
                textposition=tpos_list,
                textfont=dict(color=c_color, size=9),
                opacity=init_op_node,
                hoverinfo='text', hovertext=active_text,
                name=cls, legendgroup=cls, showlegend=False
            ))
            ti = len(fig.data) - 1
            layer_trace_indices[l_idx].append(ti)
            natural_op[ti] = 1.0

    # Draw deferred Class Nodes so they sit on top of the root lines
    for tr in class_node_traces:
        fig.add_trace(tr)
        base_trace_indices.append(len(fig.data) - 1)

    # 6. Configure Dropdown & Layout
    total_traces = len(fig.data)
    buttons = []
    
    all_layer_traces = []
    for l_idx in range(total_layers):
        all_layer_traces.extend(layer_trace_indices[l_idx])

    # Use natural_op (not initial opacity, which may be 0 for default-layer-0 mode)
    # so that "All Layers" restores full opacity regardless of starting state.
    original_opacities = []
    original_hoverinfos = []
    for idx in all_layer_traces:
        tr = fig.data[idx]
        original_opacities.append(natural_op.get(idx, 1.0))
        original_hoverinfos.append(tr.hoverinfo if tr.hoverinfo is not None else 'all')

    # 'All Layers' Button — show all traces
    buttons.append(dict(
        label="All Layers",
        method="restyle",
        args=[
            {"opacity": original_opacities, "hoverinfo": original_hoverinfos},
            all_layer_traces,
        ]
    ))

    for l_idx in range(total_layers):
        opacities = [0.0] * len(all_layer_traces)
        hinfos = ['skip'] * len(all_layer_traces)
        for i, tr_idx in enumerate(all_layer_traces):
            if tr_idx in layer_trace_indices[l_idx]:
                opacities[i] = original_opacities[i]
                hinfos[i] = original_hoverinfos[i]

        buttons.append(dict(
            label=f"Layer {l_idx}",
            method="restyle",
            args=[
                {"opacity": opacities, "hoverinfo": hinfos},
                all_layer_traces,
            ]
        ))

    comp_title  = "MLP" if args.component == 'mlp' else args.component.capitalize()
    layer_active = 1 if default_l0 else 0   # default to "Layer 0" for mlp/residual

    fig.update_layout(
        updatemenus=[
            dict(active=layer_active, buttons=buttons,
                 x=1.02, y=1.0, xanchor="left", yanchor="top",
                 bgcolor='white', bordercolor='#dddddd'),
        ],
        annotations=ast_annotations,
        title=dict(text=f"Ontology: {comp_title} AST Roots", font=dict(size=22), x=0.5, y=0.98),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(visible=False, range=[-1.65, 1.65], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-1.65, 1.65], constrain='domain'),
        width=1600, height=1600,
        margin=dict(l=40, r=200, t=90, b=40),
        showlegend=True,
        legend=dict(
            x=1.02, y=0.5, 
            xanchor="left", yanchor="middle", 
            title=dict(text="AST Classes<br>(Double-click<br>to isolate)")
        )
    )
    
    out_file = in_dir / "figures" / "head_neuron_bundling" / f"{args.component}_edge_bundling_interactive.{args.format}"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "html":
        import json as _json
        arc_js = _json.dumps({str(k): v for k, v in arc_data_store.items()})
        crv_js = _json.dumps({str(k): v for k, v in crv_data_store.items()})
        js_code = f"""
        var plot = document.getElementsByClassName('plotly-graph-div')[0];

        // ── coordinate stores (swapped on toggle, no hidden traces) ─────────
        var _arcData = {arc_js};
        var _crvData = {crv_js};
        var _usingCurves = false;

        function _swapRoutes(useCurves) {{
            _usingCurves = useCurves;
            var store = useCurves ? _crvData : _arcData;
            var indices = Object.keys(store).map(Number);
            var xs = indices.map(function(i) {{ return store[i][0]; }});
            var ys = indices.map(function(i) {{ return store[i][1]; }});
            Plotly.restyle(plot, {{x: xs, y: ys}}, indices);
        }}

        // ── inject toggle buttons above the plot ────────────────────────────
        (function() {{
            var div = document.createElement('div');
            div.style.cssText = 'position:absolute;top:6px;left:10px;z-index:999;';
            ['Arcs','Curves'].forEach(function(label, i) {{
                var btn = document.createElement('button');
                btn.textContent = label;
                btn.style.cssText = 'margin-right:4px;padding:4px 10px;cursor:pointer;'
                    + 'border:1px solid #aaa;border-radius:3px;background:#fff;font-size:13px;';
                btn.onclick = function() {{
                    _swapRoutes(i === 1);
                    div.querySelectorAll('button').forEach(function(b,j){{
                        b.style.background = (j===i) ? '#ddeeff' : '#fff';
                    }});
                }};
                if (i === 0) btn.style.background = '#ddeeff';
                div.appendChild(btn);
            }});
            plot.parentElement.style.position = 'relative';
            plot.parentElement.appendChild(div);
        }})();

        // ── annotation sync (legend toggle) ─────────────────────────────────
        plot.on('plotly_restyle', function(eventData) {{
            var annotations = plot.layout.annotations;
            if (!annotations) return;
            var update = {{}};
            var hasUpdate = false;
            for (var i = 0; i < annotations.length; i++) {{
                if (!annotations[i].text) continue;
                var annText = annotations[i].text.replace(/<[^>]*>?/gm, '');
                var isVisible = true;
                for (var j = 0; j < plot.data.length; j++) {{
                    if (plot.data[j].name === annText && plot.data[j].legendgroup === annText
                            && plot.data[j].showlegend === true) {{
                        isVisible = (plot.data[j].visible !== 'legendonly' && plot.data[j].visible !== false);
                        break;
                    }}
                }}
                if (annotations[i].visible !== isVisible) {{
                    update['annotations[' + i + '].visible'] = isVisible;
                    annotations[i].visible = isVisible;
                    hasUpdate = true;
                }}
            }}
            if (hasUpdate) Plotly.relayout(plot, update);
        }});
        """
        fig.write_html(str(out_file), include_plotlyjs=True, post_script=js_code)
    else:
        fig.write_image(str(out_file), scale=2)
        
    print(f"Saved {out_file.resolve()}")

if __name__ == "__main__":
    main()