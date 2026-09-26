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
        path = in_dir / f"02g_{args.stem}_class_neurons.json"
        if not path.exists():
            print(f"Missing required JSON file at {path}.")
            return
        with open(path) as f:
            data = json.load(f)
        # 02g structure: data["residual"][cls]["per_layer"]["L00"] = [int, ...]
        #                data["neurons"][cls]["per_layer"]["L00"] = [int, ...]
        comp_key = "neurons" if args.component == "mlp" else "residual"
        classes = sorted(data.get("classes", []))
        total_layers = 8 if args.component == "mlp" else 9
        items_per_layer = 8192 if args.component == "mlp" else 2048

        section = data.get(comp_key, {})
        for cls in classes:
            per_layer = section.get(cls, {}).get("per_layer", {})
            for l_str, i_list in per_layer.items():
                # l_str is "L00", "L01" etc.
                layer_int = int(l_str.lstrip("L"))
                for raw_idx in i_list[:args.top_k]:
                    hid = f"L{layer_int}I{int(raw_idx)}"
                    if hid not in head_owners:
                        head_owners[hid] = []
                    if cls not in head_owners[hid]:
                        head_owners[hid].append(cls)
                            
    # 1. Establish the fixed Grid
    layer_radii = np.linspace(0.7, 1.4, total_layers)
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
    highway_radii = np.linspace(0.35, 0.65, len(classes))
    class_node_traces = []
    ast_annotations = []  # unused; kept for layout compat

    # Add dummy trace for 'Show/Hide All' ASTs
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='#cccccc', size=10, symbol='square'),
        name="<b>[ 🔄 Dbl-Click: Show/Hide All ]</b>",
        legendgroup="___SelectAll___",
        showlegend=True
    ))
    base_trace_indices.append(len(fig.data) - 1)

    for cls_idx, cls in enumerate(classes):
        th_c = theta_classes[cls_idx]
        cx, cy = cls_coords[cls]
        c_color = SOFT_COLOURS[cls_idx % len(SOFT_COLOURS)]
        r_highway = highway_radii[cls_idx]
        
        # Rotated label along the spoke — textangle so text reads from center
        # outward toward the root dot. Flip left-half to avoid upside-down text.
        # Justify labels so they end exactly at a fixed radius (r=0.27) from the roots
        angle_deg = np.degrees(th_c) % 360
        if 90 < angle_deg <= 270:
            textangle = 180 - angle_deg
        else:
            textangle = - angle_deg
            
        # Estimate text length in data units (~0.0135 units per char at this scale)
        # and anchor at the geometric center of the text to prevent bounding box 
        # rotation from squashing the circle into an ellipse.
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

        # Class Node dot — deferred to draw ON TOP of roots
        class_node_traces.append(go.Scatter(
            x=[cx], y=[cy], mode='markers',
            marker=dict(color=c_color, size=14, line=dict(color='white', width=1.5)),
            hoverinfo='text', hovertext=[cls], name=cls, legendgroup=cls, showlegend=True
        ))
        
        # Draw Roots for this class split by layer
        for l_idx in range(total_layers):
            owned_heads_l = [hid for hid, owners in head_owners.items() if cls in owners and hid.startswith(f"L{l_idx}I")]
            if not owned_heads_l:
                continue
                
            route_x, route_y = [], []
            active_x, active_y, active_text = [], [], []
            tpos_list = []
            
            for hid in owned_heads_l:
                if hid not in head_thetas or hid not in head_radii:
                    continue
                th_h = head_thetas[hid]
                r_h = head_radii[hid]
                
                rx, ry = get_polar_route(r_h - 0.01, th_h, r_highway, 0.3, th_c)
                route_x.extend(rx)
                route_x.append(None)
                route_y.extend(ry)
                route_y.append(None)
                
                active_x.append(r_h * np.cos(th_h))
                active_y.append(r_h * np.sin(th_h))
                lbl = hid.replace('I', 'H') if args.component == 'heads' else hid
                owners_str = ", ".join(head_owners[hid])
                active_text.append(f"<b>{lbl}</b><br>ASTs: {owners_str}")
                tpos_list.append(get_text_position(th_h, inward=False))
                
            # Active Routes Trace
            fig.add_trace(go.Scatter(
                x=route_x, y=route_y, mode='lines',
                line=dict(color=c_color, width=1.5),
                opacity=0.6, hoverinfo='skip',
                name=cls, legendgroup=cls, showlegend=False
            ))
            layer_trace_indices[l_idx].append(len(fig.data) - 1)
            
            # Active Head Nodes (Now with Light-Up Text)
            fig.add_trace(go.Scatter(
                x=active_x, y=active_y, 
                mode='markers+text',
                marker=dict(color=c_color, size=8, line=dict(color='white', width=0.5)),
                text=[h.replace('I', 'H') if args.component == 'heads' else h for h in owned_heads_l],
                textposition=tpos_list,
                textfont=dict(color=c_color, size=9),
                hoverinfo='text',
                hovertext=active_text,
                name=cls, 
                legendgroup=cls, 
                showlegend=False
            ))
            layer_trace_indices[l_idx].append(len(fig.data) - 1)

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

    # We use opacity instead of visible so that the legend toggling works independently
    original_opacities = []
    original_hoverinfos = []
    for idx in all_layer_traces:
        tr = fig.data[idx]
        original_opacities.append(tr.opacity if tr.opacity is not None else 1.0)
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

    comp_title = "MLP" if args.component == 'mlp' else args.component.capitalize()

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.02, y=1.0,
            xanchor="left", yanchor="top",
            bgcolor='white', bordercolor='#dddddd'
        )],
        annotations=ast_annotations,
        title=dict(text=f"Ontology: {comp_title} AST Roots", font=dict(size=22), x=0.5, y=0.98),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(visible=False, range=[-1.45, 1.45], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-1.45, 1.45], constrain='domain'),
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
        js_code = """
        var plot = document.getElementsByClassName('plotly-graph-div')[0];
        plot.on('plotly_restyle', function(eventData) {
            var annotations = plot.layout.annotations;
            if (!annotations) return;
            var update = {};
            var hasUpdate = false;
            for (var i = 0; i < annotations.length; i++) {
                if (!annotations[i].text) continue;
                var annText = annotations[i].text.replace(/<[^>]*>?/gm, '');
                var isVisible = true;
                for (var j = 0; j < plot.data.length; j++) {
                    if (plot.data[j].name === annText && plot.data[j].legendgroup === annText) {
                        isVisible = (plot.data[j].visible !== 'legendonly' && plot.data[j].visible !== false);
                        break;
                    }
                }
                if (annotations[i].visible !== isVisible) {
                    update['annotations[' + i + '].visible'] = isVisible;
                    annotations[i].visible = isVisible;
                    hasUpdate = true;
                }
            }
            if (hasUpdate) Plotly.relayout(plot, update);
        });
        """
        fig.write_html(str(out_file), include_plotlyjs=True, post_script=js_code)
    else:
        fig.write_image(str(out_file), scale=2)
        
    print(f"Saved {out_file.resolve()}")

if __name__ == "__main__":
    main()