"""
02n_circuit_bundling.py  (lives in results/)

Generates a Radial Root plot to visualize which AST classes own which 
Attention Heads. The entire capacity of the model's Attention Heads 
(all layers and indices) forms the fixed outer unit circle. 
When a head is owned by an AST Class, a root grows inwards from the 
perimeter towards the class node in the center.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

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

def main():
    parser = argparse.ArgumentParser(description="Radial Roots: Inward Edge Bundling for Attention Heads")
    parser.add_argument("--stem", default="contrastive_stubs")
    parser.add_argument("--in_dir", default="../results/output")
    parser.add_argument("--top_k", type=int, default=10, help="Top items per class/layer")
    parser.add_argument("--format", default="html", help="Output format (html recommended for interactive Plotly)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    in_dir = Path(__file__).parent / args.in_dir
    
    # Load JSON data
    heads_path = in_dir / f"02l_{args.stem}_head_ownership.json"
    
    if not heads_path.exists():
        print(f"Missing required JSON file at {heads_path}.")
        return

    with open(heads_path) as f:
        head_data = json.load(f)
        
    classes = sorted(list(head_data.keys()))
    
    # 1. Establish the fixed Head Grid (the base unit of the circle)
    # Based on the OpenAI Sparse architecture in the README
    total_layers = 8
    heads_per_layer = 128
    
    head_thetas = {}
    head_radii = {}
    
    # Map 8 layers from inner r=0.7 to outer r=1.4
    layer_radii = np.linspace(0.7, 1.4, total_layers)
    theta_grid = np.linspace(0, 2 * np.pi, heads_per_layer, endpoint=False)
    
    head_grid = []
    for l in range(total_layers):
        r = layer_radii[l]
        for h in range(heads_per_layer):
            hid = f"L{l}H{h}"
            head_grid.append(hid)
            head_thetas[hid] = theta_grid[h]
            head_radii[hid] = r
    
    # 2. Map active Heads to their owning classes
    head_owners = {}
    for cls in classes:
        if cls in head_data:
            for l_str, h_list in head_data[cls].items():
                layer_int = int(l_str.replace("L", ""))
                for h_dict in h_list[:args.top_k]:
                    hid = f"L{layer_int}H{h_dict['head']}"
                    if hid not in head_owners:
                        head_owners[hid] = []
                    if cls not in head_owners[hid]:
                        head_owners[hid].append(cls)
                        
    # 3. Inner Circle for AST Classes
    theta_classes = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
    cls_coords = {cls: polar_to_cartesian(0.3, th) for cls, th in zip(classes, theta_classes)}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plotly Interactive Setup
    # ─────────────────────────────────────────────────────────────────────────
    fig = go.Figure()
    
    base_trace_indices = []
    class_trace_indices = {cls: [] for cls in classes}
    
    # 4. Draw the base concentric rings and radial spokes
    t_circle = np.linspace(0, 2 * np.pi, 200)
    
    # Inner boundary circle
    fig.add_trace(go.Scatter(
        x=0.3 * np.cos(t_circle), y=0.3 * np.sin(t_circle),
        mode='lines', line=dict(color='#dddddd', width=1, dash='dot'),
        hoverinfo='skip', showlegend=False
    ))
    base_trace_indices.append(len(fig.data) - 1)

    # Layer rings
    for r in layer_radii:
        fig.add_trace(go.Scatter(
            x=r * np.cos(t_circle), y=r * np.sin(t_circle),
            mode='lines', line=dict(color='#eeeeee', width=0.8),
            hoverinfo='skip', showlegend=False
        ))
        base_trace_indices.append(len(fig.data) - 1)
        
    # Faint Radial Spokes (joining all layers for the same head index)
    spoke_x, spoke_y = [], []
    for th in theta_grid:
        spoke_x.extend([0.7 * np.cos(th), 1.45 * np.cos(th), None])
        spoke_y.extend([0.7 * np.sin(th), 1.45 * np.sin(th), None])
        
    fig.add_trace(go.Scatter(
        x=spoke_x, y=spoke_y,
        mode='lines', line=dict(color='#f5f5f5', width=0.5),
        hoverinfo='skip', showlegend=False
    ))
    base_trace_indices.append(len(fig.data) - 1)
    
    # Outer Axis Labels (Head Indices)
    label_x = [1.52 * np.cos(th) for th in theta_grid]
    label_y = [1.52 * np.sin(th) for th in theta_grid]
    label_text = [f"H{h}" for h in range(heads_per_layer)]
    
    fig.add_trace(go.Scatter(
        x=label_x, y=label_y,
        mode='text', text=label_text,
        textfont=dict(size=7, color='#aaaaaa'),
        hoverinfo='skip', showlegend=False
    ))
    base_trace_indices.append(len(fig.data) - 1)
    
    # Inactive heads (hoverable background grid)
    bg_x, bg_y, bg_text = [], [], []
    for hid in head_grid:
        if hid not in head_owners:
            r = head_radii[hid]
            th = head_thetas[hid]
            bg_x.append(r * np.cos(th))
            bg_y.append(r * np.sin(th))
            bg_text.append(f"{hid} (Inactive)")
            
    fig.add_trace(go.Scatter(
        x=bg_x, y=bg_y,
        mode='markers', marker=dict(color='#e8e8e8', size=3),
        text=bg_text, hoverinfo='text', showlegend=False
    ))
    base_trace_indices.append(len(fig.data) - 1)
            
    # 5. Draw Interactive Classes and Roots
    highway_radii = np.linspace(0.35, 0.65, len(classes))
    for cls_idx, cls in enumerate(classes):
        th_c = theta_classes[cls_idx]
        cx, cy = cls_coords[cls]
        c_color = SOFT_COLOURS[cls_idx % len(SOFT_COLOURS)]
        r_highway = highway_radii[cls_idx]
        
        # Highway Ring Arc for this class
        hx = r_highway * np.cos(t_circle)
        hy = r_highway * np.sin(t_circle)
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode='lines',
            line=dict(color=c_color, width=1, dash='solid'),
            opacity=0.15, hoverinfo='skip',
            name=cls, legendgroup=cls, showlegend=False
        ))
        class_trace_indices[cls].append(len(fig.data) - 1)
        
        # Class Node
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers+text',
            marker=dict(color=c_color, size=18, line=dict(color='white', width=1.5)),
            text=[cls], textposition='bottom center',
            textfont=dict(color=c_color, size=11, family='Arial Black'),
            hoverinfo='text', name=cls, legendgroup=cls, showlegend=True
        ))
        class_trace_indices[cls].append(len(fig.data) - 1)
        
        # Draw Roots for this class
        owned_heads = [hid for hid, owners in head_owners.items() if cls in owners]
        route_x, route_y = [], []
        active_x, active_y, active_text = [], [], []
        
        for hid in owned_heads:
            th_h = head_thetas[hid]
            r_h = head_radii[hid]
            
            rx, ry = get_polar_route(r_h - 0.01, th_h, r_highway, 0.3, th_c)
            route_x.extend(rx)
            route_x.append(None)
            route_y.extend(ry)
            route_y.append(None)
            
            active_x.append(r_h * np.cos(th_h))
            active_y.append(r_h * np.sin(th_h))
            active_text.append(f"<b>{hid}</b><br>Owned by: {cls}")
            
        # Active Routes Trace
        fig.add_trace(go.Scatter(
            x=route_x, y=route_y, mode='lines',
            line=dict(color=c_color, width=1.5),
            opacity=0.6, hoverinfo='skip',
            name=cls, legendgroup=cls, showlegend=False
        ))
        class_trace_indices[cls].append(len(fig.data) - 1)
        
        # Active Head Nodes
        fig.add_trace(go.Scatter(
            x=active_x, y=active_y, mode='markers',
            marker=dict(color=c_color, size=8, line=dict(color='white', width=0.5)),
            text=active_text, hoverinfo='text',
            name=cls, legendgroup=cls, showlegend=False
        ))
        class_trace_indices[cls].append(len(fig.data) - 1)

    # 6. Configure Dropdown & Layout
    total_traces = len(fig.data)
    buttons = []
    
    # 'All Classes' Button
    buttons.append(dict(
        label="All Classes",
        method="restyle",
        args=["visible", [True] * total_traces]
    ))
    
    # Individual Class Filters
    for cls in classes:
        visibility = [False] * total_traces
        for idx in base_trace_indices:
            visibility[idx] = True
        for idx in class_trace_indices[cls]:
            visibility[idx] = True
            
        buttons.append(dict(
            label=cls,
            method="restyle",
            args=["visible", visibility]
        ))

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.1, y=1.0,
            xanchor="right", yanchor="top",
            bgcolor='white', bordercolor='#dddddd'
        )],
        title=dict(text="Attention Head Ontology: AST Class Roots", font=dict(size=22), x=0.5),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6]),
        width=1200, height=1200,
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=True,
        legend=dict(x=1.1, y=0.5, xanchor="right", yanchor="middle")
    )
    
    out_file = in_dir / "figures" / "head_neuron_bundling" / f"edge_bundling.{args.format}"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "html":
        fig.write_html(str(out_file))
    else:
        fig.write_image(str(out_file), scale=2)
        
    print(f"Saved {out_file.resolve()}")

if __name__ == "__main__":
    main()