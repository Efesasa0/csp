"""
02d_neuron_activation_grid.py  (lives in results/)

Reads neuron_ids JSON files (from 02b / 02c) and renders one figure per
(qty, section, class/Overall):

  figures/neuron_grids/<qty>_<section>/<ClassName>.png
  ──────────────────────────────────────────────────────
  Layers are arranged in a grid of N_LAYER_COLS columns (default 3).
  Each cell shows two panels stacked vertically:

    ┌─────────────────────────┐   ← activation heatmap
    │  viridis_r  mean_act    │     H-dim vector reshaped to GRID_SHAPE
    │  white = no activation  │     (32×64 for residual, 128×64 for MLP)
    ├─────────────────────────┤
    │  purity strip           │   ← thin binary: blue = pure, white = not pure
    └─────────────────────────┘

  Colour scale (vmax) is shared across all classes within a section so
  activation magnitudes are directly comparable across plots.

GRID_SHAPE
  residual : 2048  →  32 × 64
  mlp      : 8192  →  128 × 64

INPUTS
──────
  <in_dir>/neuron_ids/<qty>_L<ll>_neuron_ids.json

OUTPUTS  (default: <in_dir>/figures/neuron_grids/<qty>_<section>/)
────────
  Overall.png
  <ClassName>.png   (one per class)

Usage
-----
  python 02d_neuron_activation_grid.py
  python 02d_neuron_activation_grid.py --qty residual --layer_cols 4
  python 02d_neuron_activation_grid.py --qty mlp --dpi 250
  python 02d_neuron_activation_grid.py --layer 3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ─────────────────────────────────────────────────────────────────────────────
# Layout / size constants
# ─────────────────────────────────────────────────────────────────────────────

# How to reshape each qty's H-dim vector into a 2-D image
GRID_SHAPE: dict[str, tuple[int, int]] = {
    "residual": (32, 64),    # 2048  = 32 × 64
    "mlp":      (128, 64),   # 8192  = 128 × 64
}

# Cell dimensions in inches (width, height) for the activation heatmap.
# Aspect matches GRID_SHAPE so pixels look square.
CELL_SIZE: dict[str, tuple[float, float]] = {
    "residual": (4.8, 2.4),   # 64/32 = 2 → wide
    "mlp":      (2.4, 4.8),   # 64/128 = 0.5 → tall
}

# Purity strip height as a fraction of the heatmap cell height
PURE_STRIP_RATIO = 0.08

# A neuron is "class-specifically pure" if its mean activation for the class
# is at least this many times its overall mean activation across all classes.
CLASS_SELECTIVITY_RATIO = 1.5

# Left margin (small, no row labels now), top margin for titles, right for cbar
LEFT_IN  = 0.20
TOP_IN   = 0.75
CBAR_IN  = 1.10

# Default DPI (used for PNG; PDF is vector so DPI only affects rasterised elements)
DPI_DEFAULT = 300

# Default number of layer columns per figure
N_LAYER_COLS_DEFAULT = 3

# Font sizes (points) — sized for LaTeX inclusion at ~\columnwidth
FS_TITLE    = 11   # per-cell layer title
FS_ANNOT    =  7   # top-5 neuron annotations
FS_CBAR     = 11   # colourbar label
FS_CBAR_TK  =  9   # colourbar tick labels
FS_LEGEND   =  9   # legend text
FS_SUPTITLE = 13   # figure super-title


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stem",        default="contrastive_stubs")
    p.add_argument("--in_dir",  "-d", default="output",
                   help="Dir containing neuron_ids/ (relative to this script)")
    p.add_argument("--out_dir", "-o", default=None,
                   help="Output root (default: same as --in_dir)")
    p.add_argument("--qty",     choices=["residual", "mlp", "both"],
                   default="both")
    p.add_argument("--layer",   type=int, default=None,
                   help="Include only this layer index")
    p.add_argument("--layer_cols", type=int, default=N_LAYER_COLS_DEFAULT,
                   help=f"Layers per row in figure (default {N_LAYER_COLS_DEFAULT})")
    p.add_argument("--dpi",     type=int, default=DPI_DEFAULT)
    p.add_argument("--format",  choices=["png", "pdf"], default="pdf",
                   help="Output format: pdf (vector, LaTeX-ready) or png (raster)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Array helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_act_image(
    rows:       list[dict],
    act_key:    str,
    H:          int,
    grid_shape: tuple[int, int],
) -> np.ma.MaskedArray:
    """
    H-dim activation vector → 2-D masked array.
    Neurons absent from the VP set remain masked (shown as white background).
    """
    img  = np.zeros(H, dtype=np.float32)
    seen = np.zeros(H, dtype=bool)
    for r in rows:
        v = r.get(act_key)
        if v is not None and v > 0:
            nidx = r["neuron"]
            if 0 <= nidx < H:
                img[nidx]  = float(v)
                seen[nidx] = True
    return np.ma.masked_where(~seen.reshape(grid_shape), img.reshape(grid_shape))


def make_pure_image(
    rows:       list[dict],
    H:          int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Binary float32 image: 1.0 where pure, 0.0 elsewhere."""
    mask = np.zeros(H, dtype=np.float32)
    for r in rows:
        if r.get("pure"):
            nidx = r["neuron"]
            if 0 <= nidx < H:
                mask[nidx] = 1.0
    return mask.reshape(grid_shape)



# ─────────────────────────────────────────────────────────────────────────────
# Per-class figure  (layers in a grid, each cell = heatmap + purity strip)
# ─────────────────────────────────────────────────────────────────────────────

_PURE_CMAP = ListedColormap(["white", "#2196F3"])   # 0=white, 1=blue


def plot_class_layers(
    layer_data:   list[dict],
    section:      str,
    qty:          str,
    col_name:     str,          # "Overall"  or  a class name
    save_path:    Path,
    dpi:          int,
    n_layer_cols: int = N_LAYER_COLS_DEFAULT,
    fmt:          str = "pdf",
) -> None:
    """
    One figure: layers tiled in a grid of n_layer_cols columns.
    Each tile = [activation heatmap] / [thin purity strip].
    """
    grid_shape = GRID_SHAPE[qty]
    H          = grid_shape[0] * grid_shape[1]
    cell_w, cell_h = CELL_SIZE[qty]
    pure_h     = cell_h * PURE_STRIP_RATIO

    n_layers  = len(layer_data)
    n_cols    = min(n_layer_cols, n_layers)
    n_rows    = math.ceil(n_layers / n_cols)

    # ── Figure geometry ───────────────────────────────────────────────────────
    content_w = cell_w * n_cols
    content_h = (cell_h + pure_h) * n_rows
    fig_w     = LEFT_IN + content_w + CBAR_IN
    fig_h     = TOP_IN  + content_h

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Outer GridSpec: n_rows × n_cols blocks
    outer = GridSpec(
        n_rows, n_cols,
        figure = fig,
        left   = LEFT_IN  / fig_w,
        right  = 1.0 - CBAR_IN / fig_w,
        top    = 1.0 - TOP_IN  / fig_h,
        bottom = 0.02,
        wspace = 0.06,
        hspace = 0.08,
    )

    cmap     = plt.get_cmap("viridis_r").copy()  # reversed: purple=high, yellow=low
    cmap.set_bad("white")                         # masked pixels → white
    is_overall = (col_name == "Overall")
    act_key    = "overall_mean_act" if is_overall else "mean_act"

    heat_axes: list[plt.Axes] = []

    # Local function to compute per-class vmax to avoid one outlier dominating
    def _panel_vmax(rows: list[dict]) -> float:
        vals = [r.get(act_key) for r in rows if r.get(act_key) is not None and r.get(act_key) > 0]
        return float(max(vals)) if vals else 1.0

    for li, d in enumerate(layer_data):
        ri = li // n_cols
        ci = li % n_cols
        layer = d["layer"]

        sec  = d[section]
        rows = sec["overall_ranking"] if is_overall \
               else sec["by_class"].get(col_name, [])

        # Build neuron → overall_mean_act lookup for selectivity test
        overall_act: dict[int, float] = {
            r["neuron"]: float(r.get("overall_mean_act") or 0.0)
            for r in sec.get("overall_ranking", [])
        }

        # Class-specific purity: active for this class AND
        # mean_act >= CLASS_SELECTIVITY_RATIO × overall_mean_act
        def _class_pure(r: dict) -> bool:
            if is_overall:
                return bool(r.get("pure"))   # Overall view keeps global flag
            v = float(r.get("mean_act") or 0.0)
            if v <= 0:
                return False
            overall = overall_act.get(r["neuron"], 0.0)
            if overall <= 0:
                return v > 0   # no baseline → active counts as selective
            return v / overall >= CLASS_SELECTIVITY_RATIO

        # Inner sub-grid: [heatmap (tall)] / [purity strip (thin)]
        inner = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec   = outer[ri, ci],
            height_ratios  = [1.0, PURE_STRIP_RATIO],
            hspace         = 0.03,
        )
        ax_h = fig.add_subplot(inner[0])   # heatmap
        ax_p = fig.add_subplot(inner[1])   # purity strip
        heat_axes.append(ax_h)

        # ── Activation heatmap ────────────────────────────────────────────────
        act_img = make_act_image(rows, act_key, H, grid_shape)
        panel_vmax = _panel_vmax(rows)
        panel_norm = mcolors.Normalize(vmin=0.0, vmax=panel_vmax)
        ax_h.imshow(act_img, cmap=cmap, norm=panel_norm,
                    aspect="auto", interpolation="nearest", origin="upper")
        if qty == "mlp":
            for spine in ax_h.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#444444")
                spine.set_linewidth(0.8)
        else:
            ax_h.spines[:].set_visible(False)

        if li == 0:
            # First cell only: show corner neuron indices so the reader can
            # verify the layout is row-major (left→right, then down).
            nrows_g, ncols_g_ = grid_shape
            ax_h.set_xticks([0, ncols_g_ - 1])
            ax_h.set_xticklabels(["n=0", f"n={ncols_g_-1}"],
                                  fontsize=FS_ANNOT - 1, color="#444")
            ax_h.tick_params(axis="x", which="both", length=2, pad=1,
                             bottom=True, top=False, labelbottom=False, labeltop=True)
            ax_h.set_yticks([0, nrows_g - 1])
            ax_h.set_yticklabels(["", f"n={(nrows_g-1)*ncols_g_}"],
                                  fontsize=FS_ANNOT - 1, color="#444")
            ax_h.tick_params(axis="y", which="both", length=2, pad=1)
            ax_h.text(ncols_g_ - 1, -1.5, "→", ha="right", va="bottom",
                      fontsize=FS_ANNOT, color="#444",
                      transform=ax_h.transData)
            ax_h.text(-1.5, nrows_g - 1, "↓", ha="right", va="bottom",
                      fontsize=FS_ANNOT, color="#444",
                      transform=ax_h.transData)
        else:
            ax_h.set_xticks([])
            ax_h.set_yticks([])

        # Title: layer + per-panel scale range + numActive + numPure
        n_active = sum(1 for r in rows if (r.get(act_key) or 0) > 0)
        n_pure   = sum(1 for r in rows if _class_pure(r))
        ax_h.set_title(
            f"L{layer}  [0–{panel_vmax:.1f}]  {n_active} active · {n_pure} pure",
            fontsize=FS_TITLE if qty != "mlp" else FS_TITLE - 3, pad=4)

        # Annotate top-5 most active neurons with their IDs + purity flag.
        top5 = sorted(rows, key=lambda r: r.get(act_key) or 0, reverse=True)[:5]
        ncols_g = grid_shape[1]

        # Build list sorted by neuron column (gx) so x-pass is in order
        annots = []
        for r in top5:
            nidx = r["neuron"]
            gy   = nidx // ncols_g
            gx   = nidx % ncols_g
            val  = r.get(act_key) or 0.0
            pure_str = "(pure)" if _class_pure(r) else ""
            nid = r['id'].replace("Residual", "RES")
            annots.append((gx, gy, f"{nid} {pure_str}\n{val:.3f}"))
        annots.sort(key=lambda a: a[0])

        # Place labels as close as possible to their neuron, then use
        # iterative 2D repulsion (in axes-fraction space) to separate any
        # that overlap, clamping the result to stay inside the heatmap.
        nrows_g_ = grid_shape[0]
        LBOX_W   = 0.20    # estimated label width  (axes fraction)
        LBOX_H   = 0.14    # estimated label height (axes fraction)
        ARROW_R  = 0.12    # minimum arrow length (label must be this far from neuron)
        MX       = LBOX_W / 2 + 0.01
        MY_BOT   = LBOX_H / 2 + 0.03
        MY_TOP   = 1.0 - LBOX_H / 2 - 0.02

        # Neuron positions in axes fraction
        neuron_pos = []
        for gx, gy, _ in annots:
            xf = (gx + 0.5) / ncols_g
            yf = 1.0 - (gy + 0.5) / nrows_g_   # origin="upper": gy=0 → yf=1
            neuron_pos.append((xf, yf))

        # Initial label positions: start ARROW_R above each neuron
        pos = []
        for xf, yf in neuron_pos:
            lx = max(MX, min(1 - MX, xf))
            ly = max(MY_BOT, min(MY_TOP, yf + ARROW_R))
            pos.append([lx, ly])

        # Iterative repulsion between labels AND repulsion from their own neuron
        for _ in range(120):
            for i in range(len(pos)):
                # Repel from other labels
                for j in range(i + 1, len(pos)):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    ox = LBOX_W - abs(dx)
                    oy = LBOX_H - abs(dy)
                    if ox > 0 and oy > 0:
                        if ox <= oy:
                            push = ox / 2 + 0.003
                            sign = 1 if dx >= 0 else -1
                            pos[i][0] += sign * push
                            pos[j][0] -= sign * push
                        else:
                            push = oy / 2 + 0.003
                            sign = 1 if dy >= 0 else -1
                            pos[i][1] += sign * push
                            pos[j][1] -= sign * push

                # Repel from own neuron — keep label at least ARROW_R away
                nx, ny = neuron_pos[i]
                dx = pos[i][0] - nx
                dy = pos[i][1] - ny
                dist = (dx**2 + dy**2) ** 0.5
                if dist < ARROW_R:
                    if dist < 1e-6:
                        dx, dy, dist = 0.0, 1.0, 1.0  # push straight up
                    scale = (ARROW_R - dist) / dist
                    pos[i][0] += dx * scale
                    pos[i][1] += dy * scale

            # Clamp after each iteration
            for p in pos:
                p[0] = max(MX, min(1 - MX, p[0]))
                p[1] = max(MY_BOT, min(MY_TOP, p[1]))

        for (gx, gy, label), (text_x, text_y) in zip(annots, pos):
            ax_h.annotate(
                label,
                xy         = (gx, gy),
                xycoords   = "data",
                xytext     = (text_x, text_y),
                textcoords = "axes fraction",
                fontsize   = FS_ANNOT,
                color      = "black",
                ha         = "center", va = "center",
                bbox       = dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="grey", alpha=0.85),
                arrowprops = dict(arrowstyle="-", color="black",
                                  lw=0.4, alpha=0.6),
                annotation_clip = True,
                zorder     = 5,
            )

        # ── Purity strip (class-specific selectivity) ─────────────────────────
        pure_vec = np.zeros(H, dtype=np.float32)
        for r in rows:
            nidx = r["neuron"]
            if 0 <= nidx < H and _class_pure(r):
                pure_vec[nidx] = 1.0
        pure_img = pure_vec.reshape(grid_shape)
        ax_p.imshow(pure_img,
                    cmap   = _PURE_CMAP,
                    norm   = mcolors.BoundaryNorm([0, 0.5, 1.0], 2),
                    aspect = "auto",
                    interpolation = "nearest",
                    origin = "upper")
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.spines[:].set_visible(False)

    # ── Hide unused grid cells (when n_layers < n_rows × n_cols) ─────────────
    for li in range(n_layers, n_rows * n_cols):
        ri = li // n_cols
        ci = li % n_cols
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[ri, ci])
        for idx in range(2):
            ax = fig.add_subplot(inner[idx])
            ax.set_visible(False)

    # ── Shared colourbar (lower 70% of the right margin) ─────────────────────
    # Colorbar shows relative [0→1] scale — actual per-panel range is in cell title
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar_left = 1.0 - (CBAR_IN - 0.12) / fig_w
    cbar_ax   = fig.add_axes([cbar_left, 0.05, 0.16 / fig_w, 0.60])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["low", "mid", "high"])
    cb.set_label(
        "Mean activation across class samples (scale per panel; see title for range)",
        fontsize=FS_CBAR - 1)
    cb.ax.tick_params(labelsize=FS_CBAR_TK)

    # ── Legend for purity strip ───────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor="#2196F3", edgecolor="none",
              label=f"class-selective\n(≥{CLASS_SELECTIVITY_RATIO}× mean)"),
        Patch(facecolor="white",   edgecolor="#aaaaaa", label="not selective"),
    ]
    n_empty = n_rows * n_cols - n_layers
    if qty == "mlp" and n_empty >= 1:
        # Place legend in the bottom-left empty cell
        empty_li  = n_layers          # first empty slot index
        empty_ri  = empty_li // n_cols
        empty_ci  = empty_li % n_cols
        # Get the bounding box of that grid cell in figure fraction
        ss   = outer[empty_ri, empty_ci]
        fig.canvas.draw()             # needed to resolve GridSpec positions
        bb   = ss.get_position(fig)   # Bbox in figure fraction
        fig.legend(
            handles   = legend_handles,
            title     = "Selectivity strip",
            title_fontsize = FS_LEGEND,
            fontsize  = FS_LEGEND,
            loc       = "center",
            framealpha= 0.95,
            bbox_to_anchor = (bb.x0, bb.y0, bb.width, bb.height),
            bbox_transform = fig.transFigure,
        )
    else:
        fig.legend(
            handles   = legend_handles,
            title     = "Selectivity strip",
            title_fontsize = FS_LEGEND,
            fontsize  = FS_LEGEND,
            loc       = "upper right",
            framealpha= 0.95,
            bbox_to_anchor = (1.0, 0.99),
            bbox_transform = fig.transFigure,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    section_label = {"ast": "AST", "builtin": "Built-in"}.get(section, section)
    qty_label     = {"residual": "Residual stream", "mlp": "MLP neurons"}.get(qty, qty)
    subtitle = (
        f"(neurons reshaped to {grid_shape[0]}×{grid_shape[1]}  ·  "
        f"colour = mean activation across class samples)"
    )
    fig.suptitle(
        f"{qty_label}  ·  {section_label} factor  ·  {col_name}\n{subtitle}"
        if qty == "mlp" else
        f"{qty_label}  ·  {section_label} factor  ·  {col_name}   {subtitle}",
        fontsize=FS_SUPTITLE, y=1.0, va="bottom", x=0.5,
    )

    save_path = save_path.with_suffix(f".{fmt}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white",
                format=fmt)
    plt.close(fig)
    print(f"    → {save_path.name}  ({n_rows}×{n_cols} grid)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = _parse_args()
    here     = Path(__file__).parent
    in_root  = here / args.in_dir
    out_root = here / (args.out_dir if args.out_dir else args.in_dir)

    ids_dir = in_root / "neuron_ids"
    if not ids_dir.exists():
        raise FileNotFoundError(
            f"neuron_ids/ not found under {in_root}\n"
            "Run 02b_vp_projection_viz.py first."
        )

    qtys = ["residual", "mlp"] if args.qty == "both" else [args.qty]

    for qty in qtys:
        json_files = sorted(ids_dir.glob(f"{qty}_L*_neuron_ids.json"))
        if not json_files:
            print(f"[warn] no {qty} neuron_ids files in {ids_dir}")
            continue

        layer_data: list[dict] = []
        for jp in json_files:
            with open(jp) as f:
                d = json.load(f)
            if args.layer is None or d["layer"] == args.layer:
                layer_data.append(d)
        layer_data.sort(key=lambda d: d["layer"])

        if not layer_data:
            print(f"[warn] no matching layers for {qty}")
            continue

        for section in ("ast", "builtin"):
            all_classes: set[str] = set()
            for d in layer_data:
                all_classes.update(d[section]["by_class"].keys())
            col_names = ["Overall"] + sorted(all_classes)

            sec_dir = out_root / "figures" / "neuron_grids" / f"{qty}_{section}"
            sec_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n── {qty}/{section}  "
                  f"({len(layer_data)} layers, {len(col_names)} cols)")

            for col_name in col_names:
                safe = col_name.replace("/", "_").replace(" ", "_")
                plot_class_layers(
                    layer_data, section, qty, col_name,
                    sec_dir / safe,
                    args.dpi,
                    args.layer_cols,
                    fmt=args.format,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
