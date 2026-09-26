import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="X08 — Variance Partition", layout="wide")

MAATT_DIR = Path("COMP0087_CW/Code/MaaTt/AST-Revisited")
WRAPPER = Path("experiments/scripts/x08_ast_revisited.py")
STEM = "contrastive_stubs"

# Primary: wrapper output dir.  Fallback: Matt's dir.
OUT = Path("experiments/outputs/x08")
if not OUT.exists() and MAATT_DIR.exists():
    OUT = MAATT_DIR

P01 = f"01_{STEM}_"
P02 = f"02_{STEM}_"
P03 = f"03_{STEM}_"
P04 = f"04_{STEM}_"
P05 = f"05_{STEM}_"
P06 = f"06_{STEM}_"
P07 = f"07_{STEM}_"


# ── Helpers ──────────────────────────────────────────────────────────────

def _npy(name: str):
    p = OUT / name
    return np.load(p, allow_pickle=True) if p.exists() else None


def _npz(name: str):
    p = OUT / name
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def _json(name: str):
    p = OUT / name
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _img(path: Path, caption: str | None = None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    return False


def _show_plots(prefix: str, cols: int = 2):
    """Show all plot_*.png files matching prefix in a grid."""
    plots = sorted(OUT.glob(f"{prefix}plot_*.png"))
    if not plots:
        return False
    grid = st.columns(cols)
    for i, p in enumerate(plots):
        with grid[i % cols]:
            st.image(str(p), caption=p.stem, use_container_width=True)
    return True


def _show_json_table(name: str, label: str = ""):
    """Load a JSON file and render as a DataFrame."""
    data = _json(name)
    if data is None:
        return False
    if label:
        st.subheader(label)
    if isinstance(data, list):
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    elif isinstance(data, dict):
        # Try to make it a table if values are dicts/lists
        try:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        except Exception:
            st.json(data)
    return True


# ── Header ───────────────────────────────────────────────────────────────

st.title("X08 — MaaT Variance Partition (Unified)")
st.markdown(
    "Runs MaaT's 01→07 pipeline and visualises outputs. "
    f"Reading from `{OUT}`."
)

if not MAATT_DIR.exists():
    st.error(f"MaaT directory not found: {MAATT_DIR}")
    st.stop()

st.caption(f"Pipeline dir: `{MAATT_DIR}` · Output dir: `{OUT}`")


# ── Runner panel ─────────────────────────────────────────────────────────

with st.expander("Run MaaT pipeline (01–07)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    device = c1.text_input("Device", value="cpu")
    batch = c2.number_input("Batch", min_value=1, value=8, step=1)
    max_tokens = c3.number_input("Max tokens", min_value=16, value=128, step=16)
    n_perm = c4.number_input("RSA perms", min_value=100, value=1000, step=100)

    p1, p2 = st.columns(2)
    do_plot = p1.checkbox("Generate plots", value=True)
    all_layers = p2.checkbox("All layers / all concepts", value=True)

    st.markdown("**Select steps:**")
    step_cols = st.columns(8)
    step_checks = {}
    for i, s in enumerate(["01", "02", "03", "04", "05", "06", "07"]):
        step_checks[s] = step_cols[i].checkbox(s, value=True)
    run_btn = step_cols[7].button("Run", use_container_width=True)

    if run_btn:
        selected = [s for s, v in step_checks.items() if v]
        if not selected:
            st.warning("No steps selected.")
        else:
            cmd = [
                sys.executable, str(WRAPPER),
                "--steps", ",".join(selected),
                "--out", str(OUT),
                "--device", str(device),
                "--batch-size", str(int(batch)),
                "--max-tokens", str(int(max_tokens)),
                "--n-perm", str(int(n_perm)),
            ]
            if do_plot:
                cmd.append("--plot")
            else:
                cmd.append("--no-plot")
            if all_layers:
                cmd.append("--all-layers")
            else:
                cmd.append("--no-all-layers")

            st.write(f"Running: `{' '.join(cmd)}`")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            logs = "\n".join(x for x in [proc.stdout, proc.stderr] if x)
            st.code(logs if logs else "(no logs)", language="bash")
            if proc.returncode != 0:
                st.error(f"Failed with exit code {proc.returncode}")
            else:
                st.success("Pipeline finished.")


# ── Check minimal outputs ────────────────────────────────────────────────

required = [f"{P01}meta.json"]
if not any((OUT / name).exists() for name in required):
    st.warning(
        "No outputs found. Run the pipeline from the panel above, or check the output directory.\n\n"
        f"Expected: `{OUT / required[0]}`"
    )
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────────────────

tab01, tab02, tab03, tab04, tab05, tab06, tab07 = st.tabs([
    "01 Extraction",
    "02 Variance Partition",
    "03 Projection & RSA",
    "04 Additivity",
    "05 Probing",
    "06 Causal",
    "07 Loss Analysis",
])


# ── Tab 01: Extraction ──────────────────────────────────────────────────

with tab01:
    st.header("01 — Activation Extraction")

    meta = _json(f"{P01}meta.json")
    if meta is None:
        st.warning("No extraction metadata found. Run step 01 first.")
        st.stop()

    df_meta = pd.DataFrame(meta)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompts", len(df_meta))
    c2.metric("AST nodes", df_meta["ast_node"].nunique())
    c3.metric("Builtins", df_meta["builtin_obj"].nunique())
    c4.metric("Variants", df_meta["variant_type"].nunique())

    st.subheader("Extracted arrays")
    rows = []
    for name, desc in [
        (f"{P01}residual_all.npy", "Residual stream (N, L+1, H)"),
        (f"{P01}residual_final.npy", "Residual final layer (N, H)"),
        (f"{P01}head_attr.npy", "Head attribution (N, L, n_heads)"),
        (f"{P01}mlp_attr.npy", "MLP attribution (N, L)"),
        (f"{P01}mlp_neurons.npy", "MLP neurons (N, L, 4H)"),
    ]:
        p = OUT / name
        if p.exists():
            arr = np.load(p, mmap_mode="r")
            rows.append({"Array": desc, "Shape": str(arr.shape), "MB": f"{p.stat().st_size / 1e6:.1f}"})
        else:
            rows.append({"Array": desc, "Shape": "-", "MB": "-"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    edge = _npz(f"{P01}edge_graph.npz")
    if edge is not None:
        w = edge.get("weights", edge.get("weight", []))
        st.metric("Edge graph (non-zero weights)", f"{len(w):,}")

    st.subheader("Stub distribution")
    ca, cb = st.columns(2)
    with ca:
        st.bar_chart(df_meta["variant_type"].value_counts())
    with cb:
        st.bar_chart(df_meta["ast_node"].value_counts().head(15))


# ── Tab 02: Variance Partition ──────────────────────────────────────────

with tab02:
    st.header("02 — Variance Partition")

    summary = _json(f"{P02}summary.json")
    if summary is None:
        st.warning("No VP summary found. Run step 02 first.")
        st.stop()

    st.subheader("Global summary")
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    st.subheader("Mean VP per layer")
    bar_cols = st.columns(2)
    for i, qty in enumerate(["residual", "heads", "mlp_attr", "mlp_neurons"]):
        with bar_cols[i % 2]:
            if not _img(OUT / f"{P02}plot_stacked_{qty}.png", qty):
                vp = _npz(f"{P02}vp_{qty}.npz")
                if vp is not None:
                    m = lambda k: vp[k].mean(axis=-1) if vp[k].ndim > 1 else vp[k]
                    chart = pd.DataFrame(
                        {
                            "Unique AST": m("unique_ast"),
                            "Unique Builtin": m("unique_builtin"),
                            "Shared": m("shared"),
                        }
                    )
                    st.bar_chart(chart)
                    st.caption(qty)

    st.subheader("Heatmaps (layer x unit)")
    for qty in ["residual", "mlp_neurons"]:
        hm_cols = st.columns(3)
        for j, comp in enumerate(["unique_ast", "unique_builtin", "shared"]):
            with hm_cols[j]:
                _img(OUT / f"{P02}plot_heatmap_{qty}_{comp}.png", f"{qty} — {comp}")

    ba_plots = sorted(OUT.glob(f"{P02}plot_before_after_*.png"))
    if ba_plots:
        st.subheader("Before vs after VP projection (PCA)")
        for p in ba_plots:
            _img(p, p.stem)

    st.subheader("Top units by VP score")
    for qty in ["residual", "heads", "neurons"]:
        top_cols = st.columns(3)
        for j, comp in enumerate(["unique_ast", "unique_builtin", "shared"]):
            with top_cols[j]:
                _img(OUT / f"{P02}plot_top_{qty}_{comp}.png", f"{qty} — {comp}")

    st.subheader("Purity masks")
    masks = _npz(f"{P02}purity_masks.npz")
    if masks is not None:
        mcols = st.columns(3)
        for i, qty in enumerate(["residual", "heads", "neurons"]):
            with mcols[i]:
                st.markdown(f"**{qty}**")
                mrows = []
                for mt in ["ast_pure", "builtin_pure", "shared", "interaction", "noise"]:
                    key = f"{qty}_{mt}"
                    if key in masks:
                        mrows.append({"Mask": mt, "% units": f"{masks[key].mean():.1%}"})
                if mrows:
                    st.dataframe(pd.DataFrame(mrows), hide_index=True)
                _img(OUT / f"{P02}plot_purity_{qty}.png")

    st.subheader("Edge importance")
    ecols = st.columns(3)
    for j, comp in enumerate(["edge_ast_importance", "edge_builtin_importance", "edge_shared"]):
        with ecols[j]:
            _img(OUT / f"{P02}plot_edge_{comp}.png", comp)


# ── Tab 03: Projection & RSA ────────────────────────────────────────────

with tab03:
    st.header("03 — Projection & RSA")

    rsa = _json(f"{P03}rsa_results.json")
    if rsa is None:
        st.warning("No RSA results found. Run step 03 first.")
        st.stop()

    st.subheader("RSA configuration")
    st.json({k: v for k, v in rsa.items() if k not in ("results_per_layer", "conclusions")})

    if "conclusion_str" in rsa:
        st.info(rsa["conclusion_str"])

    rsa_rows = rsa.get("results_per_layer", [rsa] if "rho_ast" in rsa else [])
    if rsa_rows:
        st.subheader("RSA partial correlations per layer")
        df_rsa = pd.DataFrame(rsa_rows)
        st.dataframe(df_rsa, use_container_width=True, hide_index=True)

        if len(rsa_rows) > 1 and "layer" in df_rsa.columns:
            rho_cols = [c for c in df_rsa.columns if "rho" in c.lower()]
            if rho_cols:
                st.line_chart(df_rsa.set_index("layer")[rho_cols])

    proj_ast = _npy(f"{P03}proj_ast.npy")
    proj_blt = _npy(f"{P03}proj_builtin.npy")
    if proj_ast is not None and proj_blt is not None:
        c1, c2 = st.columns(2)
        c1.metric("AST subspace dims", proj_ast.shape[1] if proj_ast.ndim > 1 else 1)
        c2.metric("Builtin subspace dims", proj_blt.shape[1] if proj_blt.ndim > 1 else 1)

    st.subheader("Plots")
    p03_plots = sorted(OUT.glob(f"{P03}plot_*.png")) + sorted(OUT.glob(f"{P03}*scatter*.png"))
    if p03_plots:
        pcols = st.columns(2)
        for i, p in enumerate(p03_plots):
            with pcols[i % 2]:
                st.image(str(p), caption=p.stem, use_container_width=True)
    else:
        st.info("No plots. Rerun step 03 with `--plot`.")

    st.subheader("Representational Dissimilarity Matrices")
    rdm_names = [
        ("Neural RDM (raw)", f"{P03}rdm_neural_raw.npy"),
        ("Neural RDM (AST subspace)", f"{P03}rdm_neural_ast.npy"),
        ("Neural RDM (builtin subspace)", f"{P03}rdm_neural_builtin.npy"),
        ("Model RDM (AST labels)", f"{P03}rdm_model_ast.npy"),
        ("Model RDM (builtin labels)", f"{P03}rdm_model_builtin.npy"),
    ]
    rdm_found = False
    for label, fname in rdm_names:
        rdm = _npy(fname)
        if rdm is not None and rdm.shape[0] < 500:
            rdm_found = True
            import plotly.express as pfx

            st.plotly_chart(
                pfx.imshow(rdm, color_continuous_scale="RdBu_r", title=label, aspect="equal", zmin=0, zmax=2),
                use_container_width=True,
            )
    if not rdm_found:
        st.info("No RDM files found or matrices too large to display.")

    sim = _npz(f"{P03}similarity_matrices.npz")
    if sim is not None:
        st.subheader("Pairwise similarity matrices")
        for key in sorted(sim.keys()):
            mat = sim[key]
            if getattr(mat, "ndim", 0) == 2 and mat.shape[0] < 100:
                st.markdown(f"**{key}** ({mat.shape[0]}x{mat.shape[1]})")
                st.dataframe(
                    pd.DataFrame(mat).style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                    use_container_width=True,
                )


# ── Tab 04: Additivity ──────────────────────────────────────────────────

with tab04:
    st.header("04 — Additivity Testing")

    pair_summary = _json(f"{P04}pair_summary.json")
    if pair_summary is None:
        st.warning("No additivity results found. Run step 04 first.")
        st.stop()

    _show_json_table(f"{P04}pair_summary.json", "Pair summary")
    _show_json_table(f"{P04}significance.json", "Significance tests")
    _show_json_table(f"{P04}explicit_vs_proxy.json", "Explicit vs proxy comparison")

    scores = _npz(f"{P04}additivity_scores.npz")
    if scores is not None:
        st.subheader("Layer-wise additivity scores")
        for key in sorted(scores.keys()):
            arr = scores[key]
            if arr.ndim == 1:
                st.line_chart(pd.DataFrame({key: arr}, index=range(len(arr))))

    st.subheader("Plots")
    _show_plots(P04)


# ── Tab 05: Probing ─────────────────────────────────────────────────────

with tab05:
    st.header("05 — Probing")

    anova = _json(f"{P05}anova_results.json")
    if anova is None:
        st.warning("No probing results found. Run step 05 first.")
        st.stop()

    _show_json_table(f"{P05}anova_results.json", "ANOVA results")
    _show_json_table(f"{P05}cross_probe.json", "Cross-subspace accuracy")

    acc = _npz(f"{P05}probe_accuracy.npz")
    if acc is not None:
        st.subheader("Probe accuracy")
        for key in sorted(acc.keys()):
            arr = acc[key]
            if arr.ndim <= 2:
                st.markdown(f"**{key}** — shape {arr.shape}")
                st.dataframe(pd.DataFrame(arr), use_container_width=True, hide_index=True)

    auroc = _npz(f"{P05}concept_auroc.npz")
    if auroc is not None:
        st.subheader("Concept AUROC heatmap")
        for key in sorted(auroc.keys()):
            arr = auroc[key]
            if arr.ndim == 2 and arr.shape[0] < 200:
                import plotly.express as pfx
                st.plotly_chart(
                    pfx.imshow(arr, color_continuous_scale="Viridis", title=key,
                               aspect="auto", zmin=0, zmax=1),
                    use_container_width=True,
                )

    st.subheader("Plots")
    _show_plots(P05)


# ── Tab 06: Causal ──────────────────────────────────────────────────────

with tab06:
    st.header("06 — Causal Analysis")

    patch = _json(f"{P06}patch_summary.json")
    if patch is None:
        st.warning("No causal results found. Run step 06 first.")
        st.stop()

    _show_json_table(f"{P06}patch_summary.json", "Patching shift")
    _show_json_table(f"{P06}ablation_summary.json", "Ablation importance")

    st.subheader("Minimal circuits")
    for circuit_name in ["circuit_ast", "circuit_builtin", "circuit_interaction"]:
        data = _json(f"{P06}{circuit_name}.json")
        if data is not None:
            st.markdown(f"**{circuit_name}**")
            if isinstance(data, dict):
                st.json(data)
            else:
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.subheader("Plots")
    _show_plots(P06)


# ── Tab 07: Loss Analysis ───────────────────────────────────────────────

with tab07:
    st.header("07 — Loss Analysis")

    loss_anova = _json(f"{P07}loss_anova.json")
    if loss_anova is None:
        st.warning("No loss analysis results found. Run step 07 first.")
        st.stop()

    _show_json_table(f"{P07}loss_anova.json", "Loss ANOVA")
    _show_json_table(f"{P07}component_loss_pred.json", "Component loss prediction (R²)")
    _show_json_table(f"{P07}hardness_rsa.json", "RSA per loss quartile")
    _show_json_table(f"{P07}robustness.json", "Robustness stats")

    st.subheader("Plots")
    _show_plots(P07)
