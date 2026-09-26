import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="X06 — Subspace Geometry", layout="wide")

OUT = Path("experiments/outputs/x06")


def safe_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not yet generated: {path.name}")


st.title("X06 — Subspace Geometry")
st.markdown(
    "Geometric analysis of activation subspaces — reconstruction quality, probe "
    "accuracy, pair stability, and neighborhood distances across layers."
)

if not OUT.exists():
    st.warning("X06 output directory not found — run experiment X06 first.")
    st.stop()

# --- Detect Phase B subdirectories ---
subdirs = sorted([d for d in OUT.iterdir() if d.is_dir()])
h5_files = sorted(OUT.glob("*.h5"))

phase_b_exists = len(subdirs) > 0

if phase_b_exists:
    st.sidebar.header("X06 Filters")
    pair_names = [d.name for d in subdirs]
    sel_pair = st.sidebar.selectbox("Target pair", pair_names)
    pair_dir = OUT / sel_pair

    if sel_pair == "If__int":
        st.subheader("Observations — If__int")
        st.markdown(
            "So there are two hypothesis, Compositional and Entangled. We want to see "
            "Compositional so the C_additive and D_pair are close together.\n\n"
            "Interaction residual norm by layer: the range is from 0.0 to 0.7, and the "
            "Interection residual increases expo from layer 4 to 7 indicating entangled stuff.\n\n"
            "Increasing entanglement at later layers also suggests that earlier layers are more "
            "compositional: adding representation(AST) and representation(builtin) explains the pair "
            "better there, but overall explanatory power is still low.\n\n"
            "Pair stability: the between curve is above the within curve, and the gap increases clearly "
            "toward later layers from 4 to 7. The range is 0 to 10.\n\n"
            "Why this is best: low within means each pair forms a tight cluster, and high between means "
            "different pairs are clearly separated.\n\n"
            "Linear probes: this plot evaluates whether labels can be predicted from MLP output activations "
            "in R^2048. The range is 0 to 1. Across layers, all three classifiers (AST, builtin, pair) are "
            "largely flat, including pair-specific decoding.\n\n"
            "PCA cumulative explained variance: the curves look logarithmic-like because cumulative variance "
            "increases quickly with the first components, then gains get smaller as more components are added. "
            "This is computed by mean-centering activations, running SVD, converting singular values to "
            "eigenvalues, and then accumulating explained-variance ratios per component. In short, L1 appears "
            "above most others.\n\n"
            "Pair centroid distances heatmap: brighter means farther centroid distance (more different "
            "representation), darker means closer (more similar). In this case, Starred__list looks brightest, "
            "so it behaves like an outlier pair versus the rest. Likely reason is Starred unpacking syntax "
            "(e.g., head, *rest = items) is structurally different from most other pairs shown.\n\n"
            "Dimensionality interpretation: we see effective rank and participation ratio peaking in mid/deeper "
            "layers, which suggests broader, richer representations there. We would want to see this broad "
            "dimensionality together with small interaction residuals if the representation were cleanly "
            "compositional. Instead, later layers show broader structure but also stronger pair-specific "
            "interaction.\n\n"
            "Interpretation / hypothesis: if compositional structure is strong, AST and builtin should both "
            "be linearly decodable and the additive model should approach the pair model; here the AST signal "
            "is stronger than builtin, and pair-specific effects increase in later layers.\n\n"
            "R2plot: D_pair is above all, mix of B_built in and A_ast in the middle, "
            "C_additive at bottom. range is 0.10 to -0.06. It seems that throug layer 0-1 "
            "all increase especially additive, then go down to until layer 4, and rise up until 6 "
            "and go down at 7. C_additive curve is most exagerated but all look as if parallel "
            "except for the mix of A_ast and B_built in in the middle as if they are competing.\n\n"
            "COSINEplot: the range is from 1 to 0.55? all curves start at 0.94 and highly overlap. "
            "they go up at 0.97? at layer 1 and go down slightly at layer 3 to 0.94? then sharp drop "
            "at layer 4 to 0.63? and gradualy increase, until layer 7. but the increase looks like, "
            "D_pair is above all, the rest is higly overlapping as if single line."
        )
    elif sel_pair == "For__list":
        st.subheader("Observations — For__list")
        st.markdown(
            "Layer-4 note: A_ast, B_builtin, and D_pair are close, but this appears to be a low-signal layer "
            "rather than strong agreement. Their R2 values are all low (A≈0.023, B≈0.013, D≈0.026), while "
            "C_additive is clearly worse (negative R2, ≈-0.083). So the takeaway is weak reconstruction for "
            "all, with pair-specific only slightly better and additive composition breaking at this layer."
        )
    elif sel_pair == "Assign__dict":
        st.subheader("Observations — Assign__dict")
        st.markdown(
            "R2 plot: A_ast and B_builtin are intermingled. A_ast starts smaller than B_builtin, and is below "
            "B at L5 and L6, but beats B at L7. PCA cumulative variance: L1, L2, and L3 are almost overlapping, "
            "with L1 still slightly dominating."
        )
    elif sel_pair == "Call__zip":
        st.subheader("Observations — Call__zip")
        st.markdown(
            "R2 plot: there is a big split between A_ast and B_builtin. In the heatmap, Call__bool appears "
            "super yellow (most distant centroid), likely because its syntax/pattern behaves as an outlier."
        )
    elif sel_pair == "Return__len":
        st.subheader("Observations — Return__len")
        st.markdown(
            "Return__len shows partial compositionality at L1 and L6 (C_additive above A_ast at those layers), "
            "yet cumulative reconstruction quality is still weak overall. Call__len appears very yellow in the "
            "distance heatmap."
        )

    # --- PNGs ---
    st.subheader(f"Plots — {sel_pair}")

    png_groups = [
        ("Model Comparison", "model_comparison_by_layer.png"),
        ("Interaction Residual", "interaction_residual_by_layer.png"),
        ("Pair Stability", "pair_stability_by_layer.png"),
        ("Probe Accuracy", "probe_accuracy_by_layer.png"),
        ("PCA Explained Variance", "pca_explained_variance.png"),
        ("Neighborhood Distances", "neighborhood_distances.png"),
    ]

    cols = st.columns(2)
    for i, (title, fname) in enumerate(png_groups):
        with cols[i % 2]:
            safe_image(pair_dir / fname, title)

    # PCA 2D projections
    st.subheader("PCA 2D Projections")
    pca_cols = st.columns(3)
    for i, layer in enumerate([0, 4, 7]):
        with pca_cols[i]:
            safe_image(pair_dir / f"pca_2d_layer{layer}.png", f"Layer {layer}")

    # --- CSVs ---
    csv_files = {
        "Reconstruction Metrics": "reconstruction_metrics.csv",
        "Pair Stability": "pair_stability.csv",
        "Probe Accuracy": "probe_accuracy.csv",
        "Neighborhood Distances": "neighborhood_distances.csv",
    }

    for label, fname in csv_files.items():
        csv_path = pair_dir / fname
        if csv_path.exists():
            csv_df = pd.read_csv(csv_path)

            if fname == "reconstruction_metrics.csv" and "layer" in csv_df.columns:
                st.subheader(f"Reconstruction R² — {sel_pair}")
                r2_col = [c for c in csv_df.columns if "r2" in c.lower() or "r_squared" in c.lower()]
                model_col = [c for c in csv_df.columns if "model" in c.lower()]
                if r2_col:
                    y_col = r2_col[0]
                    color_col = model_col[0] if model_col else None
                    fig = px.line(csv_df, x="layer", y=y_col, color=color_col, markers=True)
                    st.plotly_chart(fig, use_container_width=True)

            if fname == "probe_accuracy.csv" and "layer" in csv_df.columns:
                st.subheader(f"Probe Accuracy — {sel_pair}")
                acc_col = [c for c in csv_df.columns if "acc" in c.lower()]
                if acc_col:
                    fig = px.line(csv_df, x="layer", y=acc_col[0], markers=True)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader(label)
            st.dataframe(csv_df, use_container_width=True)

else:
    st.info("Phase B has not been run yet. Only raw activation files (H5) are available.")

    if h5_files:
        st.subheader("Available H5 Files")
        rows = []
        for f in h5_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            rows.append({"File": f.name, "Size (MB)": f"{size_mb:.1f}"})
        st.table(rows)
    else:
        st.warning("No H5 files found in x06 output directory.")

    st.subheader("Run Phase B")
    st.code("python -m experiments.x06_subspace_geometry --phase b", language="bash")
