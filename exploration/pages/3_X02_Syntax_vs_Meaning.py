import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="X02 — Syntax vs Meaning", layout="wide")

OUT = Path("experiments/outputs/x02")


def safe_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not yet generated: {path.name}")


@st.cache_data
def load_data():
    return pd.read_csv(OUT / "ranked_circuits.csv")


@st.cache_data
def load_exploded():
    p = OUT / "ranked_circuits_by_domain_pair.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


st.title("X02 — Syntax vs Meaning")
st.markdown(
    "Ranks circuits by Jensen-Shannon divergence to separate syntax-dominant from "
    "meaning-sensitive representations."
)

csv_path = OUT / "ranked_circuits.csv"
if not csv_path.exists():
    st.warning("CSV not found — run experiment X02 first.")
    st.stop()

df = load_data()
exploded_df = load_exploded()

# --- Sidebar filters ---
layers = sorted(df["layer"].unique())
interps = sorted(df["interpretation"].unique())

st.sidebar.header("X02 Filters")
sel_layers = st.sidebar.multiselect("Layers", layers, default=layers)
sel_interps = st.sidebar.multiselect("Interpretation", interps, default=interps)
jsd_min, jsd_max = float(df["jsd"].min()), float(df["jsd"].max())
jsd_range = st.sidebar.slider("JSD range", jsd_min, jsd_max, (jsd_min, jsd_max))
pair_search = st.sidebar.text_input("Pair search")

mask = (
    df["layer"].isin(sel_layers)
    & df["interpretation"].isin(sel_interps)
    & df["jsd"].between(*jsd_range)
)
if pair_search:
    mask = mask & df["pair"].str.contains(pair_search, case=False, na=False)
filt = df[mask]

# Filter exploded df with same criteria
filt_exploded = None
if exploded_df is not None:
    emask = (
        exploded_df["layer"].isin(sel_layers)
        & exploded_df["interpretation"].isin(sel_interps)
        & exploded_df["jsd"].between(*jsd_range)
    )
    if pair_search:
        emask = emask & exploded_df["pair"].str.contains(pair_search, case=False, na=False)
    filt_exploded = exploded_df[emask]

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("JSD vs Circuit Cosine Similarity")
    if filt_exploded is not None and not filt_exploded.empty:
        fig = px.scatter(
            filt_exploded,
            x="jsd",
            y="circuit_cosine",
            color="domain_pair",
            hover_data=["pair", "layer"],
            labels={"jsd": "JSD", "circuit_cosine": "Circuit Cosine Sim"},
        )
    else:
        fig = px.scatter(
            filt,
            x="jsd",
            y="mean_cross_domain_cosine",
            color="interpretation",
            hover_data=["pair", "layer"],
            labels={"jsd": "JSD", "mean_cross_domain_cosine": "Mean Cosine Sim"},
        )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "How this plot is generated: for each (pair, layer), prompts are grouped by domain, then a mean "
        "activation vector is computed per domain. X-axis is JSD across these domain-mean vectors (after "
        "softmax(abs(vector)) normalization); Y-axis is the pairwise cosine similarity between domain-mean "
        "vectors, colored by domain pair."
    )

with col2:
    st.subheader("Interpretation Count per Layer")
    counts = filt.groupby(["layer", "interpretation"]).size().reset_index(name="count")
    fig2 = px.bar(
        counts, x="layer", y="count", color="interpretation", barmode="group",
        labels={"count": "Count", "layer": "Layer"},
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info(
        "Possible next step: if later layers are more meaning-sensitive, test a causal intervention by "
        "applying matched-norm steering vectors at early vs late layers and measuring domain-specific output "
        "shift while tracking syntax preservation."
    )

# --- Residual stream baseline scatter ---
st.subheader("JSD vs Residual Stream Cross-Domain Cosine")
if filt_exploded is not None and not filt_exploded.empty:
    fig_resid = px.scatter(
        filt_exploded,
        x="jsd",
        y="resid_cosine",
        color="domain_pair",
        hover_data=["pair", "layer"],
        labels={
            "jsd": "JSD",
            "resid_cosine": "Residual Stream Cosine Sim",
        },
    )
    st.plotly_chart(fig_resid, use_container_width=True)
elif "resid_cross_domain_cosine" in filt.columns:
    fig_resid = px.scatter(
        filt,
        x="jsd",
        y="resid_cross_domain_cosine",
        color="interpretation",
        hover_data=["pair", "layer"],
        labels={
            "jsd": "JSD",
            "resid_cross_domain_cosine": "Residual Stream Cosine Sim",
        },
    )
    st.plotly_chart(fig_resid, use_container_width=True)
st.markdown(
    "**Baseline comparison**: Y-axis is the pairwise cosine similarity of "
    "full residual stream vectors (d_model=1024, before circuit projection) across "
    "domain-mean vectors. Comparing this to the circuit-projected cosine above reveals "
    "whether a circuit amplifies or suppresses domain differences relative to the whole model."
)

# --- PNG ---
st.subheader("Domain JSD Heatmap")
st.warning(
    "Attack-first domain-sensitive pairs (high L7 JSD): "
    "Attribute__hasattr, AnnAssign__open, Lambda__max, Assign__bytearray, "
    "With__list, Assign__abs, Call__hasattr, Call__round."
)
safe_image(OUT / "domain_jsd_heatmap.png")

# --- Table ---
st.subheader("Raw Data")
st.dataframe(filt, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  SUB-EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.header("Sub-Experiments")

OUT_BASE = Path("experiments/outputs")

# ── X02-E: Amplification Index ────────────────────────────────────────────
ai_csv = OUT_BASE / "x02e" / "amplification_index.csv"
if ai_csv.exists():
    st.subheader("X02-E — Amplification Index")
    st.markdown(
        "**AI = circuit cosine / residual cosine**. AI > 1 means the circuit "
        "amplifies cross-domain similarity; AI < 1 means it suppresses it."
    )
    ai_df = pd.read_csv(ai_csv)
    ai_filt = ai_df[ai_df["pair"].isin(filt["pair"].unique())] if pair_search else ai_df
    ai_filt = ai_filt[ai_filt["layer"].isin(sel_layers)]

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_ai = px.scatter(
            ai_filt, x="jsd", y="amplification_index",
            color="interpretation", hover_data=["pair", "layer"],
            labels={"jsd": "JSD", "amplification_index": "Amplification Index"},
        )
        fig_ai.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="AI=1 (neutral)")
        st.plotly_chart(fig_ai, use_container_width=True)
    with col_e2:
        fig_ai_box = px.box(
            ai_filt, x="layer", y="amplification_index", color="interpretation",
            labels={"layer": "Layer", "amplification_index": "Amplification Index"},
        )
        fig_ai_box.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_ai_box, use_container_width=True)

# ── X02-B: Phase Transition ──────────────────────────────────────────────
trans_csv = OUT_BASE / "x02b" / "transition_curves.csv"
if trans_csv.exists():
    st.subheader("X02-B — Syntax→Meaning Phase Transition")
    st.markdown(
        "Sigmoid fit to JSD across layers per pair. **L_transition** is the layer "
        "where the syntax→meaning shift occurs; **k_steepness** is how abrupt."
    )
    trans_df = pd.read_csv(trans_csv)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        safe_image(OUT_BASE / "x02b" / "transition_histogram.png",
                   caption="Distribution of transition layers")
    with col_b2:
        valid_trans = trans_df.dropna(subset=["L_transition", "r_squared"])
        if not valid_trans.empty:
            fig_trans = px.scatter(
                valid_trans, x="L_transition", y="k_steepness",
                size="r_squared", hover_data=["pair"],
                labels={
                    "L_transition": "Transition Layer",
                    "k_steepness": "Steepness (k)",
                    "r_squared": "R²",
                },
                title="Transition Layer vs Steepness (size = R²)",
            )
            st.plotly_chart(fig_trans, use_container_width=True)
        else:
            st.info("No valid sigmoid fits found.")

# ── X02-C: Domain Probes ─────────────────────────────────────────────────
probe_csv = OUT_BASE / "x02c" / "probe_results.csv"
if probe_csv.exists():
    st.subheader("X02-C — Domain Probes (Linear Classifiability)")
    st.markdown(
        "5-fold logistic regression accuracy predicting domain from activations. "
        "**delta_acc = circuit − residual**: positive means the circuit makes domain "
        "more linearly separable than the raw residual stream."
    )
    probe_df = pd.read_csv(probe_csv)
    probe_filt = probe_df[probe_df["layer"].isin(sel_layers)]
    if pair_search:
        probe_filt = probe_filt[probe_filt["pair"].str.contains(pair_search, case=False, na=False)]

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        probe_melt = probe_filt.melt(
            id_vars=["pair", "layer"],
            value_vars=["circuit_probe_acc", "resid_probe_acc"],
            var_name="probe_type", value_name="accuracy",
        )
        probe_melt["probe_type"] = probe_melt["probe_type"].map({
            "circuit_probe_acc": "Circuit",
            "resid_probe_acc": "Residual",
        })
        fig_probe = px.box(
            probe_melt, x="layer", y="accuracy", color="probe_type",
            labels={"layer": "Layer", "accuracy": "5-Fold Accuracy"},
            title="Domain Probe Accuracy by Layer",
        )
        st.plotly_chart(fig_probe, use_container_width=True)
    with col_c2:
        fig_delta = px.histogram(
            probe_filt, x="delta_acc", nbins=40,
            labels={"delta_acc": "Delta Accuracy (circuit − residual)"},
            title="Distribution of Delta Accuracy",
        )
        fig_delta.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_delta, use_container_width=True)

# ── X02-A: Causal Ablation ───────────────────────────────────────────────
abl_csv = OUT_BASE / "x02a" / "ablation_results.csv"
if abl_csv.exists():
    st.subheader("X02-A — Causal Ablation")
    st.markdown(
        "Zero-ablating top syntax and meaning circuits, measuring KL divergence "
        "shift. Higher KL = bigger causal effect on model output."
    )
    abl_df = pd.read_csv(abl_csv)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        fig_abl = px.box(
            abl_df, x="interpretation", y="kl_divergence", color="domain",
            labels={
                "interpretation": "Circuit Type",
                "kl_divergence": "KL Divergence",
            },
            title="Ablation KL by Circuit Type and Domain",
        )
        st.plotly_chart(fig_abl, use_container_width=True)
    with col_a2:
        abl_mean = abl_df.groupby(["pair", "layer", "interpretation"])["kl_divergence"] \
            .mean().reset_index()
        fig_abl2 = px.scatter(
            abl_mean, x="layer", y="kl_divergence", color="interpretation",
            hover_data=["pair"],
            labels={"layer": "Layer", "kl_divergence": "Mean KL Divergence"},
            title="Ablation Effect by Layer",
        )
        st.plotly_chart(fig_abl2, use_container_width=True)

# ── X02-D: Positional JSD ────────────────────────────────────────────────
pos_csv = OUT_BASE / "x02d" / "jsd_by_position.csv"
if pos_csv.exists():
    st.subheader("X02-D — Positional JSD")
    st.markdown(
        "JSD computed at 5 token positions (first, 25%, 50%, 75%, last) — "
        "reveals how domain sensitivity evolves along the input sequence."
    )
    pos_df = pd.read_csv(pos_csv)
    pos_filt = pos_df[pos_df["layer"].isin(sel_layers)]
    if pair_search:
        pos_filt = pos_filt[pos_filt["pair"].str.contains(pair_search, case=False, na=False)]

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        pos_order = ["first", "q1", "mid", "q3", "last"]
        pos_mean = pos_filt.groupby(["position", "layer"])["jsd"].mean().reset_index()
        fig_pos = px.line(
            pos_mean, x="position", y="jsd", color="layer",
            category_orders={"position": pos_order},
            labels={"position": "Token Position", "jsd": "Mean JSD", "layer": "Layer"},
            title="Mean JSD by Token Position",
            markers=True,
        )
        st.plotly_chart(fig_pos, use_container_width=True)
    with col_d2:
        pos_pivot = pos_filt.groupby(["layer", "position"])["jsd"].mean().reset_index()
        pos_wide = pos_pivot.pivot(index="layer", columns="position", values="jsd")
        pos_wide = pos_wide.reindex(columns=pos_order)
        fig_heat = px.imshow(
            pos_wide, aspect="auto", color_continuous_scale="RdYlGn_r",
            labels={"x": "Token Position", "y": "Layer", "color": "JSD"},
            title="JSD Heatmap: Layer × Position",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
