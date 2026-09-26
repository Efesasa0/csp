"""
02j_sae_residual.py  (lives in results/)

Option 2: Sparse Autoencoder (SAE) on the residual stream.

Trains a separate SAE at each residual stream layer on the pre-existing
activation data (no live model required).  Each SAE feature is a learned
direction in the residual stream; we then apply the same selectivity and
exclusivity analysis as 02g to those features, identifying which SAE features
(directions) are class-selective for each AST node.  Finally, the decoder
weights map features back to the original residual stream dimensions so we
can identify which model neurons those features correspond to.

ARCHITECTURE
────────────
  Encoder:  h = ReLU(W_enc @ x + b_enc)   h ∈ R^{dict_size}
  Decoder:  x̂ = W_dec @ h + b_dec         x̂ ∈ R^{D}

  Loss = ||x - x̂||² + λ * ||h||₁

  dict_size = expansion_factor × D  (default 4×)
  W_dec columns are unit-normalised after every optimiser step (tied-norm trick).

WHY THIS WORKS ON RESIDUAL STREAM
──────────────────────────────────
The residual stream of circuit_sparsity uses dense representations (unlike MLP
neurons which are AbsTopK-sparse).  Superposition theory (Elhage et al. 2022)
predicts that concepts are encoded as near-orthogonal directions crammed into
fewer dimensions than there are concepts.  SAE recovers those directions by
finding a sparse linear code — each feature fires rarely but cleanly.

OUTPUTS  (under out_dir)
───────
  02j_<stem>_sae_weights_L{l:02d}.npz    W_enc, b_enc, W_dec, b_dec per layer
  02j_<stem>_sae_features_L{l:02d}.npz   feature activations (N, dict_size)
  02j_<stem>_sae_selectivity.npz         sel (C, L, dict_size)
  02j_<stem>_sae_class_features.json     per-class feature lists + decoder projections
  figures/sae/
    sae_loss_curves.{fmt}                training loss per layer
    sae_feature_atlas.{fmt}              selectivity heatmap (features × classes)
    sae_decoder_projection.{fmt}         how class features project onto residual dims
    sae_exclusive_features.{fmt}         bar: how many exclusive features per class

Usage
─────
  python 02j_sae_residual.py
  python 02j_sae_residual.py --stem contrastive_stubs --in_dir ../data_more
  python 02j_sae_residual.py --expansion 4 --l1_coef 1e-3 --epochs 30
  python 02j_sae_residual.py --layer 5          # single layer only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FS_TITLE    = 11
FS_SUPTITLE = 13
FS_TICK     = 8
FS_ANNOT    = 7

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAE on residual stream")
    p.add_argument("--stem",       default="contrastive_stubs")
    p.add_argument("--in_dir",     default="../data_more")
    p.add_argument("--out_dir",    default="../results/output")
    p.add_argument("--expansion",  type=int,   default=2,
                   help="dict_size = expansion × D  (default 2 for tractability)")
    p.add_argument("--l1_coef",    type=float, default=8e-4,
                   help="L1 sparsity penalty")
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--layer",      type=int,   default=None,
                   help="Train on a single layer only (default: all)")
    p.add_argument("--sel_thresh", type=float, default=0.5)
    p.add_argument("--excl_thresh",type=float, default=0.3)
    p.add_argument("--top_k",      type=int,   default=20)
    p.add_argument("--min_samples",type=int,   default=10)
    p.add_argument("--format",     default="pdf")
    p.add_argument("--dpi",        type=int,   default=200)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# SAE — pure numpy, no framework dependency
# ─────────────────────────────────────────────────────────────────────────────

class SAE:
    """Minimal sparse autoencoder trained with SGD + L1."""

    def __init__(self, D: int, dict_size: int, l1_coef: float, lr: float,
                 rng: np.random.Generator):
        scale = 1.0 / np.sqrt(D)
        self.W_enc = rng.normal(0, scale, (dict_size, D)).astype(np.float32)
        self.b_enc = np.zeros(dict_size, dtype=np.float32)
        self.W_dec = rng.normal(0, scale, (D, dict_size)).astype(np.float32)
        self.b_dec = np.zeros(D, dtype=np.float32)
        self._normalise_dec()
        self.l1   = l1_coef
        self.lr   = lr
        # Adam moments
        self._m  = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v  = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t  = 0

    def _params(self):
        return dict(We=self.W_enc, be=self.b_enc, Wd=self.W_dec, bd=self.b_dec)

    def _normalise_dec(self):
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True).clip(1e-8)
        self.W_dec /= norms

    def encode(self, x: np.ndarray) -> np.ndarray:
        """x: (B, D) → h: (B, dict_size)  ReLU activations."""
        return np.maximum(0.0, x @ self.W_enc.T + self.b_enc)

    def decode(self, h: np.ndarray) -> np.ndarray:
        """h: (B, dict_size) → x̂: (B, D)."""
        return h @ self.W_dec.T + self.b_dec

    def step(self, x: np.ndarray) -> float:
        """One SGD step on batch x (B, D). Returns scalar loss."""
        B = x.shape[0]
        # forward
        h   = self.encode(x)          # (B, F)
        x_hat = self.decode(h)        # (B, D)
        # loss
        recon = ((x - x_hat) ** 2).sum() / B
        l1    = self.l1 * h.sum() / B
        loss  = recon + l1
        # backward
        dL_dxhat  = -2.0 * (x - x_hat) / B          # (B, D)
        dL_dWd    = h.T @ dL_dxhat                   # (F, D) → need transpose
        dL_dWd    = dL_dxhat.T @ h                   # (D, F)
        dL_dbd    = dL_dxhat.sum(axis=0)
        dL_dh     = dL_dxhat @ self.W_dec + self.l1 / B  # (B, F)
        dL_dh    *= (h > 0).astype(np.float32)        # ReLU gate
        dL_dWe    = dL_dh.T @ x                       # (F, D)
        dL_dbe    = dL_dh.sum(axis=0)

        grads = dict(We=dL_dWe, be=dL_dbe, Wd=dL_dWd, bd=dL_dbd)
        self._t += 1
        β1, β2, ε = 0.9, 0.999, 1e-8
        t = self._t
        for k in grads:
            g = grads[k]
            self._m[k] = β1 * self._m[k] + (1 - β1) * g
            self._v[k] = β2 * self._v[k] + (1 - β2) * g * g
            m_hat = self._m[k] / (1 - β1 ** t)
            v_hat = self._v[k] / (1 - β2 ** t)
            step  = self.lr * m_hat / (np.sqrt(v_hat) + ε)
            if k == "We":
                self.W_enc -= step
            elif k == "be":
                self.b_enc -= step
            elif k == "Wd":
                self.W_dec -= step
            elif k == "bd":
                self.b_dec -= step

        self._normalise_dec()
        return float(loss)


def train_sae(acts: np.ndarray, expansion: int, l1_coef: float,
              epochs: int, batch_size: int, lr: float,
              rng: np.random.Generator) -> tuple[SAE, list[float]]:
    """
    acts: (N, D) — zero-mean normalised activations at one layer.
    Returns trained SAE and list of per-epoch mean losses.
    """
    N, D = acts.shape
    F    = expansion * D
    sae  = SAE(D, F, l1_coef, lr, rng)
    losses = []
    idx    = np.arange(N)
    for epoch in range(epochs):
        rng.shuffle(idx)
        epoch_losses = []
        for start in range(0, N, batch_size):
            batch = acts[idx[start:start + batch_size]]
            epoch_losses.append(sae.step(batch))
        losses.append(float(np.mean(epoch_losses)))
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={losses[-1]:.4f}")
    return sae, losses

# ─────────────────────────────────────────────────────────────────────────────
# Selectivity on SAE features  (mirrors 02g logic)
# ─────────────────────────────────────────────────────────────────────────────

def feature_selectivity(feat_acts: np.ndarray,
                        labels: np.ndarray,
                        classes: list[str],
                        min_samples: int) -> np.ndarray:
    """
    feat_acts: (N, F)
    Returns sel (C, F)
    """
    C = len(classes)
    mu_all  = feat_acts.mean(axis=0)
    std_all = feat_acts.std(axis=0) + 1e-8
    sel = np.zeros((C, feat_acts.shape[1]), dtype=np.float32)
    for c in range(C):
        mask = labels == c
        if mask.sum() < min_samples:
            continue
        mu_c = feat_acts[mask].mean(axis=0)
        sel[c] = (mu_c - mu_all) / std_all
    return sel

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _class_colours(n):
    c1 = plt.get_cmap("tab20")
    c2 = plt.get_cmap("tab20b")
    return [c1(i) if i < 20 else c2(i - 20) for i in range(n)]


def plot_loss_curves(all_losses: dict[int, list[float]],
                     fmt: str, dpi: int, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = _class_colours(len(all_losses))
    for (l, losses), col in zip(sorted(all_losses.items()), colours):
        ax.plot(losses, lw=1.5, color=col, label=f"L{l}")
    ax.set_xlabel("Epoch", fontsize=FS_TITLE)
    ax.set_ylabel("Loss  (recon + L1)", fontsize=FS_TITLE)
    ax.set_title("SAE training loss per residual stream layer", fontsize=FS_SUPTITLE)
    ax.legend(fontsize=FS_ANNOT, ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"sae_loss_curves.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'sae_loss_curves.{fmt}'}")


def plot_feature_atlas(all_sel: np.ndarray, classes: list[str],
                       excl_mask: np.ndarray, top_k: int,
                       fmt: str, dpi: int, fig_dir: Path) -> None:
    """
    all_sel:   (C, L, F) — selectivity across layers; use max across layers
    excl_mask: (F,) bool — features with top2 < excl_thresh
    """
    sel_agg = all_sel.max(axis=1)   # (C, F)
    C, F    = sel_agg.shape

    # select top_k per class
    selected = set()
    for c in range(C):
        best = np.argsort(sel_agg[c])[-top_k:]
        selected.update(best.tolist())
    selected = np.array(sorted(selected))

    sub   = sel_agg[:, selected]
    order = np.argsort(sub.argmax(axis=0), kind="stable")
    sub   = sub[:, order]
    is_excl = excl_mask[selected][order]

    fig_w = max(8, 0.12 * sub.shape[1] + 2)
    fig_h = max(5, 0.22 * C + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(2.0, float(np.abs(sub).max()))
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # mark exclusive features
    for fi, exc in enumerate(is_excl):
        if exc:
            ax.axvline(fi, color="#d73027", lw=0.6, alpha=0.7)

    ax.set_yticks(range(C))
    ax.set_yticklabels(classes, fontsize=FS_TICK)
    ax.set_xlabel("SAE features (sorted by argmax class)", fontsize=FS_TITLE)
    ax.set_ylabel("AST node class", fontsize=FS_TITLE)
    ax.set_title("SAE feature selectivity atlas  (red line = exclusive feature)",
                 fontsize=FS_SUPTITLE)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label(
        "Selectivity z-score", fontsize=FS_ANNOT)
    fig.tight_layout()
    fig.savefig(fig_dir / f"sae_feature_atlas.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'sae_feature_atlas.{fmt}'}")


def plot_exclusive_counts(excl_per_class: dict[str, int], classes: list[str],
                          excl_thresh: float,
                          fmt: str, dpi: int, fig_dir: Path) -> None:
    colours = _class_colours(len(classes))
    counts  = [excl_per_class.get(cls, 0) for cls in classes]
    order   = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(classes)), [counts[i] for i in order],
           color=[colours[i] for i in order], alpha=0.85)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([classes[i] for i in order],
                       rotation=45, ha="right", fontsize=FS_TICK)
    ax.set_ylabel("Number of exclusive SAE features", fontsize=FS_TITLE)
    ax.set_title(
        f"Exclusive SAE features per AST class  (excl_thresh = top2 < {excl_thresh})\n"
        f"SAE finds class-exclusive directions even where no single neuron is exclusive",
        fontsize=FS_SUPTITLE,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"sae_exclusive_features.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_dir / f'sae_exclusive_features.{fmt}'}")


def plot_decoder_group_heatmap(out_dir: Path, stem: str,
                               all_sel: np.ndarray,
                               classes: list[str],
                               layers: list[int],
                               D: int,
                               top_k_features: int,
                               top_n_dims: int,
                               sel_thresh: float,
                               fmt: str, dpi: int,
                               fig_dir: Path) -> None:
    """
    Per layer: one heatmap showing the two-level group structure.

      rows    = SAE features (grouped and sorted by owning AST class)
      columns = residual stream dims (sorted by which feature loads them most)
      colour  = decoder weight W_dec[dim, feature]  — how strongly that feature
                writes to that dim (this IS the group structure)

    Left colour strip = owning AST class for each feature row.
    White separators between class blocks.

    This is the correct SAE interpretation: each feature is a GROUP of dims
    that fire together; the decoder weight is the group membership weight.
    """
    colours = _class_colours(len(classes))
    C = len(classes)

    for l in layers:
        weights_path = out_dir / f"02j_{stem}_sae_weights_L{l:02d}.npz"
        if not weights_path.exists():
            print(f"  skipping L{l} — weights not found")
            continue

        w     = np.load(weights_path)
        W_dec = w["W_dec"]                           # (D, F)
        sel_l = all_sel[:, layers.index(l), :]       # (C, F)

        # assign each feature to its most selective class
        owner = sel_l.argmax(axis=0)                 # (F,)
        top1  = sel_l.max(axis=0)

        # keep only features with sel > threshold (has some class preference)
        sel_mask = top1 > sel_thresh
        if sel_mask.sum() == 0:
            print(f"  L{l}: no selective features above threshold, skipping")
            continue

        # restrict to top_k_features per class (by selectivity)
        keep = []
        for c in range(C):
            owned = np.where(sel_mask & (owner == c))[0]
            if len(owned) == 0:
                continue
            k = min(top_k_features, len(owned))
            best = owned[np.argsort(sel_l[c, owned])[-k:]]
            keep.extend(best.tolist())
        keep = np.array(sorted(set(keep)))

        if len(keep) == 0:
            continue

        F_sub    = len(keep)
        owner_sub = owner[keep]                      # (F_sub,)
        W_sub     = W_dec[:, keep]                   # (D, F_sub)  → transpose to (F_sub, D)
        W_sub     = W_sub.T                          # (F_sub, D)

        # sort rows: by class, then descending max decoder weight
        row_order = sorted(range(F_sub),
                           key=lambda i: (int(owner_sub[i]),
                                          -float(np.abs(W_sub[i]).max())))
        W_ord      = W_sub[row_order, :]             # (F_sub, D)
        owner_ord  = owner_sub[row_order]

        # select columns: top-top_n_dims dims by absolute weight across all kept features
        col_importance = np.abs(W_ord).max(axis=0)  # (D,)
        col_idx = np.argsort(col_importance)[-top_n_dims * F_sub // 4:]
        # sort cols by which row loads them most
        col_ord = col_idx[np.argsort(np.abs(W_ord[:, col_idx]).argmax(axis=0),
                                     kind="stable")]
        W_plot = W_ord[:, col_ord]                   # (F_sub, n_cols)

        # ── figure ────────────────────────────────────────────────────────────
        row_h  = max(0.15, 4.5 / F_sub)
        fig_h  = max(6.0, F_sub * row_h + 2.0)
        fig_w  = max(10.0, 0.09 * W_plot.shape[1] + 3.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        vmax = float(np.abs(W_plot).max()) or 1.0
        im   = ax.imshow(W_plot, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, interpolation="nearest")

        # class colour strip on left
        for ri, cls_i in enumerate(owner_ord):
            ax.add_patch(plt.Rectangle(
                (-0.018 * W_plot.shape[1], ri - 0.5),
                0.018 * W_plot.shape[1], 1.0,
                transform=ax.transData,
                color=colours[cls_i], clip_on=False, zorder=3,
            ))

        # class separator lines
        prev = owner_ord[0]
        for ri in range(1, F_sub):
            if owner_ord[ri] != prev:
                ax.axhline(ri - 0.5, color="white", lw=1.0, zorder=4)
                prev = owner_ord[ri]

        # y-axis: feature index labelled by class
        ax.set_yticks(range(F_sub))
        ax.set_yticklabels(
            [f"f{keep[row_order[i]]}  [{classes[owner_ord[i]]}]"
             for i in range(F_sub)],
            fontsize=max(3.5, min(6, 120 / F_sub)),
        )
        for tick, cls_i in zip(ax.get_yticklabels(), owner_ord):
            tick.set_color(colours[cls_i])

        # x-axis: residual dim IDs
        n_xt = max(1, W_plot.shape[1] // 30)
        xt   = list(range(0, W_plot.shape[1], n_xt))
        ax.set_xticks(xt)
        ax.set_xticklabels([f"d{col_ord[i]}" for i in xt],
                           rotation=90, fontsize=FS_ANNOT - 1)
        ax.set_xlabel("Residual stream dimension", fontsize=FS_TITLE)
        ax.set_ylabel("SAE feature  (= group of co-activating dims)",
                      fontsize=FS_TITLE)

        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("Decoder weight  (group membership strength)",
                     fontsize=FS_ANNOT)
        cb.ax.tick_params(labelsize=FS_ANNOT)

        seen   = sorted(set(owner_ord.tolist()))
        handles = [plt.matplotlib.patches.Patch(facecolor=colours[c],
                   label=classes[c]) for c in seen]
        ax.legend(handles=handles, fontsize=5, ncol=3,
                  loc="upper left", bbox_to_anchor=(1.02, 1),
                  framealpha=0.9, title="AST class", title_fontsize=6)

        ax.set_title(
            f"SAE feature groups — Layer {l}  ·  {F_sub} selective features shown\n"
            f"Each row = one SAE feature = a group of residual dims that fire together "
            f"for that AST class\nColour = decoder weight (how strongly each feature "
            f"writes to each dim)",
            fontsize=FS_ANNOT + 1, pad=6,
        )
        fig.tight_layout()
        out = fig_dir / f"sae_groups_L{l:02d}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    in_dir  = Path(__file__).parent / args.in_dir
    out_dir = Path(__file__).parent / args.out_dir
    fig_dir = out_dir / "figures" / "sae"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem
    fmt  = args.format

    print("Loading data …")
    resid = np.load(in_dir / f"01_{stem}_residual_all.npy")   # (N, L, D)
    meta  = json.load(open(in_dir / f"01_{stem}_meta.json"))
    j     = json.load(open(out_dir / f"02g_{stem}_class_neurons.json"))

    classes = j["classes"]
    C       = len(classes)
    N, L, D = resid.shape
    ast_idx  = np.array([classes.index(m["ast_node"]) for m in meta], dtype=np.int32)

    layers = [args.layer] if args.layer is not None else list(range(L))
    F = args.expansion * D
    print(f"  {N} prompts · {L} layers · D={D} · dict_size={F}")

    rng = np.random.default_rng(42)

    all_losses:      dict[int, list[float]]   = {}
    all_sel_layers:  list[np.ndarray]         = []   # (C, F) per layer
    excl_per_class:  dict[str, int]           = {cls: 0 for cls in classes}
    class_features:  dict                    = {cls: {} for cls in classes}

    best_sae  = None
    best_sel  = None
    best_layer_idx = layers[0]

    for l in layers:
        print(f"\nLayer {l} — training SAE (expansion={args.expansion}, "
              f"l1={args.l1_coef}, epochs={args.epochs}) …")
        acts = resid[:, l, :].astype(np.float32)   # (N, D)

        # zero-mean per dimension (important for SAE training stability)
        mu   = acts.mean(axis=0)
        acts = acts - mu

        sae, losses = train_sae(
            acts, args.expansion, args.l1_coef,
            args.epochs, args.batch_size, args.lr, rng,
        )
        all_losses[l] = losses

        # save weights
        np.savez_compressed(
            out_dir / f"02j_{stem}_sae_weights_L{l:02d}.npz",
            W_enc=sae.W_enc, b_enc=sae.b_enc,
            W_dec=sae.W_dec, b_dec=sae.b_dec,
            mu=mu,
        )

        # feature activations for all prompts
        feats = sae.encode(acts)   # (N, F)
        np.savez_compressed(out_dir / f"02j_{stem}_sae_features_L{l:02d}.npz",
                            feats=feats)

        # selectivity
        sel_l = feature_selectivity(feats, ast_idx, classes, args.min_samples)
        all_sel_layers.append(sel_l)   # (C, F)

        # exclusivity at this layer
        top1_l = sel_l.max(axis=0)
        top2_l = np.sort(sel_l, axis=0)[-2]
        excl_l = (top1_l > args.sel_thresh) & (top2_l < args.excl_thresh)
        owner_l = sel_l.argmax(axis=0)

        n_excl_this_layer = 0
        for c, cls in enumerate(classes):
            owned = np.where(excl_l & (owner_l == c))[0]
            excl_per_class[cls] += len(owned)
            n_excl_this_layer   += len(owned)
            if len(owned) > 0:
                class_features[cls][f"L{l:02d}"] = {
                    "exclusive_features": owned.tolist(),
                    "top1_sel": [float(top1_l[f]) for f in owned],
                }

        print(f"  Exclusive features at L{l}: {n_excl_this_layer}")

        # track best SAE (most exclusive features)
        if n_excl_this_layer >= sum(
            len(v) for cls in classes
            for k, v in class_features[cls].items()
            if k == f"L{best_layer_idx:02d}"
        ):
            best_sae        = sae
            best_sel        = sel_l
            best_layer_idx  = l

    # aggregate selectivity (C, L, F)
    all_sel = np.stack(all_sel_layers, axis=1) if len(all_sel_layers) > 1 \
              else all_sel_layers[0][:, np.newaxis, :]

    np.savez_compressed(
        out_dir / f"02j_{stem}_sae_selectivity.npz",
        sel=all_sel, layers=np.array(layers), classes=np.array(classes),
    )

    with open(out_dir / f"02j_{stem}_sae_class_features.json", "w") as f:
        json.dump({"classes": classes,
                   "excl_per_class": excl_per_class,
                   "features": class_features}, f, indent=2)

    print(f"\nTotal exclusive SAE features per class:")
    total = 0
    for cls in classes:
        n = excl_per_class[cls]
        total += n
        if n > 0:
            print(f"  {cls:22s}  {n}")
    print(f"  Total: {total}")

    # overall excl mask (any layer)
    sel_agg  = all_sel.max(axis=1)  # (C, F)
    top1_agg = sel_agg.max(axis=0)
    top2_agg = np.sort(sel_agg, axis=0)[-2]
    excl_agg = (top1_agg > args.sel_thresh) & (top2_agg < args.excl_thresh)

    print("\nPlotting …")
    plot_loss_curves(all_losses, fmt, args.dpi, fig_dir)
    plot_feature_atlas(all_sel, classes, excl_agg, args.top_k, fmt, args.dpi, fig_dir)
    plot_exclusive_counts(excl_per_class, classes, args.excl_thresh,
                          fmt, args.dpi, fig_dir)

    print("Plotting SAE group heatmaps (all layers) …")
    plot_decoder_group_heatmap(
        out_dir, stem, all_sel, classes, layers, D,
        top_k_features=8,
        top_n_dims=40,
        sel_thresh=args.sel_thresh,
        fmt=fmt, dpi=args.dpi, fig_dir=fig_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
