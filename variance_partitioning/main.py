"""
main.py — Run the full constructor-stubs pipeline end-to-end.

Usage (from submission/ directory):
    python main.py                        # defaults: batch=1024, device=auto
    python main.py --batch 512 --device cuda
    python main.py --skip_extraction      # if data/ already populated

Outputs land in data/ and data/figures/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE    = Path(__file__).parent           # submission/
SCRIPTS = HERE / "scripts"
DATA    = HERE / "data"
STEM    = "constructor_stubs"
JSONL   = HERE / f"{STEM}.jsonl"



def run(cmd: list[str], cwd: Path = SCRIPTS) -> None:
    """Run a command, raising on failure."""
    print(f"\n{'='*60}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"  cwd: {cwd}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + [str(c) for c in cmd], cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Step failed (exit {result.returncode}). Aborting.")


def main() -> None:
    p = argparse.ArgumentParser(description="Constructor-stubs full pipeline")
    p.add_argument("--batch",            type=int, default=1024,
                   help="Batch size for activation extraction (default: 1024)")
    p.add_argument("--device",           default="auto",
                   help="Device for extraction: auto | cpu | cuda | cuda:N")
    p.add_argument("--skip_extraction",  action="store_true",
                   help="Skip step 1 (use if data/ is already populated)")
    p.add_argument("--skip_vp",          action="store_true",
                   help="Skip step 2 (variance partitioning)")
    args = p.parse_args()

    DATA.mkdir(exist_ok=True)

    # ── Step 1: extract activations ───────────────────────────────────────────
    if not args.skip_extraction:
        print("\n[Step 1] Extracting activations …")
        run([
            "01_extraction.py",
            "--input",   JSONL,
            "--out_dir", DATA,
            "--batch",   args.batch,
            "--device",  args.device,
        ])

    # ── Step 2: variance partitioning ─────────────────────────────────────────
    if not args.skip_vp:
        print("\n[Step 2] Variance partitioning …")
        run([
            "02_variance_partition.py",
            "--stem",    STEM,
            "--in_dir",  DATA,
            "--out_dir", DATA,
        ])

    # ── Step 3: MLP neuron selectivity ────────────────────────────────────────
    print("\n[Step 3] Computing MLP neuron selectivity …")
    run([
        "02g_class_neurons.py",
        "--stem",    STEM,
        "--in_dir",  DATA,
        "--out_dir", DATA,
    ])

    # ── Step 3b: verify monosemantic MLP neurons ──────────────────────────────
    print("\n[Step 3b] Verifying monosemantic MLP neurons …")
    run(["scripts/check_neurons.py", "--in_dir", DATA], cwd=HERE)

    # ── Step 3c: polysemanticity figures (Assert spotlight) ───────────────────
    print("\n[Step 3c] Polysemanticity figures …")
    run([
        "02h_polysemanticity.py",
        "--stem",    STEM,
        "--in_dir",  DATA,
        "--out_dir", DATA,
    ])

    # ── Step 4: attention head selectivity ────────────────────────────────────
    print("\n[Step 4] Computing attention head selectivity …")
    run([
        "02l_head_selectivity.py",
        "--stem",    STEM,
        "--in_dir",  DATA,
        "--out_dir", DATA,
    ])

    # ── Step 5: head diagonal-grid figure ─────────────────────────────────────
    print("\n[Step 5] Generating head ownership grid …")
    run([
        "02m_head_viz.py",
        "--stem",    STEM,
        "--in_dir",  DATA,
        "--out_dir", DATA,
    ])

    # ── Step 6: residual stream grouping (k=14) ───────────────────────────────
    print("\n[Step 6] Residual stream group selectivity (k=14) …")
    run([
        "02p_group_selectivity.py",
        "--stem",   STEM,
        "--in_dir", DATA,
        "--k_list", "14",
    ])

    # ── Step 7: AST similarity graph ──────────────────────────────────────────
    print("\n[Step 7] Building AST similarity graph …")
    run([
        "02f_ast_graph.py",
        "--stem",   STEM,
        "--in_dir", DATA,
    ])

    # ── Step 8: edge-bundling interactive ─────────────────────────────────────
    print("\n[Step 8] Building edge-bundling interactive figure …")
    run([
        "03n_circuit_bundling_interactive.py",
        "--stem",   STEM,
        "--in_dir", DATA,
        "--format", "html",
    ])

    print("\n" + "="*60)
    print("  Pipeline complete.  Key outputs:")
    figures = [
        DATA / "figures/polysemanticity/assert_spotlight.pdf",
        DATA / "figures/heads/head_all_pure_grid_diagonal.pdf",
        DATA / "figures/group_selectivity/residual_group_heatmap_k14.pdf",
        DATA / "figures/ast_graph/ast_graph_all_layers.html",
        DATA / "figures/head_neuron_bundling/heads_edge_bundling_interactive.html",
    ]
    for f in figures:
        status = "OK" if f.exists() else "MISSING"
        print(f"  [{status}]  {f.relative_to(HERE)}")
    print("="*60)


if __name__ == "__main__":
    main()
