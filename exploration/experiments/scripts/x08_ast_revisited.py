#!/usr/bin/env python
"""X08 — Thin wrapper that runs MaaT's AST-Revisited pipeline (steps 01–07).

Calls Matt's scripts in-place via subprocess, redirecting all outputs to
``experiments/outputs/x08/`` (or wherever ``--out`` points).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATT = ROOT / "COMP0087_CW" / "Code" / "MaaTt" / "AST-Revisited"
STEM = "contrastive_stubs"

ALL_STEPS = ["01", "02", "03", "04", "05", "06", "07"]


def _resolve_stubs(stubs_arg: str) -> Path:
    """Return an existing stubs JSON/JSONL path, falling back to Matt's copy."""
    p = Path(stubs_arg)
    if p.exists():
        return p.resolve()
    matt_copy = MATT / "contrastive_stubs.json"
    if matt_copy.exists():
        return matt_copy.resolve()
    sys.exit(f"ERROR: stubs not found at {stubs_arg} or {matt_copy}")


def _run(label: str, cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n{'=' * 54}")
    print(f"  {label}")
    print(f"{'=' * 54}")
    print(f"  cmd: {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.exit(f"FAILED ({proc.returncode}) after {elapsed:.1f}s")
    print(f"  Done in {elapsed:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="X08 — Run MaaT AST-Revisited pipeline")
    ap.add_argument("--stubs", default="data/contrastive_stubs.json",
                    help="Path to contrastive stubs JSON/JSONL (default: data/contrastive_stubs.json)")
    ap.add_argument("--out", default="experiments/outputs/x08",
                    help="Output directory (default: experiments/outputs/x08)")
    ap.add_argument("--device", default="auto",
                    help="Torch device for step 01 (default: auto)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for step 01 (default: 8)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="Max tokens for step 01 (default: 128)")
    ap.add_argument("--steps", default="all",
                    help="Comma-separated steps to run, e.g. '01,02,03' (default: all)")
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="Permutations for RSA / significance tests (default: 1000)")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="Generate plots (default: on)")
    ap.add_argument("--all-layers", action=argparse.BooleanOptionalAction, default=True,
                    help="Run all-layer analyses where supported (default: on)")
    args = ap.parse_args()

    # Resolve paths
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stubs = _resolve_stubs(args.stubs)
    python = sys.executable

    # Determine which steps to run
    if args.steps.strip().lower() == "all":
        steps = ALL_STEPS
    else:
        steps = [s.strip().zfill(2) for s in args.steps.split(",")]
        for s in steps:
            if s not in ALL_STEPS:
                sys.exit(f"Unknown step: {s}. Valid: {ALL_STEPS}")

    if not MATT.exists():
        sys.exit(f"ERROR: MaaT directory not found: {MATT}")

    print(f"Steps:  {steps}")
    print(f"Stubs:  {stubs}")
    print(f"Output: {out}")
    print(f"Device: {args.device}")

    # Step 01: extraction — writes outputs next to input file,
    # so we copy stubs into OUT first.
    if "01" in steps:
        local_stubs = out / "contrastive_stubs.json"
        if not local_stubs.exists() or local_stubs.resolve() != stubs:
            shutil.copy2(stubs, local_stubs)
        cmd = [
            python, str(MATT / "01_extraction.py"),
            "--input", str(local_stubs),
            "--device", args.device,
            "--batch", str(args.batch_size),
            "--max_tokens", str(args.max_tokens),
        ]
        _run("Step 01 — Activation Extraction", cmd)

    # Steps 02–07: all accept --stem and --dir
    step_configs: dict[str, tuple[str, str, list[str]]] = {
        "02": ("02_variance_partition.py", "Step 02 — Variance Partition", []),
        "03": ("03_projection.py", "Step 03 — Projection & RSA", []),
        "04": ("04_additivity.py", "Step 04 — Additivity Testing", []),
        "05": ("05_probing.py", "Step 05 — Probing", []),
        "06": ("06_causal.py", "Step 06 — Causal Analysis", []),
        "07": ("07_loss_analysis.py", "Step 07 — Loss Analysis", []),
    }

    for step_id in steps:
        if step_id == "01":
            continue
        if step_id not in step_configs:
            continue

        script_name, label, extra = step_configs[step_id]
        cmd = [
            python, str(MATT / script_name),
            "--stem", STEM,
            "--dir", str(out),
        ]

        if args.plot:
            cmd.append("--plot")

        # Step-specific args
        if step_id == "03":
            cmd += ["--n_perm", str(args.n_perm)]
            if args.all_layers:
                cmd.append("--all_layers")
        elif step_id == "04":
            cmd += ["--n_perm", str(args.n_perm)]
        elif step_id == "05":
            if args.all_layers:
                cmd.append("--all_concepts")
        elif step_id == "07":
            local_stubs = out / "contrastive_stubs.json"
            if local_stubs.exists():
                cmd += ["--input_json", str(local_stubs)]

        cmd += extra
        _run(label, cmd)

    print(f"\nAll requested steps complete. Outputs in: {out}")


if __name__ == "__main__":
    main()
