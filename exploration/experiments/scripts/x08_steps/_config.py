"""Shared configuration dataclass for all X08 pipeline steps."""
from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class StepConfig:
    stem: str              # e.g. "contrastive_stubs"
    dir: Path              # output directory (absolute)
    plot: bool = True

    # Step 01
    input_path: Path | None = None
    device: str = "cpu"
    batch_size: int = 8
    max_tokens: int = 128
    skip_edge_graph: bool = False

    # Step 02
    threshold: float = 0.05
    purity: float = 0.70
    plot_layer: int = -1

    # Steps 03, 04
    n_perm: int = 1000
    alpha: float = 0.05

    # Steps 03, 05
    all_layers: bool = True
    all_concepts: bool = True

    # Step 05
    n_splits: int = 5

    # Step 06
    max_pairs: int = 300
    circuit_threshold: float = 0.80

    # Step 07
    input_json: Path | None = None
    ref_stem: str | None = None
    n_quartiles: int = 4
