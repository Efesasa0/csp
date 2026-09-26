"""X08 step registry — lazy imports so ``--steps 02`` doesn't load torch."""
from __future__ import annotations

ALL_STEP_IDS = ["01", "02", "03", "04", "05", "06", "07"]

STEP_LABELS = {
    "01": "Step 01 — Activation Extraction",
    "02": "Step 02 — Variance Partition",
    "03": "Step 03 — Projection & RSA",
    "04": "Step 04 — Additivity Testing",
    "05": "Step 05 — Probing",
    "06": "Step 06 — Causal Analysis",
    "07": "Step 07 — Loss Analysis",
}


def get_runner(step_id: str):
    """Lazy import and return the ``run()`` function for a step."""
    if step_id == "01":
        from .step01_extraction import run
    elif step_id == "02":
        from .step02_variance_partition import run
    elif step_id == "03":
        from .step03_projection import run
    elif step_id == "04":
        from .step04_additivity import run
    elif step_id == "05":
        from .step05_probing import run
    elif step_id == "06":
        from .step06_causal import run
    elif step_id == "07":
        from .step07_loss_analysis import run
    else:
        raise ValueError(f"Unknown step: {step_id}")
    return run
