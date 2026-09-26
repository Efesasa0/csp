"""
Phase 4 — Compare old-trained vs ideal-trained SAE summaries.

Reads:
  data/study_v2/reports/summary_old_saes.json
  data/study_v2/reports/summary_ideal.json

Outputs:
  data/study_v2/reports/comparison.json   (per-layer aggregates + diffs)
  Also prints a human-readable summary table.
"""
import os, sys, json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import resolve_path, data_path

REPORTS_DIR = data_path("study_v2/reports")


def agg(summary):
    """For each (layer, site), count validated-specific circuits."""
    out = {}
    for label, rows in summary.items():
        if label.startswith("_"):
            continue
        # Validated = AST-specific + replicates on old + p < 0.05 on target vs other
        kinds = Counter()
        effects = []
        valid = 0
        for r in rows:
            kinds[r["classification"]] += 1
            if r["cohens_d"] is not None:
                effects.append(r["cohens_d"])
            p = r.get("p_vs_other")
            if (r["classification"] == "AST-specific"
                    and r.get("replicates_old")
                    and p is not None and p < 0.05):
                valid += 1
        out[label] = {
            "n_nodes": len(rows),
            "validated_ast_specific": valid,
            "by_classification": dict(kinds),
            "mean_cohens_d": (sum(effects)/len(effects)) if effects else None,
        }
    return out


def main():
    old = json.load(open(os.path.join(REPORTS_DIR, "summary_old_saes.json")))
    ideal = json.load(open(os.path.join(REPORTS_DIR, "summary_ideal.json")))

    ag_old = agg(old)
    ag_ideal = agg(ideal)

    comparison = {"old_saes": ag_old, "ideal": ag_ideal, "delta": {}}
    for label in ag_old:
        o, i = ag_old[label], ag_ideal.get(label, {})
        comparison["delta"][label] = {
            "validated_ast_specific": i.get("validated_ast_specific", 0) - o["validated_ast_specific"],
            "mean_cohens_d": (i.get("mean_cohens_d") or 0) - (o.get("mean_cohens_d") or 0),
        }

    out = os.path.join(REPORTS_DIR, "comparison.json")
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"{'layer/site':<15} {'old valid':>10} {'ideal valid':>12} {'Δ':>5}  {'old d':>8} {'ideal d':>8}")
    print("-" * 65)
    for label in sorted(ag_old):
        o = ag_old[label]; i = ag_ideal.get(label, {})
        d = comparison["delta"][label]["validated_ast_specific"]
        od = o.get("mean_cohens_d") or 0
        id_ = i.get("mean_cohens_d") or 0
        print(f"{label:<15} {o['validated_ast_specific']:>10d} {i.get('validated_ast_specific',0):>12d} {d:>+5d}  {od:>8.3f} {id_:>8.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
