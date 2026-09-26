#!/usr/bin/env python3
"""Lightweight runtime environment checks for extraction scripts."""

from __future__ import annotations

import sys
from importlib import metadata


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _major_minor(v: str) -> tuple[int, int]:
    core = v.split("+", 1)[0]
    parts = core.split(".")
    return int(parts[0]), int(parts[1])


def main() -> int:
    issues: list[str] = []

    versions = {
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "torchvision": _version("torchvision"),
        "transformers": _version("transformers"),
        "accelerate": _version("accelerate"),
        "pandas": _version("pandas"),
        "scikit-learn": _version("scikit-learn"),
        "numexpr": _version("numexpr"),
        "bottleneck": _version("bottleneck"),
    }

    print("Detected package versions:")
    for k, v in versions.items():
        print(f"  {k:13s} {v or '(missing)'}")

    torch_v = versions["torch"]
    tv_v = versions["torchvision"]
    if torch_v and tv_v:
        t_mm = _major_minor(torch_v)
        tv_mm = _major_minor(tv_v)
        expected = {
            (2, 4): (0, 19),
            (2, 5): (0, 20),
            (2, 6): (0, 21),
        }
        want = expected.get(t_mm)
        if want and tv_mm != want:
            issues.append(
                f"torch {torch_v} is usually paired with torchvision {want[0]}.{want[1]}.* "
                f"(found {tv_v})."
            )

    np_v = versions["numpy"]
    if np_v:
        np_mm = _major_minor(np_v)
        if np_mm[0] >= 2:
            for dep in ["numexpr", "bottleneck"]:
                dv = versions[dep]
                if dv:
                    d_mm = _major_minor(dv)
                    if dep == "numexpr" and d_mm < (2, 10):
                        issues.append(f"{dep} {dv} may be too old for NumPy {np_v}.")
                    if dep == "bottleneck" and d_mm < (1, 4):
                        issues.append(f"{dep} {dv} may be too old for NumPy {np_v}.")

    if not versions["accelerate"]:
        issues.append("accelerate is missing (needed by some transformers loading paths).")

    print()
    if issues:
        print("Potential compatibility issues:")
        for i in issues:
            print(f"  - {i}")
        print("\nSuggested fix: pip install -r requirements.cuda121-py310.txt")
        return 1

    print("No obvious compatibility mismatches detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
