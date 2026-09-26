"""Configuration, path resolution, and environment setup."""

import os
import sys
import json
import glob
import types
import inspect
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_project_root():
    """The pipeline directory (parent of core/); all relative paths resolve against it."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROJECT_ROOT = _find_project_root()


def resolve_path(rel):
    """Resolve a path relative to the project root."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(PROJECT_ROOT, rel)


def load_config(path=None):
    if path is None:
        # config.json lives in ATLAS_Pipeline_New/, one level up from core/
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json",
        )
        if not os.path.exists(path):
            # Fallback: look next to caller
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config.json"
            )
    with open(path, "r") as f:
        return json.load(f)


def data_path(*parts):
    """Path under the pipeline's data/ directory."""
    return os.path.join(resolve_path("data"), *parts)


def ensure_dirs(cfg):
    for d in cfg["paths"]["dirs"].values():
        os.makedirs(resolve_path(d), exist_ok=True)


def get_path(cfg, category, filename):
    return os.path.join(resolve_path(cfg["paths"]["dirs"][category]), filename)


def patch_environment():
    """Download and patch the circuit-sparsity module from HuggingFace."""
    from huggingface_hub import hf_hub_download

    try:
        base = os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/openai"
        )
        gpt = glob.glob(os.path.join(base, "*/gpt.py"))[0]
        hook = os.path.join(os.path.dirname(gpt), "hook_utils.py")
    except Exception:
        gpt = hf_hub_download("openai/circuit-sparsity", "gpt.py")
        hook = hf_hub_download("openai/circuit-sparsity", "hook_utils.py")

    pkg = types.ModuleType("circuit_sparsity")
    pkg.__path__ = [os.path.dirname(gpt)]
    sys.modules["circuit_sparsity"] = pkg

    def load(name, path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("circuit_sparsity.hook_utils", hook)
    gpt_mod = load("circuit_sparsity.gpt", gpt)

    _Real = gpt_mod.GPTConfig
    valid = set(inspect.signature(_Real.__init__).parameters) - {"self"}

    class Safe(_Real):
        def __init__(self, **kwargs):
            super().__init__(**{k: v for k, v in kwargs.items() if k in valid})

    gpt_mod.GPTConfig = Safe
