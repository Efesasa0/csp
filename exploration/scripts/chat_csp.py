#!/usr/bin/env python3
"""Interactive terminal chat for the CSP model.

Default model: openai/circuit-sparsity
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import inspect
import os
import random
import sys
import types
from dataclasses import dataclass

# Work around restricted environments where Intel OpenMP shared memory is blocked.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_TORCH = None
_AUTO_TOKENIZER = None
_AUTO_MODEL = None


def _ensure_ml_libs():
    global _TORCH, _AUTO_TOKENIZER, _AUTO_MODEL
    if _TORCH is not None and _AUTO_TOKENIZER is not None and _AUTO_MODEL is not None:
        return _TORCH, _AUTO_TOKENIZER, _AUTO_MODEL

    try:
        import torch as torch_mod
        from transformers import AutoModelForCausalLM as model_cls
        from transformers import AutoTokenizer as tok_cls
    except Exception as exc:
        raise RuntimeError(
            "Failed to import torch/transformers. "
            "If you hit OpenMP SHM errors, run with: "
            "KMP_USE_SHM=0 OMP_NUM_THREADS=1 python scripts/chat_csp.py"
        ) from exc

    _TORCH = torch_mod
    _AUTO_TOKENIZER = tok_cls
    _AUTO_MODEL = model_cls
    return _TORCH, _AUTO_TOKENIZER, _AUTO_MODEL


@dataclass
class ChatConfig:
    model: str
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int | None


def _patch_circuit_sparsity_config(model_id: str) -> None:
    """Patch remote circuit_sparsity GPTConfig to ignore unknown kwargs.

    This mirrors the compatibility workaround used in notebooks.
    """
    from huggingface_hub import hf_hub_download

    tm_base = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/openai")
    tm_gpt_hits = glob.glob(os.path.join(tm_base, "*/gpt.py"))

    if tm_gpt_hits:
        gpt_path = tm_gpt_hits[0]
        hook_path = os.path.join(os.path.dirname(gpt_path), "hook_utils.py")
    else:
        gpt_path = hf_hub_download(model_id, "gpt.py")
        hook_path = hf_hub_download(model_id, "hook_utils.py")

    hf_dir = os.path.dirname(gpt_path)

    cs_pkg = types.ModuleType("circuit_sparsity")
    cs_pkg.__path__ = [hf_dir]
    cs_pkg.__package__ = "circuit_sparsity"
    sys.modules["circuit_sparsity"] = cs_pkg

    hook_spec = importlib.util.spec_from_file_location("circuit_sparsity.hook_utils", hook_path)
    if hook_spec is None or hook_spec.loader is None:
        raise RuntimeError("Failed to load circuit_sparsity.hook_utils")
    hook_mod = importlib.util.module_from_spec(hook_spec)
    hook_mod.__package__ = "circuit_sparsity"
    sys.modules["circuit_sparsity.hook_utils"] = hook_mod
    hook_spec.loader.exec_module(hook_mod)
    cs_pkg.hook_utils = hook_mod

    gpt_spec = importlib.util.spec_from_file_location("circuit_sparsity.gpt", gpt_path)
    if gpt_spec is None or gpt_spec.loader is None:
        raise RuntimeError("Failed to load circuit_sparsity.gpt")
    gpt_mod = importlib.util.module_from_spec(gpt_spec)
    gpt_mod.__package__ = "circuit_sparsity"
    sys.modules["circuit_sparsity.gpt"] = gpt_mod
    gpt_spec.loader.exec_module(gpt_mod)

    real_cfg = gpt_mod.GPTConfig
    valid_params = set(inspect.signature(real_cfg.__init__).parameters) - {"self"}

    class PatchedGPTConfig(real_cfg):
        def __init__(self, **kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in valid_params}
            super().__init__(**filtered)

    gpt_mod.GPTConfig = PatchedGPTConfig
    cs_pkg.gpt = gpt_mod


def _resolve_device(device_arg: str) -> str:
    torch, _, _ = _ensure_ml_libs()
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_model_and_tokenizer(cfg: ChatConfig):
    _, AutoTokenizer, AutoModelForCausalLM = _ensure_ml_libs()
    device = _resolve_device(cfg.device)

    # Always patch first, exactly as the notebooks do.
    _patch_circuit_sparsity_config(cfg.model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(cfg.model, trust_remote_code=True)
    model.to(device).eval()

    return tokenizer, model, device


def format_prompt(history: list[tuple[str, str]], user_text: str) -> str:
    # This is a completion model — just concatenate prior turns as plain text
    # and append the new user input so the model continues from there.
    parts: list[str] = []
    for _role, text in history:
        parts.append(text)
    parts.append(user_text)
    return " ".join(parts)


def generate_reply(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    torch, _, _ = _ensure_ml_libs()
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    # Use the model's native top-k generate (temperature=1 is greedy-ish at low temp).
    # top_p → approximate top_k: keep tokens covering ~top_p of the mass.
    top_k = max(1, int(top_p * model.circuit_model.config.vocab_size))
    effective_temp = temperature if temperature > 0 else 1e-8

    # circuit_model.generate returns the full sequence (prompt + new tokens).
    output_ids = model.circuit_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=effective_temp,
        top_k=top_k,
    )

    new_ids = output_ids[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    if eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id)]

    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text.strip() or "[empty response]"


def print_help(cfg: ChatConfig) -> None:
    print("Commands:")
    print("  /help   Show commands")
    print("  /reset  Clear conversation history")
    print("  /exit   Quit")
    print("Settings:")
    print(f"  model={cfg.model} device={cfg.device} max_new_tokens={cfg.max_new_tokens} temperature={cfg.temperature} top_p={cfg.top_p}")


def parse_args() -> ChatConfig:
    parser = argparse.ArgumentParser(description="Interactive chat with openai/circuit-sparsity")
    parser.add_argument("--model", default="openai/circuit-sparsity", help="Hugging Face model id")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    return ChatConfig(
        model=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


def main() -> int:
    cfg = parse_args()
    torch, _, _ = _ensure_ml_libs()

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    print("Loading model, this may take a while on first run...")
    tokenizer, model, resolved_device = load_model_and_tokenizer(cfg)
    print(f"Ready. Model={cfg.model} Device={resolved_device}")
    print("Type /help for commands.")

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_text = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_text:
            continue
        if user_text == "/exit":
            print("Exiting.")
            return 0
        if user_text == "/reset":
            history.clear()
            print("History cleared.")
            continue
        if user_text == "/help":
            print_help(cfg)
            continue

        prompt = format_prompt(history, user_text)
        try:
            reply = generate_reply(
                model=model,
                tokenizer=tokenizer,
                device=resolved_device,
                prompt=prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            continue

        print(f"Model > {reply}")
        history.append(("user", user_text))
        history.append(("assistant", reply))


if __name__ == "__main__":
    raise SystemExit(main())
