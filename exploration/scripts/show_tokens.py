#!/usr/bin/env python3
"""Show the vocabulary of the openai/circuit-sparsity tokenizer.

Usage:
    python scripts/show_tokens.py                   # full vocab table
    python scripts/show_tokens.py --search def      # grep for a string
    python scripts/show_tokens.py --ids 0 1 42 99   # decode specific ids
    python scripts/show_tokens.py --encode "def f"  # encode a string
    python scripts/show_tokens.py --limit 50        # show first N tokens
"""
from __future__ import annotations

import argparse


MODEL_ID = "openai/circuit-sparsity"


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("[error] transformers not installed: pip install transformers")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _repr(s: str) -> str:
    """Human-readable token — show escape sequences for whitespace."""
    return repr(s)[1:-1]  # strip outer quotes from repr


def cmd_vocab(tok, limit: int | None, search: str | None) -> None:
    vocab: dict[str, int] = tok.get_vocab()
    # sort by id
    by_id = sorted(vocab.items(), key=lambda x: x[1])

    if search:
        by_id = [(t, i) for t, i in by_id if search.lower() in t.lower()]
        print(f"Tokens matching {search!r}: {len(by_id)} found\n")

    if limit is not None:
        by_id = by_id[:limit]

    col_w = max((len(_repr(t)) for t, _ in by_id), default=10)
    col_w = min(col_w, 40)

    print(f"{'ID':>6}  {'Token':<{col_w}}  Decoded bytes")
    print(f"{'─'*6}  {'─'*col_w}  {'─'*30}")
    for token_str, token_id in by_id:
        decoded = tok.decode([token_id])
        print(f"{token_id:>6}  {_repr(token_str):<{col_w}}  {_repr(decoded)}")

    print(f"\nShowing {len(by_id)} / {len(vocab)} tokens")


def cmd_ids(tok, ids: list[int]) -> None:
    print(f"Decoding {len(ids)} token id(s):\n")
    for tid in ids:
        try:
            single = tok.decode([tid])
            print(f"  {tid:>6}  →  {_repr(single)}")
        except Exception as exc:
            print(f"  {tid:>6}  →  [error: {exc}]")


def cmd_encode(tok, text: str) -> None:
    ids = tok.encode(text, add_special_tokens=False)
    print(f"Input : {text!r}")
    print(f"IDs   : {ids}")
    print(f"Tokens:")
    for tid in ids:
        print(f"  {tid:>6}  →  {_repr(tok.decode([tid]))}")


def cmd_summary(tok) -> None:
    vocab = tok.get_vocab()
    print(f"Model          : {MODEL_ID}")
    print(f"Vocab size     : {len(vocab)}")
    print(f"EOS token      : {tok.eos_token!r}  (id={tok.eos_token_id})")
    print(f"PAD token      : {tok.pad_token!r}  (id={tok.pad_token_id})")
    print(f"BOS token      : {tok.bos_token!r}  (id={tok.bos_token_id})")
    print(f"UNK token      : {tok.unk_token!r}  (id={tok.unk_token_id})")
    print(f"Padding side   : {tok.padding_side}")
    print(f"Tokenizer type : {type(tok).__name__}")

    # character/byte coverage
    all_tokens = sorted(vocab.items(), key=lambda x: x[1])
    lengths = [len(t) for t, _ in all_tokens]
    print(f"\nToken length   : min={min(lengths)}  max={max(lengths)}  "
          f"mean={sum(lengths)/len(lengths):.1f}")

    # show first and last 5
    print("\nFirst 5 tokens:")
    for t, i in all_tokens[:5]:
        print(f"  {i:>6}  {_repr(t)}")
    print("Last 5 tokens:")
    for t, i in all_tokens[-5:]:
        print(f"  {i:>6}  {_repr(t)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect openai/circuit-sparsity tokenizer")
    p.add_argument("--search", metavar="STR",
                   help="Filter vocab to tokens containing STR")
    p.add_argument("--ids", nargs="+", type=int, metavar="ID",
                   help="Decode specific token IDs")
    p.add_argument("--encode", metavar="TEXT",
                   help="Encode TEXT and show token breakdown")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Show only first N tokens (default: all)")
    p.add_argument("--summary", action="store_true",
                   help="Show tokenizer summary only (no full vocab)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading tokenizer from {MODEL_ID}...")
    tok = load_tokenizer()
    print()

    if args.ids:
        cmd_ids(tok, args.ids)
    elif args.encode:
        cmd_encode(tok, args.encode)
    elif args.summary:
        cmd_summary(tok)
    else:
        cmd_summary(tok)
        print()
        cmd_vocab(tok, limit=args.limit, search=args.search)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
