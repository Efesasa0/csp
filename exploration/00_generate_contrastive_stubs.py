"""
00_generate_contrastive_stubs.py

Generates a contrastive stub dataset for AST x builtin mechanistic
interpretability using OpenAI circuit_sparsity residual-stream analysis.

Design
------
  - 29 AST nodes, 55 builtins, 1405 (AST, builtin) pairs (full cross-coverage)
  - 6 variable-name variations per (AST, builtin, template) triple
  - AST baseline stubs  — pure AST construct, NO named builtin call
  - Builtin baseline stubs — builtin in neutral Assign context, minimal AST
  - Proxy variants — semantically equivalent prompts without the builtin token

Baseline rationale
------------------
To test whether AST features and builtin features are orthogonal, compute:

    cos_sim(act(AST, builtin),  act(AST_baseline) + act(builtin_baseline))

If cosine similarity is high -> additive / independent representations.
If low -> there is an interaction circuit specific to the (AST, builtin) pair.

Output: JSONL + Parquet, one record per line.
Field names match small_40x50x50_validated_prompts.json exactly:
  ast_node, builtin_obj, variation_id, prompt_text, sequence_loss,
  token_length, ast_verified, prompt_id
Extra fields preserved for contrastive analysis:
  variant_type, category, ast_group, builtin_group, note,
  ast_keyword_pos, builtin_token_pos

Usage
-----
  # full dataset (default)
  python 00_generate_contrastive_stubs.py

  # custom output name and size
  python 00_generate_contrastive_stubs.py --output my_stubs.json --n 1000

  # just the explicit stubs, 500 samples
  python 00_generate_contrastive_stubs.py --n 500 --output explicit_only.json
"""

from __future__ import annotations

import ast as _ast
import json
from collections import defaultdict
from pathlib import Path

# ── optional tiktoken for token-position annotation ───────────────────────────
try:
    import tiktoken
    _ENC = tiktoken.encoding_for_model("gpt-2")
    def _tok(t):           return _ENC.encode(t)
    def _find(toks, word):
        for pfx in (word, " " + word):
            target = _ENC.encode(pfx)
            for i in range(len(toks) - len(target) + 1):
                if toks[i:i+len(target)] == target:
                    return i
        return None
except ImportError:
    _ENC = None
    def _tok(t):           return []
    def _find(toks, word): return None

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_stubs: list[dict] = []

_AST_KEYWORDS: dict[str, str] = {
    "For": "for", "While": "while", "If": "if",
    "ListComp": "for", "DictComp": "for", "SetComp": "for",
    "GeneratorExp": "for",
    "Lambda": "lambda", "Return": "return", "Assert": "assert",
    "Raise": "raise", "With": "with", "Try": "try",
    "FunctionDef": "def", "AsyncFunctionDef": "async",
    "ClassDef": "class", "Yield": "yield", "YieldFrom": "yield",
    "AugAssign": "+=", "AnnAssign": ":", "Delete": "del",
    "IfExp": "if", "BoolOp": "and", "Compare": ">",
    "Subscript": "[", "Break": "break", "Continue": "continue",
    "Global": "global", "Import": "import",
}


def _register(
    ast_node: str,
    builtin:  str,
    text:     str,
    variant:  str = "explicit",   # explicit | proxy | baseline_ast | baseline_builtin
    tmpl_id:  int = 0,
    note:     str = "",
) -> None:
    toks = _tok(text)
    _stubs.append({
        # ── fields matching small_40x50x50_validated_prompts.json ─────────────
        "ast_node":      ast_node,
        "builtin_obj":   builtin,
        "variation_id":  None,           # assigned globally in _build()
        "prompt_text":   text,
        "sequence_loss": None,           # filled in after model inference
        "token_length":  len(toks) if toks else len(text.split()),
        "ast_verified":  _verify(text, ast_node),
        "prompt_id":     None,           # assigned globally in _build()
        # ── extra fields for contrastive analysis ─────────────────────────────
        "variant_type":     variant,
        "category":         f"{ast_node}__{builtin}",
        "ast_group":        f"ast__{ast_node}",
        "builtin_group":    f"builtin__{builtin}",
        "note":             note,
        "ast_keyword_pos":  _find(toks, _AST_KEYWORDS.get(ast_node, "")),
        "builtin_token_pos":_find(toks, builtin) if variant == "explicit" else None,
        # ── internal — removed before output ──────────────────────────────────
        "_valid":           _valid(text),
        "_tmpl_id":         tmpl_id,
    })


def _valid(t: str) -> bool:
    try: _ast.parse(t); return True
    except SyntaxError: return False

def _verify(t: str, node: str) -> bool:
    try:
        tree = _ast.parse(t)
        return any(type(n).__name__ == node for n in _ast.walk(tree))
    except SyntaxError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Variable-name variation sets
# ─────────────────────────────────────────────────────────────────────────────
# (data_var, result_var, iter_var)
_VARS = [
    ("data",     "result",  "item"),
    ("items",    "output",  "element"),
    ("values",   "out",     "x"),
    ("elements", "res",     "entry"),
    ("nums",     "rv",      "val"),
    ("seq",      "acc",     "node"),
]


def _expand(template: str, D: str, R: str, I: str, B: str) -> str:
    """Substitute placeholders.  Returns None if result is syntactically invalid."""
    return (template
            .replace("{D}", D)
            .replace("{R}", R)
            .replace("{I}", I)
            .replace("{B}", B))


# ─────────────────────────────────────────────────────────────────────────────
# Builtin call expressions per role
# ─────────────────────────────────────────────────────────────────────────────
# {D} = data variable,  {X} = element variable (loop var)

_ITER_BUILTINS: list[tuple[str, str]] = [
    ("range",     "range(10)"),
    ("range",     "range(len({D}))"),
    ("list",      "list({D})"),
    ("sorted",    "sorted({D})"),
    ("enumerate", "enumerate({D})"),
    ("zip",       "zip({D}, {D})"),
    ("map",       "map(str, {D})"),
    ("filter",    "filter(None, {D})"),
    ("reversed",  "reversed({D})"),
    ("tuple",     "tuple({D})"),
    ("frozenset", "frozenset({D})"),
    ("set",       "set({D})"),
    ("iter",      "iter({D})"),
    ("bytes",     "bytes({D})"),
]

_COND_BUILTINS: list[tuple[str, str]] = [
    ("len",        "len({D})"),
    ("any",        "any({D})"),
    ("all",        "all({D})"),
    ("bool",       "bool({D})"),
    ("isinstance", "isinstance({D}, list)"),
    ("issubclass", "issubclass(type({D}), list)"),
    ("hasattr",    "hasattr({D}, '__len__')"),
    ("callable",   "callable({D})"),
    ("sum",        "sum({D})"),
    ("max",        "max({D})"),
]

_TRANSFORM_BUILTINS: list[tuple[str, str]] = [
    ("str",      "str({X})"),
    ("int",      "int({X})"),
    ("float",    "float({X})"),
    ("abs",      "abs({X})"),
    ("repr",     "repr({X})"),
    ("bool",     "bool({X})"),
    ("len",      "len({X})"),
    ("sorted",   "sorted({X})"),
    ("list",     "list({X})"),
    ("tuple",    "tuple({X})"),
    ("set",      "set({X})"),
    ("round",    "round({X}, 2)"),
    ("hash",     "hash({X})"),
    ("type",     "type({X})"),
    ("id",       "id({X})"),
    ("frozenset","frozenset({X})"),
]

_SCALAR_BUILTINS: list[tuple[str, str]] = [
    ("len",   "len({D})"),
    ("sum",   "sum({D})"),
    ("max",   "max({D})"),
    ("min",   "min({D})"),
    ("abs",   "abs({D}[0])"),
    ("round", "round({D}[0], 2)"),
    ("hash",  "hash({D}[0])"),
    ("id",    "id({D})"),
    ("int",   "int({D}[0])"),
    ("float", "float({D}[0])"),
]

_BODY_BUILTINS: list[tuple[str, str]] = [
    ("len",       "len({D})"),
    ("sorted",    "sorted({D})"),
    ("list",      "list({D})"),
    ("tuple",     "tuple({D})"),
    ("set",       "set({D})"),
    ("dict",      "dict({D})"),
    ("str",       "str({D})"),
    ("int",       "int({D}[0])"),
    ("float",     "float({D}[0])"),
    ("sum",       "sum({D})"),
    ("max",       "max({D})"),
    ("min",       "min({D})"),
    ("any",       "any({D})"),
    ("all",       "all({D})"),
    ("bool",      "bool({D})"),
    ("abs",       "abs({D}[0])"),
    ("range",     "range(len({D}))"),
    ("enumerate", "enumerate({D})"),
    ("zip",       "zip({D}, {D})"),
    ("reversed",  "reversed({D})"),
    ("print",     "print({D})"),
    ("repr",      "repr({D})"),
    ("isinstance","isinstance({D}, list)"),
    ("hasattr",   "hasattr({D}, '__len__')"),
    ("callable",  "callable({D})"),
    ("hash",      "hash({D}[0])"),
    ("id",        "id({D})"),
    ("type",      "type({D})"),
    ("round",     "round({D}[0], 2)"),
    ("frozenset", "frozenset({D})"),
    ("iter",      "iter({D})"),
    ("next",      "next(iter({D}))"),
    ("getattr",   "getattr({D}, '__class__')"),
    ("complex",   "complex({D}[0])"),
    ("memoryview","memoryview(bytes({D}))"),
    ("bytearray", "bytearray(bytes({D}))"),
    ("bytes",     "bytes({D})"),
    ("map",       "map(str, {D})"),
    ("filter",    "filter(None, {D})"),
]

_EXCEPTION_BUILTINS: list[tuple[str, str]] = [
    ("ValueError",          "ValueError('invalid input')"),
    ("TypeError",           "TypeError('wrong type')"),
    ("RuntimeError",        "RuntimeError('operation failed')"),
    ("KeyError",            "KeyError('missing key')"),
    ("IndexError",          "IndexError('out of bounds')"),
    ("NotImplementedError", "NotImplementedError('not implemented')"),
    ("OSError",             "OSError('file error')"),
    ("AttributeError",      "AttributeError('missing attr')"),
    ("StopIteration",       "StopIteration()"),
]

_CONTEXT_BUILTINS: list[tuple[str, str]] = [
    ("open", "open({D})"),
    ("open", "open({D}, 'r')"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core generation helper
# ─────────────────────────────────────────────────────────────────────────────

def _gen(
    ast_node: str,
    role_builtins: list[tuple[str, str]],
    templates: list[str],         # use {D},{R},{I},{B}; {B} uses {D} or {X}={I}
    element_is_X: bool = False,   # True for transform role: replace {X} with iter var
) -> None:
    """Generate stubs for all (builtin, template, variation) combinations."""
    for builtin_name, builtin_expr_tmpl in role_builtins:
        for tmpl_idx, template in enumerate(templates):
            for var_idx, (D, R, I) in enumerate(_VARS):
                # Build the builtin call expression for this variation
                B = builtin_expr_tmpl.replace("{D}", D).replace("{X}", I)
                text = _expand(template, D, R, I, B)
                if _valid(text):
                    _register(
                        ast_node=ast_node,
                        builtin=builtin_name,
                        text=text,
                        variant="explicit",
                        tmpl_id=tmpl_idx * 100 + var_idx,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Main stubs — explicit (AST, builtin) pairs
# ─────────────────────────────────────────────────────────────────────────────

# ── FOR ───────────────────────────────────────────────────────────────────────
_gen("For", _ITER_BUILTINS, [
    "for {I} in {B}:\n    {R} = {I}",
    "for {I} in {B}:\n    pass",
    "{R} = []\nfor {I} in {B}:\n    {R}.append({I})",
])

# ── WHILE ─────────────────────────────────────────────────────────────────────
_gen("While", _COND_BUILTINS, [
    "{D} = [1, 2, 3]\nwhile {B}:\n    {D} = {D}[1:]",
    "{D} = [1, 2, 3]\nwhile {B}:\n    {D}.pop()",
    "{D} = [1, 2, 3]\nflag = True\nwhile flag and {B}:\n    {D} = {D}[1:]\n    flag = bool({D})",
])

# ── IF ────────────────────────────────────────────────────────────────────────
_gen("If", _COND_BUILTINS, [
    "if {B}:\n    pass",
    "if {B}:\n    {R} = True\nelse:\n    {R} = False",
    "{R} = None\nif {B}:\n    {R} = {D}",
])

# ── LISTCOMP ──────────────────────────────────────────────────────────────────
_gen("ListComp", _TRANSFORM_BUILTINS, [
    "{R} = [{B} for {I} in {D}]",
    "{R} = [{B} for {I} in {D} if {I}]",
    "{R} = [{B} for {I} in {D}[:5]]",
])

# ── DICTCOMP ──────────────────────────────────────────────────────────────────
_gen("DictComp", _TRANSFORM_BUILTINS, [
    "{R} = {{k: {B} for k, {I} in enumerate({D})}}",
    "{R} = {{str(k): {B} for k, {I} in enumerate({D})}}",
])

# ── SETCOMP ───────────────────────────────────────────────────────────────────
_gen("SetComp", _TRANSFORM_BUILTINS, [
    "{R} = {{{B} for {I} in {D}}}",
    "{R} = {{{B} for {I} in {D} if {I}}}",
])

# ── GENERATOREXP ──────────────────────────────────────────────────────────────
_gen("GeneratorExp", _TRANSFORM_BUILTINS, [
    "{R} = list({B} for {I} in {D})",
    "{R} = tuple({B} for {I} in {D})",
    "{R} = sum({B} for {I} in {D})",  # only works for numeric transforms
])

# ── LAMBDA ────────────────────────────────────────────────────────────────────
_gen("Lambda", _TRANSFORM_BUILTINS, [
    "func = lambda {I}: {B}",
    "transform = lambda {I}: {B}\n{R} = transform({D}[0])",
    "{R} = list(map(lambda {I}: {B}, {D}))",
])

# ── RETURN ────────────────────────────────────────────────────────────────────
_gen("Return", _BODY_BUILTINS, [
    "def process({D}):\n    return {B}",
    "def compute({D}):\n    {R} = {B}\n    return {R}",
    "def transform({D}):\n    return {B}",
])

# ── ASSERT ────────────────────────────────────────────────────────────────────
_gen("Assert", _COND_BUILTINS, [
    "assert {B}",
    "assert {B}, 'assertion failed'",
    "assert {B}, f'check failed on {{{D}}}'",
])

# ── RAISE ─────────────────────────────────────────────────────────────────────
_gen("Raise", _EXCEPTION_BUILTINS, [
    "if not {D}:\n    raise {B}",
    "raise {B}",
    "def validate({D}):\n    if not {D}:\n        raise {B}",
])

# ── WITH ──────────────────────────────────────────────────────────────────────
_gen("With", _CONTEXT_BUILTINS, [
    "with {B} as f:\n    {R} = f.read()",
    "with {B} as f:\n    {D} = f.readlines()",
    "with {B} as f:\n    for line in f:\n        {R} = line",
])

# ── TRY ───────────────────────────────────────────────────────────────────────
_gen("Try", _BODY_BUILTINS[:20], [   # subset for conciseness
    "try:\n    {R} = {B}\nexcept Exception:\n    pass",
    "try:\n    {R} = {B}\nexcept Exception as e:\n    {R} = None",
    "try:\n    {R} = {B}\nexcept (TypeError, ValueError):\n    {R} = None",
])

# ── FUNCTIONDEF ───────────────────────────────────────────────────────────────
_gen("FunctionDef", _BODY_BUILTINS, [
    "def process({D}):\n    {R} = {B}\n    return {R}",
    "def compute({D}):\n    {R} = {B}\n    return {R}",
    "def run({D}):\n    {R} = {B}\n    return {R}",
])

# ── ASYNCFUNCTIONDEF ──────────────────────────────────────────────────────────
_gen("AsyncFunctionDef", _BODY_BUILTINS[:15], [
    "async def process({D}):\n    {R} = {B}\n    return {R}",
    "async def compute({D}):\n    {R} = {B}\n    return {R}",
])

# ── CLASSDEF ──────────────────────────────────────────────────────────────────
_gen("ClassDef", _BODY_BUILTINS, [
    "class Processor:\n    def run(self):\n        return {B}",
    "class Processor:\n    def compute(self):\n        {R} = {B}\n        return {R}",
    "class Handler:\n    def process(self, {D}):\n        {R} = {B}\n        return {R}",
])

# ── YIELD ─────────────────────────────────────────────────────────────────────
_gen("Yield", _BODY_BUILTINS[:20], [
    "def gen({D}):\n    yield {B}",
    "def generate({D}):\n    {R} = {B}\n    yield {R}",
    "def stream({D}):\n    for {I} in {D}:\n        yield {B}",
])

# ── YIELDFROM ─────────────────────────────────────────────────────────────────
_gen("YieldFrom", _ITER_BUILTINS, [
    "def gen({D}):\n    yield from {B}",
    "def chain({D}):\n    yield from {B}",
])

# ── AUGASSIGN ─────────────────────────────────────────────────────────────────
_gen("AugAssign", _SCALAR_BUILTINS, [
    "count = 0\ncount += {B}",
    "{R} = 0\n{R} += {B}",
    "total = 0\ntotal += {B}",
])

# ── ANNASSIGN ─────────────────────────────────────────────────────────────────
_gen("AnnAssign", _BODY_BUILTINS[:20], [
    "{R}: int = {B}",
    "def process({D}):\n    {R}: int = {B}\n    return {R}",
])

# ── IFEXP (ternary) ───────────────────────────────────────────────────────────
_gen("IfExp", _BODY_BUILTINS[:20], [
    "{R} = {B} if {D} else None",
    "{R} = {B} if {D} else []",
    "{R} = {B} if {D} else 0",
])

# ── BOOLOP ────────────────────────────────────────────────────────────────────
_gen("BoolOp", _COND_BUILTINS, [
    "{R} = {B} and True",
    "{R} = {B} or False",
    "{R} = bool({D}) and {B}",
])

# ── COMPARE ───────────────────────────────────────────────────────────────────
_gen("Compare", _SCALAR_BUILTINS, [
    "{R} = {B} > 0",
    "{R} = {B} == 0",
    "{R} = {B} >= 1",
])

# ── SUBSCRIPT ─────────────────────────────────────────────────────────────────
_gen("Subscript", _ITER_BUILTINS[:8], [
    "{R} = {B}[0]",
    "{R} = {B}[-1]",
    "{R} = {B}[1:]",
])

# ── BREAK ─────────────────────────────────────────────────────────────────────
_gen("Break", _COND_BUILTINS, [
    "for {I} in {D}:\n    if {B}:\n        break",
    "for {I} in {D}:\n    if {B}:\n        break\n    pass",
])

# ── CONTINUE ──────────────────────────────────────────────────────────────────
_gen("Continue", _COND_BUILTINS, [
    "for {I} in {D}:\n    if not {B}:\n        continue\n    pass",
    "for {I} in {D}:\n    if not {B}:\n        continue\n    {R} = {I}",
])

# ── GLOBAL ────────────────────────────────────────────────────────────────────
_gen("Global", _BODY_BUILTINS[:15], [
    "x = None\ndef process({D}):\n    global x\n    x = {B}",
    "counter = 0\ndef update({D}):\n    global counter\n    counter = {B}",
])

# ── DELETE ────────────────────────────────────────────────────────────────────
_gen("Delete", [("len", "len({D})"), ("int", "int({D}[0])")], [
    "{D} = [1, 2, 3, 4, 5]\ndel {D}[{B} - 1]",
    "{D} = [1, 2, 3, 4, 5]\ndel {D}[:{B}]",
])

# ── IMPORT ────────────────────────────────────────────────────────────────────
# (hard to tie to a builtin; use a thin set)
for _b in ["os", "sys", "math", "json", "re"]:
    for D, R, I in _VARS[:3]:
        _register("Import", _b,
                  f"import {_b}\n{R} = {_b}.path.exists({D})" if _b == "os"
                  else f"import {_b}\n{R} = {_b}",
                  note="import as minimal AST context")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-pairings: fill all (AST node, builtin) cells not covered above
# Ensures a balanced 29×55 VP design matrix for variance partitioning.
# Modules (os, sys, math, json, re) are excluded — they require explicit
# imports and only appear naturally with the Import AST node.
# ─────────────────────────────────────────────────────────────────────────────

_REAL_AST_NODES = list(_AST_KEYWORDS.keys())
_MODULE_BUILTINS = {"os", "sys", "math", "json", "re"}

# Build deduplicated cross-builtin list; _BODY_BUILTINS first for best exprs
_seen_cross: set[str] = set()
_CROSS_BUILTINS: list[tuple[str, str]] = []
for _lst in [_BODY_BUILTINS, _ITER_BUILTINS, _COND_BUILTINS,
             _TRANSFORM_BUILTINS, _SCALAR_BUILTINS]:
    for _b, _e in _lst:
        if _b not in _seen_cross and _b not in _MODULE_BUILTINS:
            _CROSS_BUILTINS.append((_b, _e))
            _seen_cross.add(_b)
for _b, _ in _EXCEPTION_BUILTINS:
    if _b not in _seen_cross:
        _CROSS_BUILTINS.append((_b, f"{_b}('error')"))
        _seen_cross.add(_b)
if "open" not in _seen_cross:
    _CROSS_BUILTINS.append(("open", "open({D})"))
    _seen_cross.add("open")

# Snapshot covered explicit pairs before generating cross-pairs
_covered_cross: set[tuple[str, str]] = {
    (s["ast_node"], s["builtin_obj"])
    for s in _stubs if s.get("variant_type") == "explicit"
}

# Generic body templates: {B} embeds any builtin expression safely in the AST
_CROSS_BODY: dict[str, list[str]] = {
    "For":              ["for {I} in {D}:\n    {R} = {B}"],
    "While":            ["{D} = list(range(3))\nwhile {D}:\n    {R} = {B}\n    break"],
    "If":               ["if {D}:\n    {R} = {B}"],
    "ListComp":         ["{R} = [{B} for {I} in {D}]"],
    "DictComp":         ["{R} = {{k: {B} for k, {I} in enumerate({D})}}"],
    "SetComp":          ["{R} = {{str({B}) for {I} in {D}}}"],
    "GeneratorExp":     ["{R} = list({B} for {I} in {D})"],
    "Lambda":           ["func = lambda {I}: {B}\n{R} = func({D}[0])"],
    "Return":           ["def process({D}):\n    return {B}"],
    "Assert":           ["assert {B} is not None"],
    "Raise":            ["if not {D}:\n    raise Exception('error')\n{R} = {B}"],
    "With":             ["with open({D}) as f:\n    {R} = {B}"],
    "Try":              ["try:\n    {R} = {B}\nexcept Exception:\n    pass"],
    "FunctionDef":      ["def process({D}):\n    {R} = {B}\n    return {R}"],
    "AsyncFunctionDef": ["async def process({D}):\n    {R} = {B}\n    return {R}"],
    "ClassDef":         ["class Processor:\n    def run(self, {D}):\n        return {B}"],
    "Yield":            ["def gen({D}):\n    yield {B}"],
    "YieldFrom":        ["def gen({D}):\n    yield from [{B}]"],
    "AugAssign":        ["total = 0\ntotal += 1\n{R} = {B}"],
    "AnnAssign":        ["def process({D}):\n    {R}: int = 0\n    {R} = {B}\n    return {R}"],
    "IfExp":            ["{R} = {B} if {D} else None"],
    "BoolOp":           ["{R} = bool({B}) and True"],
    "Compare":          ["{R} = ({B}) is not None"],
    "Subscript":        ["{R} = [{B}][0]"],
    "Break":            ["for {I} in {D}:\n    {R} = {B}\n    break"],
    "Continue":         ["for {I} in {D}:\n    {R} = {B}\n    continue"],
    "Global":           ["_g = None\ndef process({D}):\n    global _g\n    _g = {B}"],
    "Delete":           ["{D} = [1, 2, 3]\ndel {D}[0]\n{R} = {B}"],
    "Import":           None,  # modules only; no cross-pairings needed
}

for _ast_node in _REAL_AST_NODES:
    _tmpls = _CROSS_BODY.get(_ast_node)
    if _tmpls is None:
        continue
    for _b_name, _b_expr_tmpl in _CROSS_BUILTINS:
        if (_ast_node, _b_name) in _covered_cross:
            continue
        for _tmpl_idx, _tmpl in enumerate(_tmpls):
            for _vi, (D, R, I) in enumerate(_VARS):
                _B = _b_expr_tmpl.replace("{D}", D).replace("{X}", I)
                _text = _expand(_tmpl, D, R, I, _B)
                if _valid(_text):
                    _register(
                        ast_node=_ast_node,
                        builtin=_b_name,
                        text=_text,
                        variant="explicit",
                        tmpl_id=900 + _tmpl_idx * 100 + _vi,
                        note="cross-pairing for balanced VP design",
                    )


# ─────────────────────────────────────────────────────────────────────────────
# AST baseline stubs
# purpose: isolate the AST-node direction in the residual stream
# design:  same structural template as the main stubs but with NO named
#          Python builtin — use only variable ops or dummy user-defined calls
# category: "{node}____none__"
# ─────────────────────────────────────────────────────────────────────────────

_AST_BASE_TEMPLATES: dict[str, list[str]] = {
    "For": [
        "for {I} in {D}:\n    {R} = {I}",
        "for {I} in {D}:\n    process({I})",
        "for {I} in {D}:\n    pass",
        "{R} = []\nfor {I} in {D}:\n    {R}.append({I})",
    ],
    "While": [
        "{D} = [1, 2, 3]\nwhile {D}:\n    {D} = {D}[1:]",
        "flag = True\nwhile flag:\n    flag = False",
        "{D} = 5\nwhile {D} > 0:\n    {D} -= 1",
    ],
    "If": [
        "if {D}:\n    pass",
        "if {D}:\n    {R} = True\nelse:\n    {R} = False",
        "if not {D}:\n    pass",
    ],
    "ListComp": [
        "{R} = [{I} for {I} in {D}]",
        "{R} = [{I} for {I} in {D} if {I}]",
        "{R} = [{I} * 2 for {I} in {D}]",
    ],
    "DictComp": [
        "{R} = {{k: {I} for k, {I} in enumerate({D})}}",
    ],
    "SetComp": [
        "{R} = {{{I} for {I} in {D}}}",
    ],
    "GeneratorExp": [
        "{R} = list({I} for {I} in {D})",
        "{R} = tuple({I} for {I} in {D})",
    ],
    "Lambda": [
        "func = lambda {I}: {I}",
        "func = lambda {I}: {I} * 2",
        "transform = lambda {I}: {I}\n{R} = transform({D}[0])",
    ],
    "Return": [
        "def process({D}):\n    return {D}",
        "def compute({D}):\n    {R} = {D}\n    return {R}",
        "def identity({D}):\n    return {D}",
    ],
    "Assert": [
        "assert {D}",
        "assert {D}, 'failed'",
        "assert {D} is not None",
    ],
    "Raise": [
        "if not {D}:\n    raise Exception('error')",
        "raise Exception('failed')",
        "def validate({D}):\n    if not {D}:\n        raise Exception()",
    ],
    "With": [
        "with open({D}) as f:\n    {R} = f.read()",
    ],
    "Try": [
        "try:\n    {R} = {D}[0]\nexcept Exception:\n    pass",
        "try:\n    {R} = {D}\nexcept Exception as e:\n    {R} = None",
    ],
    "FunctionDef": [
        "def process({D}):\n    {R} = {D}\n    return {R}",
        "def compute({D}):\n    pass",
        "def run({D}):\n    return {D}",
    ],
    "AsyncFunctionDef": [
        "async def process({D}):\n    {R} = {D}\n    return {R}",
        "async def run({D}):\n    pass",
    ],
    "ClassDef": [
        "class Processor:\n    def run(self):\n        pass",
        "class Handler:\n    def process(self, {D}):\n        return {D}",
    ],
    "Yield": [
        "def gen({D}):\n    yield {D}",
        "def generate({D}):\n    for {I} in {D}:\n        yield {I}",
    ],
    "YieldFrom": [
        "def gen({D}):\n    yield from {D}",
    ],
    "AugAssign": [
        "count = 0\ncount += 1",
        "{R} = 0\n{R} += 1",
        "total = 0\ntotal += {D}[0]",
    ],
    "AnnAssign": [
        "{R}: int = 0",
        "def process({D}):\n    {R}: int = {D}[0]\n    return {R}",
    ],
    "IfExp": [
        "{R} = {D} if {D} else None",
        "{R} = {D}[0] if {D} else 0",
    ],
    "BoolOp": [
        "{R} = {D} and True",
        "{R} = {D} or False",
    ],
    "Compare": [
        "{R} = {D}[0] > 0",
        "{R} = {D}[0] == 0",
    ],
    "Subscript": [
        "{R} = {D}[0]",
        "{R} = {D}[1:]",
    ],
    "Break": [
        "for {I} in {D}:\n    break",
        "for {I} in {D}:\n    if {I}:\n        break",
    ],
    "Continue": [
        "for {I} in {D}:\n    if not {I}:\n        continue\n    pass",
    ],
    "Global": [
        "x = None\ndef process({D}):\n    global x\n    x = {D}",
    ],
    "Delete": [
        "{D} = [1, 2, 3]\ndel {D}[0]",
    ],
    "Import": [
        "import os",
        "import sys",
    ],
}

for _ast_node, _templates in _AST_BASE_TEMPLATES.items():
    for _tmpl_idx, _tmpl in enumerate(_templates):
        for _vi, (D, R, I) in enumerate(_VARS):
            _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
            if _valid(_text):
                _register(
                    ast_node=_ast_node,
                    builtin="__none__",
                    text=_text,
                    variant="baseline_ast",
                    tmpl_id=_tmpl_idx * 10 + _vi,
                    note=f"AST-only baseline — no builtin token",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Builtin baseline stubs
# purpose: isolate the builtin direction in the residual stream
# design:  builtin called in a neutral Assign context (Assign + Call only)
#          no other interesting AST nodes — the simplest possible wrapper
# category: "__assign__{builtin}"
# ─────────────────────────────────────────────────────────────────────────────

# All unique builtins in the main stubs
_ALL_BUILTINS: list[tuple[str, str]] = list({
    (b, e)
    for lst in [_ITER_BUILTINS, _COND_BUILTINS, _TRANSFORM_BUILTINS,
                _SCALAR_BUILTINS, _BODY_BUILTINS, _EXCEPTION_BUILTINS,
                _CONTEXT_BUILTINS]
    for b, e in lst
})
# deduplicate by builtin name (keep first expression form)
_seen_b: set[str] = set()
_UNIQUE_BUILTINS: list[tuple[str, str]] = []
for _b, _e in _ALL_BUILTINS:
    if _b not in _seen_b:
        _UNIQUE_BUILTINS.append((_b, _e))
        _seen_b.add(_b)

# Neutral assignment templates for baselines
_BASE_ASSIGN_TMPLS = [
    "{R} = {B}",
    "x = {B}",
    "out = {B}",
    "{R} = {B}\nprint({R})",
    "{R} = {B}\nassert {R} is not None",
]

for _b_name, _b_expr_tmpl in _UNIQUE_BUILTINS:
    # exceptions can't be used in assignment context → skip
    if any(_b_name == exc for exc, _ in _EXCEPTION_BUILTINS):
        # Use raise context as the "simplest" wrapper for exceptions
        for _vi, (D, R, I) in enumerate(_VARS):
            _B = _b_expr_tmpl.replace("{D}", D).replace("{X}", I)
            _text = f"raise {_B}"
            if _valid(_text):
                _register("__raise__", _b_name, _text,
                          variant="baseline_builtin",
                          note="builtin-only baseline (exception in raise)")
        continue
    if _b_name == "open":
        for _vi, (D, R, I) in enumerate(_VARS[:3]):
            _B = _b_expr_tmpl.replace("{D}", D)
            _text = f"with {_B} as f:\n    {R} = f.read()"
            if _valid(_text):
                _register("__with__", _b_name, _text,
                          variant="baseline_builtin",
                          note="builtin-only baseline (open in with)")
        continue
    for _tmpl_idx, _base_tmpl in enumerate(_BASE_ASSIGN_TMPLS):
        for _vi, (D, R, I) in enumerate(_VARS):
            _B = _b_expr_tmpl.replace("{D}", D).replace("{X}", I)
            _text = _base_tmpl.replace("{R}", R).replace("{B}", _B)
            if _valid(_text):
                _register(
                    ast_node="__assign__",
                    builtin=_b_name,
                    text=_text,
                    variant="baseline_builtin",
                    tmpl_id=_tmpl_idx * 10 + _vi,
                    note="builtin-only baseline — neutral Assign context",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Proxy variants
# ─────────────────────────────────────────────────────────────────────────────

# len  →  .__len__()
for _anode, _tmpl in [
    ("For",          "for {I} in range({D}.__len__()):\n    {R} = {I}"),
    ("If",           "if {D}.__len__():\n    pass"),
    ("Assert",       "assert {D}.__len__() > 0"),
    ("ListComp",     "{R} = [{I}.__len__() for {I} in {D}]"),
    ("Lambda",       "func = lambda {I}: {I}.__len__()"),
    ("Return",       "def process({D}):\n    return {D}.__len__()"),
    ("FunctionDef",  "def process({D}):\n    {R} = {D}.__len__()\n    return {R}"),
    ("Compare",      "{R} = {D}.__len__() > 0"),
    ("AugAssign",    "count = 0\ncount += {D}.__len__()"),
]:
    for D, R, I in _VARS[:3]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "len", _text, variant="proxy", note="len via .__len__()")

# list  →  [*data]
for _anode, _tmpl in [
    ("For",         "for {I} in [*{D}]:\n    {R} = {I}"),
    ("Return",      "def process({D}):\n    return [*{D}]"),
    ("ListComp",    "{R} = [[*{I}] for {I} in {D}]"),
    ("Lambda",      "func = lambda {I}: [*{I}]"),
    ("FunctionDef", "def process({D}):\n    {R} = [*{D}]\n    return {R}"),
    ("YieldFrom",   "def gen({D}):\n    yield from [*{D}]"),
]:
    for D, R, I in _VARS[:3]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "list", _text, variant="proxy", note="list via [*data]")

# str  →  f-string
for _anode, _tmpl in [
    ("ListComp",    '{R} = [f"{{{I}}}" for {I} in {D}]'),
    ("Lambda",      'func = lambda {I}: f"{{{I}}}"'),
    ("Return",      'def process({D}):\n    return f"{{{D}}}"'),
    ("FunctionDef", 'def process({D}):\n    {R} = f"{{{D}}}"\n    return {R}'),
    ("Yield",       'def gen({D}):\n    yield f"{{{D}}}"'),
    ("AnnAssign",   '{R}: str = f"{{{D}[0]}}"'),
]:
    for D, R, I in _VARS[:3]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "str", _text, variant="proxy", note="str via f-string")

# isinstance  →  type() is
for _anode, _tmpl in [
    ("If",          "if type({D}) is list:\n    pass"),
    ("Assert",      "assert type({D}) is list"),
    ("While",       "{D} = [1, 2, 3]\nwhile type({D}) is list and {D}:\n    {D}.pop()"),
    ("Return",      "def process({D}):\n    return type({D}) is list"),
    ("BoolOp",      "{R} = type({D}) is list and True"),
    ("Compare",     "{R} = type({D}) is list"),
]:
    for D, R, I in _VARS[:3]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "isinstance", _text, variant="proxy",
                      note="isinstance via type() is")

# sorted  →  copy + .sort()
for _anode, _tmpl in [
    ("Return",     "def process({D}):\n    {R} = {D}[:]\n    {R}.sort()\n    return {R}"),
    ("FunctionDef","def process({D}):\n    {R} = {D}[:]\n    {R}.sort()\n    return {R}"),
    ("For",        "tmp = {D}[:]\ntmp.sort()\nfor {I} in tmp:\n    {R} = {I}"),
    ("Yield",      "def gen({D}):\n    tmp = {D}[:]\n    tmp.sort()\n    yield from tmp"),
]:
    for D, R, I in _VARS[:3]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "sorted", _text, variant="proxy",
                      note="sorted via copy+.sort()")

# any  →  next() with generator sentinel
for _anode, _tmpl in [
    ("If",     "if next(({I} for {I} in {D} if {I}), None) is not None:\n    pass"),
    ("Assert", "assert next(({I} for {I} in {D} if {I}), None) is not None"),
    ("Return", "def process({D}):\n    return next(({I} for {I} in {D} if {I}), None) is not None"),
    ("BoolOp", "{R} = next(({I} for {I} in {D} if {I}), None) is not None and True"),
]:
    for D, R, I in _VARS[:3]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "any", _text, variant="proxy",
                      note="any via next()+generator")

# range  →  [0..n-1] list comprehension (implied integer sequence)
for _anode, _tmpl in [
    ("For",     "n = 10\nfor {I} in [i for i in range(n)]:\n    {R} = {I}"),
    ("Return",  "def process(n):\n    return [i for i in range(n)]"),
]:
    for D, R, I in _VARS[:2]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "range", _text, variant="proxy",
                      note="range via list comprehension")

# sum  →  manual accumulation
for _anode, _tmpl in [
    ("Return",     "def process({D}):\n    total = 0\n    for {I} in {D}:\n        total += {I}\n    return total"),
    ("FunctionDef","def compute({D}):\n    acc = 0\n    for {I} in {D}:\n        acc += {I}\n    return acc"),
]:
    for D, R, I in _VARS[:2]:
        _text = _tmpl.replace("{D}", D).replace("{R}", R).replace("{I}", I)
        if _valid(_text):
            _register(_anode, "sum", _text, variant="proxy",
                      note="sum via manual accumulation")


# ─────────────────────────────────────────────────────────────────────────────
# Build output
# ─────────────────────────────────────────────────────────────────────────────

def _build() -> list[dict]:
    """
    Finalise all stubs:
      - drop invalid Python
      - assign global prompt_id (sequential, 0-indexed)
      - assign variation_id (per ast_node+builtin_obj group, 0-indexed)
      - strip internal-only fields (_valid, _tmpl_id)
    Returns list of clean record dicts ready for JSONL output.
    """
    invalid = [s for s in _stubs if not s["_valid"]]
    if invalid:
        print(f"WARNING: {len(invalid)} invalid stubs excluded:")
        for s in invalid[:5]:
            print(f"  {s['ast_node']} x {s['builtin_obj']}: {s['prompt_text'][:60]!r}")

    valid = [s for s in _stubs if s["_valid"]]

    # variation_id: 0-indexed counter within each (ast_node, builtin_obj) group
    variation_counter: dict[tuple, int] = defaultdict(int)
    records = []
    for prompt_id, s in enumerate(valid):
        key = (s["ast_node"], s["builtin_obj"])
        variation_id = variation_counter[key]
        variation_counter[key] += 1

        record = {k: v for k, v in s.items() if not k.startswith("_")}
        record["prompt_id"]   = prompt_id
        record["variation_id"]= variation_id
        records.append(record)

    return records


def main() -> None:
    import argparse
    import random

    parser = argparse.ArgumentParser(
        description="Generate contrastive AST x builtin stubs dataset."
    )
    parser.add_argument(
        "--output", "-o",
        default="contrastive_stubs.json",
        help="Output filename (JSONL). Parquet written alongside with same stem. "
             "Relative paths are resolved from the script directory. "
             "(default: contrastive_stubs.json)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of prompts to emit. Prompts are sampled randomly "
             "without replacement after generation. Omit to emit all prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --n sampling (default: 42).",
    )
    args = parser.parse_args()

    records = _build()

    # ── optional down-sampling ─────────────────────────────────────────────────
    if args.n is not None and args.n < len(records):
        random.seed(args.seed)
        records = random.sample(records, args.n)
        # reassign prompt_id to keep it sequential and 0-indexed
        for i, r in enumerate(records):
            r["prompt_id"] = i

    # ── resolve output path ────────────────────────────────────────────────────
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path

    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; pass a different --output to avoid overwriting it.")

    # ── JSONL ──────────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Parquet ────────────────────────────────────────────────────────────────
    pq_path = out_path.with_suffix(".parquet")
    try:
        import pandas as pd
        pd.DataFrame(records).to_parquet(pq_path, index=False)
        print(f"Wrote parquet -> {pq_path}")
    except ImportError:
        print("Parquet skipped (pip install pandas pyarrow)")

    # ── summary stats ─────────────────────────────────────────────────────────
    variant_counts: dict[str, int] = defaultdict(int)
    ast_counts:     dict[str, int] = defaultdict(int)
    builtin_counts: dict[str, int] = defaultdict(int)
    for r in records:
        variant_counts[r["variant_type"]] += 1
        ast_counts[r["ast_node"]] += 1
        builtin_counts[r["builtin_obj"]] += 1

    pairs = {(r["ast_node"], r["builtin_obj"])
             for r in records
             if not r["builtin_obj"].startswith("__")
             and not r["ast_node"].startswith("__")}

    has_pos = sum(1 for r in records if r["ast_keyword_pos"] is not None)

    print(f"\nWrote {len(records)} prompts (JSONL) -> {out_path}")
    print(f"  explicit        : {variant_counts.get('explicit', 0)}")
    print(f"  proxy           : {variant_counts.get('proxy', 0)}")
    print(f"  baseline_ast    : {variant_counts.get('baseline_ast', 0)}")
    print(f"  baseline_builtin: {variant_counts.get('baseline_builtin', 0)}")
    print(f"  AST verified    : {sum(1 for r in records if r['ast_verified'])}/{len(records)}")
    print(f"  Token positions : {has_pos}/{len(records)} "
          f"({'tiktoken available' if _ENC else 'install tiktoken to enable'})")
    print(f"  AST nodes       : {sum(1 for k in ast_counts if not k.startswith('__'))}")
    print(f"  Builtins        : {sum(1 for k in builtin_counts if not k.startswith('__'))}")
    print(f"  (AST, builtin_obj) pairs: {len(pairs)}")
    print()

    print("Prompts per AST node:")
    for node, cnt in sorted(ast_counts.items()):
        print(f"  {node:<22s} {cnt:>5d}")
    print()

    print("Prompts per builtin_obj (top 30):")
    for b, cnt in sorted(builtin_counts.items(), key=lambda x: -x[1])[:30]:
        print(f"  {b:<22s} {cnt:>5d}")

    print()
    print("Contrastive groupings:")
    print("  Fix AST,     vary builtin -> filter by 'ast_group'")
    print("  Fix builtin, vary AST     -> filter by 'builtin_group'")
    print("  Explicit vs proxy         -> filter by 'variant_type'")
    print("  AST baselines             -> variant_type == 'baseline_ast'")
    print("  Builtin baselines         -> variant_type == 'baseline_builtin'")
    print()
    print("Baseline similarity test (pseudocode):")
    print("  act_combined = get_residuals(r['prompt_text'])[-1]   # last layer, last token")
    print("  act_ast_base = mean([get_residuals(s)[-1]")
    print("                       for s in stubs if s['ast_group']==r['ast_group']")
    print("                       and s['variant_type']=='baseline_ast'])")
    print("  act_b_base   = mean([get_residuals(s)[-1]")
    print("                       for s in stubs if s['builtin_group']==r['builtin_group']")
    print("                       and s['variant_type']=='baseline_builtin'])")
    print("  cos_sim(act_combined, act_ast_base + act_b_base)  # ~1 if additive")


if __name__ == "__main__":
    main()
