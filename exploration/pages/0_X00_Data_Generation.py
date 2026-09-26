"""
X00 — How the Dataset Was Built
Sidebar page explaining the design logic behind the contrastive stubs dataset.
"""
import streamlit as st

st.set_page_config(page_title="X00 — Data Generation", layout="wide")

st.title("How the Dataset Was Built")

st.markdown(
    """
    The dataset is the product of five deliberate design choices that
    together let us attribute activation differences to **syntax**,
    **semantics**, or their **interaction**.
    """
)

st.divider()

# ── 1. Factorial Design ─────────────────────────────────────────────────────

st.header("1. Factorial Design")
st.markdown(
    """
    Every AST construct is crossed with every builtin to create the
    `explicit` condition. The full crossing gives **32 AST nodes × 56
    builtins = 1 792 cells**; after filtering for syntactic plausibility
    (e.g. `Raise` cannot wrap every builtin) the dataset retains the
    maximal feasible subset.

    For example, crossing `For` with `sorted` and `If` with `len`:
    """
)
col1, col2 = st.columns(2)
with col1:
    st.caption("`For` × `sorted`")
    st.code("for item in sorted(data):\n    result = item", language="python")
with col2:
    st.caption("`If` × `len`")
    st.code("if len(data):\n    result = True\nelse:\n    result = False", language="python")

# ── 2. Variable-Name Variation ──────────────────────────────────────────────

st.header("2. Variable-Name Variation")
st.markdown(
    """
    Each template is expanded with **6 variable-name sets** to prevent
    the model from latching onto specific variable tokens:

    | Set | Input var | Output var | Iterator var |
    |-----|-----------|------------|--------------|
    | 1   | `data`     | `result`  | `item`       |
    | 2   | `items`    | `output`  | `element`    |
    | 3   | `values`   | `out`     | `x`          |
    | 4   | `elements` | `res`     | `entry`      |
    | 5   | `nums`     | `rv`      | `val`        |
    | 6   | `seq`      | `acc`     | `node`       |

    The same `For` × `sorted` template produces six prompts:
    """
)
col1, col2, col3 = st.columns(3)
with col1:
    st.code("for item in sorted(data):\n    result = item", language="python")
    st.code("for element in sorted(items):\n    output = element", language="python")
with col2:
    st.code("for x in sorted(values):\n    out = x", language="python")
    st.code("for entry in sorted(elements):\n    res = entry", language="python")
with col3:
    st.code("for val in sorted(nums):\n    rv = val", language="python")
    st.code("for node in sorted(seq):\n    acc = node", language="python")
st.markdown(
    """
    Any direction that survives averaging over these six sets cannot be
    explained by variable-name identity alone.
    """
)

# ── 3. Baseline Rationale ──────────────────────────────────────────────────

st.header("3. Baseline Rationale")
st.markdown(
    r"""
    Baselines exist to enable the **additivity test**:

    $$
    \text{cos\_sim}\!\bigl(\,\mathbf{act}(\text{AST}, \text{builtin}),\;
    \mathbf{act}(\text{AST\_baseline}) + \mathbf{act}(\text{builtin\_baseline})\bigr)
    $$

    **High similarity** → the model represents syntax and semantics
    independently (additive / linear superposition).
    **Low similarity** → there is a non-linear interaction circuit
    specific to that (AST, builtin) pair.

    The two baselines isolate each factor:
    """
)
col1, col2 = st.columns(2)
with col1:
    st.caption("**AST baseline** — `For` with no builtin")
    st.code("for item in data:\n    result = item", language="python")
    st.markdown("Pure AST structure; builtin is `__none__`.")
with col2:
    st.caption("**Builtin baseline** — `sorted` with no AST construct")
    st.code("result = sorted(data)", language="python")
    st.markdown("Minimal `Assign` context; no `for`, `if`, etc.")

# ── 4. Proxy Rationale ─────────────────────────────────────────────────────

st.header("4. Proxy Rationale")
st.markdown(
    """
    Proxy variants test **lexical vs. conceptual** representation.
    Instead of calling the builtin token directly, the proxy invokes
    the same semantics indirectly.

    If the model's builtin direction tracks the *token*, proxy
    activations will diverge from `explicit`. If it tracks the
    *concept*, they will align — evidence that the representation is
    genuinely semantic, not surface-level.
    """
)
col1, col2 = st.columns(2)
with col1:
    st.caption("**Explicit** — direct `len()` call")
    st.code("if len(data):\n    pass", language="python")
    st.caption("**Explicit** — direct `sorted()` call")
    st.code("def process(data):\n    result = data[:]\n    return sorted(result)", language="python")
with col2:
    st.caption("**Proxy** — `len` via `.__len__()`")
    st.code("if data.__len__():\n    pass", language="python")
    st.caption("**Proxy** — `sorted` via `copy + .sort()`")
    st.code("def process(data):\n    result = data[:]\n    result.sort()\n    return result", language="python")

# ── 5. AST Verification ────────────────────────────────────────────────────

st.header("5. AST Verification")
st.markdown(
    """
    Every generated prompt is parsed with Python's `ast` module and
    verified to contain the intended AST node. Prompts that fail
    parsing or whose AST tree does not include the target node type
    are discarded before the dataset is finalised.

    For example, a prompt tagged `For` is parsed and its AST is walked
    to confirm a `For` node exists:
    """
)
st.code(
    'import ast\n\n'
    'source = "for item in sorted(data):\\n    result = item"\n'
    'tree = ast.parse(source)\n\n'
    '# Walk the tree and check for the target node\n'
    'has_for = any(isinstance(n, ast.For) for n in ast.walk(tree))\n'
    'assert has_for  # ✓ verified',
    language="python",
)
