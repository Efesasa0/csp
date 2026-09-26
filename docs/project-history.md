# Project history

The final study came out of several changes of direction over about three months (January to April 2026). They are recorded here because the dead ends shaped the method.

## 1. Neural masquerading (January to February)

The original proposal asked whether an LLM can produce harmful output while keeping its internal state looking "safe" to linear probes. The planned method was to measure representational drift between honest and deceptive activations with orthogonal Procrustes alignment, on Llama 3 and Qwen, using the ETHICS and jailbreak datasets. Early code downloaded ETHICS and extracted layer activations.

Why we moved on: the scope needed large models and a fine-tuning step the course compute could not support.

## 2. Circuit sparsity and CSP-Atlas (February to March)

We switched to OpenAI's weight-sparse code model [`openai/circuit-sparsity`](https://huggingface.co/openai/circuit-sparsity). It is small, and its circuits are sparse by construction. The model is weak as a general coder but reliable on syntax. So the question became how syntax-level behaviour is implemented and organised, rather than trying to make the model behave a certain way.

Piotr Wilam built [CSP-Atlas](https://github.com/piotrwilam/CSP-Atlas). It generates prompts over the (AST node x builtin) grid, filters them by model loss, extracts activations, and builds "universal circuits" by binarising and intersecting activation masks.

## 3. Contrastive stubs and variance partitioning (March)

The intersection masks were hard to test statistically. We rebuilt the dataset as minimal contrastive stubs with AST-only, builtin-only and proxy baselines, and replaced mask intersection with variance partitioning. That split each unit's R² into AST, builtin, shared and unexplained parts. Probing, RSA, additivity and causal-ablation experiments from this phase are in [`exploration/`](../exploration/).

## 4. Delimiter control and SAEs (April)

Some AST groupings turned out to follow the prompt's closing token. `ListComp` and `Subscript` both end in `]`, and `DictComp` and `SetComp` both end in `}`. The constructor-stubs dataset rewrites these so every prompt ends in `)`, and the final VP results use it ([`variance_partitioning/`](../variance_partitioning/)). Sparse autoencoders were trained as an independent second lens ([`sparse_autoencoders/`](../sparse_autoencoders/)).
