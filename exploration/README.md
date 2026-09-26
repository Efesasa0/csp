## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd csp-atlas/exploration

# 2. Install dependencies
pip install -r requirements.cuda121-py310.txt

# 3. Download all artifacts from Hugging Face
bash scripts/fetch_artifacts.sh

# 4. Launch the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501` with sidebar pages for data generation,
model details, extraction pipeline, and analysis results.

## OpenAI circuit_sparsity Architecture
It's a weight-sparse GPT-2-style decoder-only transformer with two non-standard components:

```
Token Embedding (wte)
       +
Position Embedding (wpe)  ← concatenated inside each block, NOT added to residual
       ↓
  [Transformer Block] × N
  ┌─────────────────────────────┐
  │  RMSNorm                    │  ← replaces LayerNorm
  │  Multi-Head Attention       │
  │    └─ AbsTopK activation    │  ← sparsifies attention (non-standard)
  │  MLP                        │
  │    └─ AbsTopK activation    │  ← sparsifies MLP neurons
  └─────────────────────────────┘
       ↓
  RMSNorm (ln_f)
       ↓
  LM Head (W_U)
       +
  Bigram Table                  ← learned per-token logit bias, bypasses transformer
       ↓
  Logits
  ```

## What to look at for interpretability — ranked
### Residual Stream — highest priority
This is the backbone. Every component (attention heads, MLPs) reads from and writes to it additively. This is where your additivity test lives. The get_residuals() pattern in your notebook captures it at every layer for the last token.

- Why it matters for your work: your (AST, builtin) factorisation hypothesis is a claim about the residual stream — that AST and builtin directions are approximately orthogonal subspaces within it.

### Attention Head Activations (logit attribution) — high priority
Your notebooks already compute signed per-head logit contributions:
```
contribution ≈ W_U[target] · W_O[head] · attn_output[head]
```
This tells you which heads route information for a given (AST, builtin) combination. Because the model is weight-sparse, most heads will have near-zero contribution — the active ones stand out clearly.

- For your work: do the same heads activate for For + range as for While + range? If so, that's a builtin-specific head. If the head pattern changes with the AST node, it's an interaction head.

### MLP Neuron Activations — high priority, especially with AbsTopK
AbsTopK is the key architectural quirk. Instead of ReLU/GELU, it keeps only the top-K neurons by absolute value and zeros the rest. This means:

- Very few neurons fire per forward pass
- The neurons that do fire are strongly committed (positive OR negative)
- MLP layers here behave more like sparse feature detectors than dense transformations

- For your work: a neuron that fires for len(data) across multiple AST contexts is a builtin-specific feature. One that fires only for for x in len(data) is an interaction feature.

4. Bigram Table — medium priority, important to control for
This is a `(vocab_size, vocab_size)` learned bias added to every logit prediction. It encodes "token X was just seen → bias logits toward token Y". It completely bypasses the transformer.

- For your work: this is a confounder. If sorted( always predicts ) via the bigram table, your residual stream attribution is measuring something the transformer didn't actually need to learn. Always subtract it from logit lens analysis:


logit_without_bigram = lm_head_output[0, -1, :] - bigram_table[last_token_id]

### RMSNorm — low priority for interpretability directly
RMSNorm rescales the residual stream before each attention/MLP block. It doesn't add information — it just normalises magnitude. You mostly need to account for it when computing exact attribution scores (the norm gate changes vector magnitudes), but you don't need to interpret it.

- Practical note: because of RMSNorm, cosine similarity is more meaningful than Euclidean distance for comparing residual vectors.

### cat_pos_emb (concatenated positional embeddings) — medium priority, subtle
Position embeddings are concatenated inside each block rather than added to the residual stream. This means:

- The residual stream dimension is hidden_dim (clean, no positional contamination)
- But each block internally sees [hidden; pos_emb] — position modulates attention and MLP
- Your get_residuals() hooks correctly capture the clean residual (before concatenation)
- For your work: token-position effects are partially factored out of the residual stream. This is actually beneficial — your AST/builtin directions in residual space are less polluted by positional information than in standard GPT-2.

### Edge Connectivity (weight sparsity graph) — useful for circuit-finding, not first priority
Because weights are explicitly sparse, you can draw a literal graph of which neurons connect to which. This is closer to the Circuits paper approach. High-impact for identifying minimal circuits, but requires more setup.

### Priority ordering for your specific experiment

1. Residual stream                ← your additivity test lives here
2. Attention head attribution     ← which heads are builtin-specific vs AST-specific
3. MLP AbsTopK neurons            ← sparse feature detectors; easy to isolate
4. Bigram table (as control)      ← subtract before logit attribution
5. cat_pos_emb (awareness only)   ← know it's there, your hooks handle it correctly
6. RMSNorm (awareness only)       ← use cosine similarity, not L2
7. Edge connectivity              ← for follow-up circuit-finding work
The weight sparsity is your structural advantage over standard GPT-2 interpretability — most weights are zero, so the circuits that do exist are fewer and cleaner to isolate.


## an you have a truly "pure" prompt?
No. It's structurally impossible in Python.
 Even the most minimal valid statements contain multiple AST nodes:

```
len(x)
# AST nodes: Module, Expr, Call, Name(id='len'), Name(id='x'), Load, Load
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unavoidable overhead
```
```
for i in range(10):
    pass
# AST nodes: Module, For, Name, Store, Call, Name(id='range'), Constant, Pass
#            ^^^^^^^^^^^^^^^^^^^^^ the For you want + 7 others
```
Every prompt has an unavoidable syntactic overhead layer:

| Layer	| Nodes | Always present? |
|--|--|--|
| Root glue | `Module, Expr, Load, Store` | Always |
| Call glue	| `Call, Name` | Whenever a builtin is called |
| Primary AST | `For, While, If, Lambda...` | By design |
| Primary builtin | `len, sorted, range...` | By design |

The key insight is that the overhead nodes appear in every prompt — they contribute a constant offset to all activations, not a confounding signal. They affect the intercept, not the contrast.


## Is the current dataset suitable?
The small_40x50x50 dataset has problems for this purpose:

Problem	Detail
Structural wrappers vary	Same (AST, builtin) pair appears bare, in a function, in a class — this adds FunctionDef/ClassDef variance orthogonal to your labels
No baselines	No AST-only or builtin-only control condition
Variable token length	Confounds positional encoding effects
Thematic vocabulary varies	ledger_entries, dna_samples, shopping_cart — domain-specific names may activate distinct circuits
contrastive_stubs_v2.json is better because templates are minimal and baselines exist, but it still has the fundamental overhead problem above. The overhead isn't fatal — it just needs to be handled statistically.


## How to avoid / deconfound contamination
Design-level controls (do before running inference)
1. Fix the structural wrapper. Use only one template form — e.g. always bare statements, never inside a class. This removes ClassDef/FunctionDef from the variance entirely.

2. Fix variable names. Use data, result, item consistently (your v2 generator already does this via _VARS, but pick one and stick to it for the baseline comparison).

3. Equate token length. Pad or truncate to the same length, or stratify your analysis by n_tokens. The model's positional encoding means a 30-token prompt and a 60-token prompt are not directly comparable at the last-token position.

4. The overhead nodes are a constant — use them as your intercept. Run the most minimal possible "null" prompt (e.g. x = data) that has only overhead nodes and no primary AST or builtin. Its activation is your empirical zero point.

### Statistical deconfounding (do after running inference)
#### Method 1: Variance Partitioning (most direct)
This is the Venn-diagram approach from ecology/neuroscience. Fit three linear models predicting activation at each hidden dimension:

```
Model A:  activation ~ AST_label               → R²_A
Model B:  activation ~ builtin_label            → R²_B
Model AB: activation ~ AST_label + builtin_label → R²_AB
```

Then decompose:
```
Unique to AST     = R²_AB - R²_B      ← variance only AST explains
Unique to builtin = R²_AB - R²_A      ← variance only builtin explains
Shared            = R²_A + R²_B - R²_AB  ← crosstalk / collinear variance
Unexplained       = 1 - R²_AB
```

If Shared is large, AST and builtin representations are entangled in activation space. If small, they're orthogonal. This directly answers your question.

#### Method 2: Representational Similarity Analysis (RSA)
Build two label-similarity matrices and one activation-similarity matrix:

```
# AST label matrix: M_ast[i,j] = 1 if stub i and j have same AST node
# Builtin label matrix: M_b[i,j] = 1 if stub i and j have same builtin
# Activation matrix: M_act[i,j] = cos_sim(act_i, act_j)

# Then:
r_ast    = spearman(upper_tri(M_act), upper_tri(M_ast))
r_b      = spearman(upper_tri(M_act), upper_tri(M_b))
r_both   = partial_spearman(M_act, M_ast, controlling_for=M_b)
```

r_both is the AST effect after partialling out the builtin effect — this is your crosstalk-free signal.

#### Method 3: Linear Probing with Ablation
```
# Train probe: activation → AST label
acc_ast_full     = probe(act, label="ast_node")

# Now residualise activations: remove the builtin direction first
act_residualised = act - project_onto(act, builtin_subspace)
acc_ast_residual = probe(act_residualised, label="ast_node")

# Drop in accuracy = how much AST information was carried in the builtin subspace
crosstalk = acc_ast_full - acc_ast_residual
```

### # Method 4: Factorial ANOVA on probe accuracy
Since your dataset is factorial (AST × builtin), you can run a 2-way ANOVA:

```
probe_accuracy ~ AST_node + builtin + AST_node:builtin + error
The interaction term AST_node:builtin directly measures whether knowing both factors together predicts accuracy better than knowing each separately — i.e., whether the model has learned joint circuits for specific (AST, builtin) pairs.
```

The interaction term AST_node:builtin directly measures whether knowing both factors together predicts accuracy better than knowing each separately — i.e., whether the model has learned joint circuits for specific (AST, builtin) pairs.


## Recommended approach for your specific experiment
Given you have the residual stream and the contrastive_stubs_v2.json dataset:

```
Step 1 — Extract residuals at the final layer for all stubs

Step 2 — Variance partitioning per hidden dimension
          → shows which dimensions encode AST, which encode builtin,
            which encode both (the crosstalk dimensions)

Step 3 — RSA with partial correlation
          → gives a single scalar: "how much does AST structure
            predict activation similarity, controlling for builtin?"

Step 4 — Explicit vs proxy comparison (already in contrastive_analysis.py)
          → separates lexical from semantic crosstalk
            (is the builtin direction driven by the token 'len'
             or by the concept of "counting"?)

Step 5 — Factorial ANOVA on linear probe accuracy
          → tests for interaction circuits
```
The crosstalk question splits into two distinct phenomena worth distinguishing:

| Type | Meaning | Detected by |
|--|--|--|
| Structural crosstalk | for loops always call something → For and Call are correlated | Variance partitioning |
| Semantic entanglement | The model represents For+range as a single concept, not For + range separately |Factorial ANOVA interaction term |
| Lexical bleed | len activates len-specific features even when the concept is conveyed without the token | Explicit vs proxy comparison |

The structural crosstalk is a design problem (minimise with template control). The semantic entanglement and lexical bleed are findings, not problems — they're exactly what you want to measure.



The information flow argument
In a transformer, the residual stream is the sum of all component contributions up to that point:


residual[L] = embedding
            + Σ attn_output[0..L]
            + Σ mlp_output[0..L]
So the final-layer residual stream is a compressed aggregate of everything every head and every MLP has written. When you measure it, you're measuring the net result of all circuits combined.


Layer 0:  [embed] ──→ attn₀ ──→ mlp₀ ──→ residual₀
Layer 1:  residual₀ ──→ attn₁ ──→ mlp₁ ──→ residual₁
Layer 2:  residual₁ ──→ attn₂ ──→ mlp₂ ──→ residual₂  ← final layer
                                                          ↑
                                            what you read off here
                                            already contains everything
This is why it's Step 1 — it tells you whether AST and builtin information is separable at all, before you go looking at which specific mechanism does the separating.

The three things you're trading off
Signal	Granularity	Noise	What it answers
Residual stream (final)	Low — aggregate	Low — averaged out	Does the model separate AST from builtin?
Residual stream (per layer)	Medium	Medium	When does the separation happen?
Attention head outputs	High — per head	High	Which heads implement which separation?
MLP neuron activations	Highest — per neuron	Highest	Which features encode AST vs builtin?
Going straight to attention heads or MLP neurons is like trying to identify which specific neuron in a brain encodes "cat" before you've confirmed the brain responds differently to cats at all. You'd be doing expensive, high-variance analysis on a question you haven't confirmed has a signal.

Why attention heads specifically are harder to start with
Each attention head produces a (seq_len, head_dim) output — for your purpose you'd need to decide:

Which token position to read? (last token? the builtin token? the AST keyword token?)
Which head? There are n_layers × n_heads of them — in a typical circuit_sparsity model that's ~100+
How to aggregate across heads at the same layer?
The residual stream sidesteps all of these choices — at the last token position, it's a single (hidden_dim,) vector that represents the model's entire state going into the prediction.

Why MLP neurons are harder to start with
AbsTopK means most neurons are zero. The neurons that fire vary per prompt, so the active set itself is a sparse, high-dimensional binary vector. For your variance partitioning or RSA analysis, you'd need to either:

Work with sparse vectors (computationally tricky)
Aggregate to a dense representation (losing the sparsity advantage)
Pre-select neurons — but which ones, and based on what?
Again, the residual stream first tells you where to look in the MLP.

The right order

Residual stream (final layer)
    → confirms signal exists, gives global picture
    ↓
Residual stream (per layer)
    → tells you which layer the AST/builtin directions crystallise
    ↓
Logit attribution per attention head  ← your notebooks already do this
    → tells you which specific heads route the information
    ↓
MLP AbsTopK neurons at the identified layers
    → tells you which features implement the concepts
    ↓
Edge connectivity (weight sparsity graph)
    → minimal circuit description
Each step uses the previous to constrain the search space. Without Step 1 telling you "the signal is mainly at layer 4 in the residual stream", Step 3 means running attribution on all 100+ heads across all layers blind.

One exception: when to skip straight to heads/neurons
If your research question changes from "are AST and builtin separable?" to "which head implements the For+range circuit?", then yes — you skip the residual stream and go directly to head-level attribution using the logit attribution pattern already in your QQQ notebook. But that's a circuit-finding question, not a representation question. Your current question is the latte


## for a given prompt, how can I analysis whether a builtin concept, say `if`, permeates throughout all the layers of a sparse model?
To analyze whether a concept like an if statement permeates throughout the layers of a sparse model, you should focus on the Residual Stream as your primary diagnostic tool, supplemented by Logit Lens and Feature Probing.

In a Transformer, the residual stream acts as the model's "working memory." If a concept is "permeating," it means the vector representation of the tokens related to if maintains a specific direction in that high-dimensional space across successive layers.

### The Residual Stream Analysis (The "Main Bus")
The residual stream ($x_i$) at layer L is the sum of all previous layer outputs:
$$x_L = x_0 + \sum_{i=1}^{L} \text{Sublayer}_i(x_{i-1})$$
To see if if is present:

Directional Persistence: Perform a Principal Component Analysis (PCA) or use a Sparse Autoencoder (SAE) on the residual stream at every layer.

What to look for: Does a specific "feature" (a direction in the vector space) that represents "conditional logic" appear in Layer 2 and remain active until Layer 32?

### Logit Lens (Direct Decoding)
The Logit Lens technique allows you to "peek" at what the model is thinking mid-stream.

The Method: Take the hidden state h 
L
​
  from an intermediate layer and pass it directly through the final Unembedding Layer (the layer that turns vectors back into words).

The Test: If you pass the token if through the model, and at every layer from 5 to 30 the Logit Lens shows the model is predicting tokens like else, :, or True, then the "concept" of the conditional is actively permeating those layers.

### Probing for "Circuit Edges"
In OpenAI's sparse architecture, you can move beyond simple activations to Edge Attribution.

Step 1: Identify the MLP neurons in Layer 5 that fire for if.

Step 2: Check the Edge Connectivity. Are there non-zero weights connecting those Layer 5 neurons to neurons in Layer 6, 7, and 8 that also handle logical flow?

Significance: If the edges are "dense" (many connections) between layers for this specific concept, it is a Circuit. If the signal dies out or the edges disappear, the model has "compressed" that information and moved on to a different abstraction.

### Summary of Diagnostics


|Technique | What it tells you|
|--|--|
|SAE Probing | If the feature of "if-ness" is active in the residual stream.|
|Logit Lens | If the model's prediction is still influenced by the if.|
|Ablation |If you zero out the if neurons in Layer 2, does the logic fail in Layer 20? (Proof of permeation).|

Comparison Table: Concept Depth
|Concept Type | Permeation Profile|
|--|--|
|Bigram (e.g., "New" -> "York") |Shallow. Disappears after 1–2 layers into the Bigram Table.
|Syntactic (e.g., if statement)| Deep. Persists through the middle layers to maintain "state."|
|Semantic (e.g., Sentiment)| Late-stage. Often forms in the final 25% of the model.|

```python
import torch
from transformer_lens import HookedTransformer

# 1. Load a model (using a standard GPT-2 as a proxy for sparse logic)
model = HookedTransformer.from_pretrained("gpt2-small")

prompt = "if x > 5:"
tokens = model.to_tokens(prompt)

# 2. Run the model and capture all intermediate activations (cache)
logits, cache = model.run_with_cache(tokens)

# 3. Analyze the Residual Stream ("The Highway")
# We look at the final token position (the colon ':') to see what's predicted next
last_token_index = -1 

print(f"{'Layer':<10} | {'Top Predictions from Residual Stream'}")
print("-" * 50)

for layer in range(model.cfg.n_layers):
    # Get the residual stream accumulated up to this layer
    residual_state = cache["resid_post", layer][0, last_token_index, :]
    
    # Scale and project back to vocabulary (The "Lens" part)
    # This mimics the final layer's 'unembedding' process
    scaled_residual = model.ln_final(residual_state)
    layer_logits = model.unembed(scaled_residual)
    
    # Get top 3 predictions
    top_tokens = torch.topk(layer_logits, k=3).indices
    prediction_strings = [model.to_string(t) for t in top_tokens]
    
    print(f"Layer {layer:<3} | {prediction_strings}")
```


## To Run Stuff on Piotr's Data
```
# ── Dataset 1: contrastive stubs ──────────────────────────────────────────────
python 01_extraction.py --input contrastive_stubs.json
python 02_variance_partition.py --stem contrastive_stubs
python 03_projection.py --stem contrastive_stubs --plot
python 04_additivity.py --stem contrastive_stubs --plot
python 05_probing.py --stem contrastive_stubs --all_concepts --plot
python 06_causal.py --stem contrastive_stubs --plot

# ── Dataset 2: small_40x50x50 (for step 07 extension) ────────────────────────
python 01_extraction.py --input small_40x50x50_validated_prompts.json \
    --stem small_40x50x50_validated_prompts
python 02_variance_partition.py --stem small_40x50x50_validated_prompts
python 07_loss_analysis.py \
    --stem small_40x50x50_validated_prompts \
    --input_json ../../CSP-Atlas-raw-data/small_40x50x50_validated_prompts.json \
    --ref_stem contrastive_stubs \
    --plot
```
Do you want to start running the pipeline now, or is there anything you'd like to tweak in the scripts first? The most common things people adjust before a first run are:

Output directory — currently writes alongside inputs; you may want --out_dir results/
--max_pairs in step 06 — defaults to 500 pairs per factor; reduce if causal analysis is slow
--n_perm in steps 03/04 — defaults to 1000 permutations for Mantel test; reduce to 200 for a quick sanity run
