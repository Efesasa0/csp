# Setup

## Environment

Tested with Python 3.11. Any recent PyTorch works: CUDA on Linux, MPS or CPU
on macOS.

```bash
uv venv -p 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Plain `pip install -r requirements.txt` in a fresh virtualenv works too.
`requirements.txt` installs `circuit_sparsity` from OpenAI's GitHub. The model
weights (`openai/circuit-sparsity`, about 1.6 GB) download from Hugging Face on
first use. Setting `HF_TOKEN` is optional but avoids rate limits.

## Full runs

Variance partitioning. The prompt dataset (`constructor_stubs.jsonl`) ships
with the repo. Extraction over 14,683 prompts takes a while, so use a GPU:

```bash
cd variance_partitioning
python main.py --device cuda
python main.py --skip_extraction   # rerun analysis on existing activations
```

Outputs go to `variance_partitioning/data/`.

Sparse autoencoders:

```bash
cd sparse_autoencoders
python prepare_data.py          # data/constructor_stubs.parquet
python training/retrain.py      # 8 layers x {mlp, resid} SAEs -> data/saes/
python pipeline.py              # feature discovery at layer 7 -> data/reports/
```

See [`sparse_autoencoders/README.md`](sparse_autoencoders/README.md) for the
follow-up analyses and the `study_v2/` factorial study.

## Dashboard

The Streamlit app in `exploration/` browses the earlier contrastive-stubs
experiments. Its artifacts come from the group's Hugging Face dataset
(`AST-Revisited/contrastive-stubs`):

```bash
cd exploration
bash scripts/fetch_artifacts.sh
streamlit run app.py
```
