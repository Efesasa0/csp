# Logs

- 2026-09-26 - Curated public repo from the COMP0087 group repo: VP pipeline, SAE pipeline, exploration experiments and dashboard.
- 2026-09-26 - SAE pipeline paths now resolve relative to `sparse_autoencoders/` instead of the old git root.
- 2026-09-26 - Removed hardcoded personal paths and run logs; added `prepare_data.py` for the SAE parquet input.
- 2026-09-26 - Added MIT license with all five authors as joint copyright holders.
- 2026-09-26 - Added the report PDF under docs/ and restructured the README (idea, conclusion, layout, quickstart, citation).
- 2026-09-26 - Swapped in the camera-ready report with author names.
- 2026-09-26 - Added SETUP.md with install and full-run instructions.
- 2026-09-26 - Fixed crash in 02p_group_selectivity.py: subplot grid was 4 columns but indexed as one row, failing for k > 4.
- 2026-09-26 - check_neurons.py takes --in_dir instead of hardcoding data/.
- 2026-09-27 - VP pipeline now also runs 02h_polysemanticity.py (report Fig. 2, Assert neuron).
- 2026-09-27 - Removed a hardcoded Windows path in 02q_sae_vp_bridge.py; it now reads SAE outputs from sparse_autoencoders/data/.
- 2026-09-27 - SAE scripts resolve all data paths through core.data_path() instead of cwd-relative "data/..." strings.
- 2026-09-27 - Added a status note to the README: research code, provided as-is.
- 2026-09-27 - Status note: the older contrastive dataset is included (study_v2 metadata_old.parquet); ignored .venv/.
