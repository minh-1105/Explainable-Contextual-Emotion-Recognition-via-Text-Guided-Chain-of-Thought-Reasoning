# Workflow

Notebooks are the entry points: each one mounts Drive; `02`, `03`, and `04` also clone this repository into `/content/contextual-emotion-reasoning`, install `requirements.txt`, and import shared code from `src/`. Drive holds data, notebooks, and results; `/content` holds the code and the extracted images and is lost when the runtime resets.

Research question: can visual, contextual and semantic-relation evidence improve contextual emotion recognition, and can a VLM verbalize model-derived evidence without inventing unsupported reasons?

## Drive layout

```text
/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition/
├── data/raw/raw/emotic/
│   ├── PAMI/
│   │   ├── Annotations/Annotations/Annotations.mat
│   │   └── pami_emotic.zip        # top folder emotic/, 23,185 files
│   └── CVPR/
│       ├── CVPR17_Annotations.mat
│       └── cvpr_emotic.zip        # top folder cvpr_emotic/, 18,302 files
├── notebooks/
├── outputs/
│   ├── runs/<experiment_name>/<run_id>/
│   ├── resnet_evaluation/<run_id>/
│   ├── multimodal_pipeline/<run_id>/   # 04
│   ├── qwen3_multimodal/<run_id>/      # 04 before the restructure (run H100)
│   └── vlm_pilot/<run_id>/
└── models/<model_name>/<run_id>/
```

Each notebook extracts both zips into `/content/emotic/{PAMI,CVPR}` once per runtime (marker file `.extracted`) and asserts the file counts above. Reading tens of thousands of small images directly from Drive starves the GPU. Training, evaluation, and the pilot read the PAMI annotations and resolve images across both extracted roots.

## Notebooks

| Notebook | Status | What it does |
|---|---|---|
| `01_Inspect.ipynb` | done | Loads PAMI and CVPR annotations, counts images and people per split, checks that the first 100 records of each split resolve to an image, draws every person's bbox on one sample and prints its categories and VAD. |
| `02_Train_Visual_Baseline.ipynb` | done | Trains one visual baseline per run (`person_only`, `context_only`, `person_context`), selects `best.pt` by validation mAP, exports the best model. |
| `03_Evaluate_Visual_Baseline.ipynb` | done | Loads the three exported models, fixes thresholds and constant top-k on validation, then runs test metrics, union/majority sensitivity analysis, image-level paired bootstrap, region ablation, and target swap. |
| `04_VLM_Pilot.ipynb` | restructured, not run yet; run `20261104-071259-h100` done on Modal H100 with the previous layout | One notebook in five parts: Qwen3-VL cues for every PAMI train, val and test person (imported from the H100 run by default), frozen image features (ViT-B/16, CLIP ViT-B/16 or SigLIP) and BERT node features, a variant × seed training matrix that saves per-person dev and eval probabilities, modality-ablation evidence with Qwen3-VL explanations for every eval person, and a single test-set evaluation behind `RUN_TEST` with ResNet-50 comparison, bootstrap and test explanations. Its default configuration reproduces the H100 run. |
| groundedness review, region/relation attribution | planned | Manually review cue and explanation grounding, then add region occlusion, Integrated Gradients and relation ablation. |

Notebook 03 contains the first test-set path. The completed Colab T4 run `20261101-034030` is stored on Drive under `outputs/resnet_evaluation/` and records source commit `0e4b81fc`.

Notebook 03 writes `outputs/resnet_evaluation/<run_id>/`: `config.json`, `environment.txt`, `selected_models.json`, `run_summary.json`, compressed validation/test/ablation/swap predictions, metric JSON/CSV files, and a comparison figure. It accepts finalized `model.pt` exports from either `models/` or the current Drive `checkpoints/` directory; optimizer state is not required for evaluation.

## Visual baselines (`02`)

- ResNet-50 (ImageNet weights) with the final layer removed; `person_context` concatenates a person encoder and a context encoder.
- Head: dropout → 512 → ReLU → dropout → 26 logits. `BCEWithLogitsLoss`, AdamW, cosine schedule, AMP.
- Inputs resized to 224×224 with ImageNet normalization, no augmentation. Labels use the `union` category policy; person crops are padded by 5%.
- Validation mAP is the mean AP over classes present in the split.

Run directory `outputs/runs/<experiment_name>/<run_id>/`:

```text
config.json  environment.txt  run_summary.json
checkpoints/last.pt  checkpoints/best.pt
logs/history.csv
metrics/val_metrics_latest.json
predictions/val_predictions_latest.npz
```

- `last.pt` is written after every epoch (saved to `/content` first, then copied to Drive); `best.pt` is a copy of it when validation mAP improves.
- Resume: set `RESUME_CHECKPOINT` to a `last.pt`. The notebook reuses that run directory and its `config.json`, refuses to resume if `experiment_name`, `backbone`, `image_size`, `category_policy` or `crop_padding` differ, and accepts `RESUME_TOTAL_EPOCHS` to extend the run.
- Export `models/emotic_<experiment_name>_<backbone>/<run_id>/`: `model.pt` (state dict only), `config.json`, `label_map.json`, `metrics.json` (validation metrics of the best epoch), `README.md`. Re-running the export cell overwrites these files.

Run the three configurations one after another, restarting the runtime between them.

## Explainable multimodal prototype (`04`)

For the current Modal account, `cloud/modal_nb04_local.py` runs the local snapshot
on the existing `emotic` volume (see `SETUP.md`). It imports H100 cues but recomputes
features because that volume has no legacy feature caches. It keeps test disabled by default
and compares eval metrics with H100 even without imported features. The snapshot
manifest and transformed input notebook record the runtime configuration; these
executed notebooks are kept separately rather than attached to the Colab source.
Use `--run-id <id> --run-test` to score a finalized run with the existing checkpoints;
the CPU preflight checks the saved checkpoints and ResNet test comparison inputs first.

- Uses all 23,706 PAMI train people for training. All 3,334 PAMI val people are divided by image into dev/eval (`src/datasets/validation_split.py`), so people from the same validation image cannot cross partitions; with seed 42 and fraction 0.5 the split is asserted equal to the H100 `samples.json`. Only section 7 reads test labels, and only when `RUN_TEST = True`, with checkpoints, thresholds and `k` already fixed on dev.
- Section 3: Qwen3-VL 4B receives the full image with the target marked by a red box and extracts visible person cues, context cues, relations and a situation without seeing the emotion label list, for train, val and test (images and boxes only). With `cue_import_from`, cues are copied from the H100 run or an earlier pipeline run after checking model, cue limits and prompt; only missing people are sent to Qwen. Generation benchmarks batch 16/32/48, keeps the fastest with at least 95% schema-valid output, stops if the projected hours exceed `CUE_BUDGET_HOURS`, and closes the JSONL every `BATCHES_PER_SYNC` batches so Drive syncs it.
- Section 4: a frozen image backbone encodes the person crop and the full image; frozen BERT encodes each cue node. Caches are per split. With ViT-B/16 and cues imported unchanged from H100, the H100 caches are imported row by row and the first 32 val people are re-encoded with the new code; the notebook stops if a feature differs by more than 1e-3.
- Section 5: a Graph Transformer and a Multimodal Transformer (`src/models/multimodal_fusion.py`, parameter names equal to the H100 `model.pt`) are trained for every variant × seed (42-44): `full`, `vit_only` (graph token zeroed), `no_person_image`, `no_context_image` (H100 `no_person`, `no_context`), `no_person_cues`, `no_context_cues` (graph nodes of that kind masked), `full_no_affect` (22 affect words deleted from every cue). The checkpoint is the epoch with the best dev mAP; per-label thresholds are selected on dev on the legacy grid (0.05-0.95) and the extended grid (from 1e-8). `reproduction_check` compares the eval rows with the H100 `test_results.json`. On CPU with synthetic data the refactored code gives probabilities and thresholds bit-identical to the previous notebook code.
- Section 6: person, context and graph tokens of one checkpoint (default `full`, seed 42) are zeroed one at a time, and kept one at a time, to build evidence packages. Qwen3-VL verbalizes only this evidence; H100 explanations are reused for people whose evidence package is identical. Explanations are scored for agreement with the largest-drop modality, cited-number accuracy, mentions of unpredicted emotions, and ERASER comprehensiveness and sufficiency against the most frequent, a random and the measured modality. Token-zero ablation is a prototype intervention and does not replace region/relation attribution.
- Section 7: every variant × seed on all 7,280 PAMI test people, constant top-k chosen on dev, majority-label mAP, the notebook 03 ResNet-50 predictions, and per-seed image-level paired bootstrap on mAP and macro-F1 of the primary variant against every other variant, each ResNet and constant top-k; then evidence and explanations for every test person. Once `test/test_results.json` exists, the section only reads it back.
- Resume refuses a changed config (except `git_commit`) or prompt, feature caches refuse a changed `spec.json`, finished variant × seed runs are skipped, and a new commit appends the current environment to `environment.txt`.

Run directory `outputs/multimodal_pipeline/<run_id>/`:

```text
config.json  environment.txt  prompts.json  samples.json
cues/          source_artifacts.json  cues_{train,val,test}.jsonl  batch_benchmark.json  cue_stats.json
features/      <image_backbone>/{spec.json,train.npz,val.npz,test.npz}  bert/{spec.json,cues_<split>.npz,affect_free_<split>.npz}  feature_check.json
train/         checkpoints/<variant>_seed<seed>.pt  predictions/<variant>_seed<seed>.npz  history/<variant>_seed<seed>.json  metrics.json  runs.csv  figures/
explanations/  interventions_eval.npz  evidence_packages_eval.json  explanations_eval.jsonl  explanation_metrics.json (keys eval, test)
test/          predictions_test.npz  test_runs.csv  test_results.json  interventions_test.npz  evidence_packages_test.json  explanations_test.jsonl
```

`train/predictions/*.npz` hold dev and eval sample ids and probabilities and `thresholds_legacy`, `thresholds_extended`; checkpoints also store the config, variant, input dims and best dev mAP. Notebook 05 still reads the H100 run and has to be pointed at `train/predictions/`.

### Run `20261104-071259-h100` (previous layout)

Run `20261104-071259-h100` (Drive `outputs/qwen3_multimodal/20261104-071259-h100/`) continues minhtb's run `20261104-071259`: 2,275 train cues came from unbatched generation on a T4, the other 24,765 from batch 48 on Colab A100 and Modal H100. The H100 part took about 1.9 hours. Section 13, mean over seeds 42-44, mAP eval / test: full 40.5 / 28.5, ViT-only 37.5 / 27.0, no-context 40.1 / 29.8, no-person 39.8 / 30.4, affect-free cues 39.7 / 30.0; constant top-26 23.8 / 17.0; best ResNet-50 (context only) 27.35 on test. Affect-free minus ViT-only macro-F1 on seed 42: +2.8 (95% CI 0.7 to 5.2) on eval and +2.7 (1.9 to 3.4) on test; full minus ViT-only on test -0.3 (-1.1 to 0.5). On eval, 1,667 of 1,691 explanations are usable; 99.99% of cited numbers are in the evidence, but the VLM names the person in 1,660 of them and agrees with the largest-drop modality in 38.3% against 40.3% for always naming the most common one; ERASER comprehensiveness / sufficiency 0.072 / 0.123 for the stated modality, 0.078 / 0.120 for the most common one. On test, 7,193 of 7,280 explanations are usable; 99.97% of cited numbers are in the evidence, the VLM names the person in 7,158, and the person is also the most common largest-drop modality, so agreement (38.8%) and ERASER (0.077 / 0.125) equal those of always naming the person. BERT and ViT feature caches stay on the Modal volume. The papermill run `modal_executions/20261107-133419` reproduced the earlier numbers exactly and drew the figures; the executed notebook of that layout is `outputs/modal_executions/20261107-133419/04_VLM_Pilot.ipynb` on the Modal volume `ecer-nb04` (in git: commit `e304c0ca`).

Its run directory `outputs/qwen3_multimodal/<run_id>/`: `config.json`, `samples.json`, `prompts.json`, `environment.txt`, `batch_benchmark.json`, `qwen_cues.jsonl`, `bert_graph_features.npz`, `vit_features.npz`, `model.pt`, `metrics.json`, `predictions_eval.npz`, `comparison.json`, `evidence_packages.json`, `explanation_prompt_used.txt`, `explanations.jsonl`, `explanation_metrics.json` (keys `eval` and `test`), `qwen_cues_test.jsonl`, `test_features.npz`, `affect_free_features.npz`, `test_results.json`, `predictions_test.npz`, `evidence_packages_test.json`, `explanations_test.jsonl`, and `figures/` (training curves, variant comparison, per-label AP). On Modal, the executed notebook and a timestamped `run.log` go to `outputs/modal_executions/<UTC start>/`. `metrics.json` holds the per-epoch train loss, dev mAP and seconds of the main model; `comparison.json` holds the same history for every variant run.

## Experiment matrix

| Experiment | Person | Context | VLM | Structured reasoning |
|---|:-:|:-:|:-:|:-:|
| A — Person only | ✓ |  |  |  |
| B — Context only |  | ✓ |  |  |
| C — Person + Context | ✓ | ✓ |  |  |
| D — Qwen3 cue graph | ✓ | ✓ | ✓ | ✓ |
| E — ViT + BERT + Graph/Multimodal Transformer | ✓ | ✓ | ✓ | ✓ |
| F — Evidence-grounded explanation | ✓ | ✓ | ✓ | ✓ |
