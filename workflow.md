# Workflow

Notebooks are the entry points: each one mounts Drive, clones this repository into `/content/contextual-emotion-reasoning`, installs `requirements.txt`, and imports shared code from `src/`. Drive holds data, notebooks, and results; `/content` holds the code and the extracted images and is lost when the runtime resets.

Research question: does structured text-guided reasoning improve contextual emotion recognition and the quality of its explanations?

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
│   └── vlm_pilot/<run_id>/
└── models/<model_name>/<run_id>/
```

Each notebook extracts both zips into `/content/emotic/{PAMI,CVPR}` once per runtime (marker file `.extracted`) and asserts the file counts above. Reading tens of thousands of small images directly from Drive starves the GPU. Training and the pilot read the PAMI annotations and resolve images across both extracted roots.

## Notebooks

| Notebook | Status | What it does |
|---|---|---|
| `01_Inspect.ipynb` | done | Loads PAMI and CVPR annotations, counts images and people per split, checks that the first 100 records of each split resolve to an image, draws every person's bbox on one sample and prints its categories and VAD. |
| `02_Train_Visual_Baseline.ipynb` | done | Trains one visual baseline per run (`person_only`, `context_only`, `person_context`), selects `best.pt` by validation mAP, exports the best model. |
| `03_Evaluate_Visual_Baseline.ipynb` | done | Loads the three exported models, fixes thresholds and constant top-k on validation, then runs test metrics, union/majority sensitivity analysis, image-level paired bootstrap, region ablation, and target swap. |
| `04_VLM_Pilot.ipynb` | done | Pilot of D (direct) vs E (structured) prompting on 300 PAMI val people with `Qwen/Qwen2.5-VL-3B-Instruct`. |
| full D / E runs, result analysis, LoRA | planned | LoRA only after D and E zero-shot results are stable. |

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

## VLM pilot (`04`)

- The target person is marked with a red box on the full image; both prompts use the same model, image, label list, token limit (384), and greedy decoding.
- D asks for `{"emotions": [...]}`. E asks for one object with keys `person_cues`, `context_cues`, `situation`, `emotions`, in that order.
- Each answer is appended to `predictions_<mode>.jsonl` as soon as it is generated; re-running with the same `RUN_ID` skips finished samples and refuses a changed config or prompt.
- Metrics: unparsable answers count as empty predictions; macro F1 over labels present in the sample; `schema_rate` checks form only; 95% CI of the E − D macro-F1 difference from a paired bootstrap that resamples images (2,000 draws).

Run directory `outputs/vlm_pilot/<run_id>/`: `config.json`, `samples.json`, `prompts.json`, `environment.txt`, `predictions_direct.jsonl`, `predictions_structured.jsonl`, `metrics.json`, `examples.json`.

## Experiment matrix

| Experiment | Person | Context | VLM | Structured reasoning |
|---|:-:|:-:|:-:|:-:|
| A — Person only | ✓ |  |  |  |
| B — Context only |  | ✓ |  |  |
| C — Person + Context | ✓ | ✓ |  |  |
| D — Direct VLM | ✓ | ✓ | ✓ |  |
| E — Structured VLM | ✓ | ✓ | ✓ | ✓ |
