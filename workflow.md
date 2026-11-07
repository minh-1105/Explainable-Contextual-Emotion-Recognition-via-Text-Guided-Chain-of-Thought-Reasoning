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
| `04_VLM_Pilot.ipynb` | done; run `20261104-071259-h100` on Modal H100 | Generates Qwen3-VL cues for all 23,706 PAMI train and 3,334 PAMI val people and trains on all train people. BERT encodes a semantic graph, ViT encodes person/context, and Graph/Multimodal Transformers predict emotions. Compares full, ViT-only, no-context and no-person variants and scores the explanations. |
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

- Uses all 23,706 PAMI train people for training. All 3,334 PAMI val people are divided by image into dev/eval, so people from the same validation image cannot cross partitions. It never reads the test split.
- Qwen3-VL 4B receives the full image with the target marked by a red box and extracts visible person cues, context cues, relations and a situation without seeing the emotion label list.
- Frozen BERT encodes those semantic nodes; frozen ViT-B/16 encodes the person crop and full image. A trainable Graph Transformer and Multimodal Transformer predict the 26 labels.
- The best checkpoint and per-label thresholds are selected only on the internal dev partition. Reported `eval` metrics refer to a held-out part of PAMI val, not EMOTIC test.
- Person, context and graph tokens are ablated one at a time to create a preliminary evidence package. Qwen3-VL then verbalizes only this model-derived evidence. Token-zero ablation is a prototype intervention and does not replace region/relation attribution.
- Cue generation benchmarks batch sizes 16/32/48 on people that already have cues and keeps the fastest one with at least 95% schema-valid output. It projects the hours left and stops if they exceed `CUE_BUDGET_HOURS`, then generates cues for every missing person, closing the JSONL every `BATCHES_PER_SYNC` batches so Drive syncs it; later cells refuse to run until every person has a cue; BERT and ViT features are cached. Resume refuses a changed config (except `git_commit`) or prompt; a new commit appends the current environment to `environment.txt`.
- Variants `full`, `vit_only` (graph token zeroed), `no_context` and `no_person` are retrained with seeds 42-44 on the same split, compared with constant top-k and an image-level paired bootstrap. Thirty explanations are scored for modality agreement against the largest intervention drop, cited-number accuracy and mentions of unpredicted emotions. Integer sample limits remain available only for smoke tests; the official configuration uses `None` to select every person.

Run `20261104-071259-h100` (Drive `outputs/qwen3_multimodal/20261104-071259-h100/`) continues minhtb's run `20261104-071259`: 2,275 train cues came from unbatched generation on a T4, the other 24,765 from batch 48 on Colab A100 and Modal H100. The H100 part took about 1.9 hours. Eval mean over seeds 42-44: full mAP 40.9, ViT-only 38.1, no-context 39.8, no-person 40.7; full minus ViT-only macro-F1 +3.6 (95% CI 1.9 to 5.8). The 30 explanations cite only numbers present in the evidence, but name the largest-drop modality in 30% of cases against 53% for always naming the most common one. BERT and ViT feature caches stay on the Modal volume. `notebooks/04_VLM_Pilot.ipynb` carries the outputs of the papermill rerun `modal_executions/20261107-104312` (28 minutes with all cues cached), which reproduced these numbers and drew the figures.

Run directory `outputs/qwen3_multimodal/<run_id>/`: `config.json`, `samples.json`, `prompts.json`, `environment.txt`, `batch_benchmark.json`, `qwen_cues.jsonl`, `bert_graph_features.npz`, `vit_features.npz`, `model.pt`, `metrics.json`, `predictions_eval.npz`, `comparison.json`, `evidence_packages.json`, `explanation_prompt_used.txt`, `explanations.jsonl`, `explanation_metrics.json`, and `figures/` (training curves, variant comparison, per-label AP). On Modal, the executed notebook and a timestamped `run.log` go to `outputs/modal_executions/<UTC start>/`. `metrics.json` holds the per-epoch train loss, dev mAP and seconds of the main model; `comparison.json` holds the same history for every variant run.

## Experiment matrix

| Experiment | Person | Context | VLM | Structured reasoning |
|---|:-:|:-:|:-:|:-:|
| A — Person only | ✓ |  |  |  |
| B — Context only |  | ✓ |  |  |
| C — Person + Context | ✓ | ✓ |  |  |
| D — Qwen3 cue graph | ✓ | ✓ | ✓ | ✓ |
| E — ViT + BERT + Graph/Multimodal Transformer | ✓ | ✓ | ✓ | ✓ |
| F — Evidence-grounded explanation | ✓ | ✓ | ✓ | ✓ |
