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
│   ├── qwen3_multimodal/<run_id>/
│   └── modal_executions/<UTC start>/   # Modal volume only
├── models/<model_name>/<run_id>/
└── checkpoints/<model_name>/<run_id>/  # where the three ResNet exports currently are
```

Each notebook extracts both zips into `/content/emotic/{PAMI,CVPR}` once per runtime (marker file `.extracted`) and asserts the file counts above. Reading tens of thousands of small images directly from Drive starves the GPU. Training, evaluation, and notebook 04 read the PAMI annotations and resolve images across both extracted roots.

## Notebooks

| Notebook | Status | What it does |
|---|---|---|
| `01_Inspect.ipynb` | done | Loads PAMI and CVPR annotations, counts images and people per split, checks that the first 100 records of each split resolve to an image, draws every person's bbox on one sample and prints its categories and VAD. |
| `02_Train_Visual_Baseline.ipynb` | done | Trains one visual baseline per run (`person_only`, `context_only`, `person_context`), selects `best.pt` by validation mAP, exports the best model. |
| `03_Evaluate_Visual_Baseline.ipynb` | done | Loads the three exported models, fixes thresholds and constant top-k on validation, then runs test metrics, union/majority sensitivity analysis, image-level paired bootstrap, region ablation, and target swap. |
| `04_Qwen3_Multimodal.ipynb` | done; run `20261104-071259-h100` on Modal H100 | Generates Qwen3-VL cues for all 23,706 PAMI train and 3,334 PAMI val people and trains on all train people. BERT encodes a semantic graph, ViT encodes person/context, and Graph/Multimodal Transformers predict emotions. Compares full, ViT-only, no-context and no-person variants, explains every eval person and scores the explanations with agreement, cited-number, unpredicted-emotion and ERASER comprehensiveness/sufficiency metrics. Section 13 scores the frozen pipeline once on all 7,280 PAMI test people, adds a variant on cues with affect words removed, and compares with the notebook 03 ResNet-50 predictions on the same people; section 14 builds evidence and explanations for every test person. |
| groundedness review, region/relation attribution | planned | Manually review cue and explanation grounding, then add region occlusion, Integrated Gradients and relation ablation. |

The completed Colab T4 run `20261101-034030` of notebook 03 is stored on Drive under `outputs/resnet_evaluation/` and records source commit `0e4b81fc`; section 13b of notebook 04 reads its thresholds and test predictions.

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
- Resume: set `RESUME_CHECKPOINT` to a `last.pt`. The notebook reuses that run directory and its saved `config.json` (the `TRAIN_CONFIG` typed in the cell is ignored) and stops if the checkpoint's `experiment_name`, `backbone`, `image_size`, `category_policy` or `crop_padding` differ from it. `RESUME_TOTAL_EPOCHS` only raises the number of epochs the loop runs: the restored cosine schedule keeps its old length and `config.json` in the run directory keeps the old epoch count.
- Export `models/emotic_<experiment_name>_<backbone>/<run_id>/`: `model.pt` (state dict only), `config.json`, `label_map.json`, `metrics.json` (validation metrics of the best epoch), `README.md`. Re-running the export cell overwrites these files.

Run the three configurations one after another, restarting the runtime between them.

## Explainable multimodal prototype (`04`)

- Uses all 23,706 PAMI train people for training. All 3,334 PAMI val people are divided by image into dev/eval, so people from the same validation image cannot cross partitions. Only sections 13 and 14 read the test split, with the checkpoint, thresholds and `k` already fixed on dev.
- Qwen3-VL 4B receives the full image with the target marked by a red box and extracts visible person cues, context cues, relations and a situation without seeing the emotion label list.
- Frozen BERT encodes those semantic nodes; frozen ViT-B/16 encodes the person crop and full image. A trainable Graph Transformer and Multimodal Transformer predict the 26 labels.
- The best checkpoint and per-label thresholds are selected only on the internal dev partition. `eval` metrics refer to a held-out part of PAMI val; `test` metrics refer to all 7,280 PAMI test people.
- Person, context and graph tokens are ablated one at a time to create a preliminary evidence package. Qwen3-VL then gets the image with the target marked plus this evidence package, which also carries the cues Qwen extracted for that person, and is told to explain from the evidence only; nothing stops it from using the image. Token-zero ablation is a prototype intervention and does not replace region/relation attribution.
- Cue generation benchmarks batch sizes 16/32/48 on the first 16, 32 and 48 people (comparing with their saved cues when they exist) and keeps the fastest one with at least 95% schema-valid output. It projects the hours left and stops if they exceed `CUE_BUDGET_HOURS`, then generates cues for every missing person, closing the JSONL every `BATCHES_PER_SYNC` batches so Drive syncs it; later cells refuse to run until every person has a cue; BERT and ViT features are cached. Resume refuses a changed config (except `git_commit`) or prompt; a new commit appends the current environment to `environment.txt`.
- Variants `full`, `vit_only` (graph token zeroed), `no_context` and `no_person` are retrained with seeds 42-44 on the same split, compared with constant top-k and an image-level paired bootstrap. Section 13 retrains these four plus `full_no_affect` (22 affect words deleted from every cue before BERT) with seeds 42-44 and scores each on eval and test; 13b compares with the notebook 03 ResNet-50 test predictions. Explanations for every eval person (section 12) and every test person (section 14) are scored for modality agreement against the largest intervention drop, cited-number accuracy (decimals and percentages only), mentions of unpredicted emotions (whole-word matches of label names, which cannot catch `Doubt/Confusion` or synonyms), and ERASER comprehensiveness and sufficiency against the most frequent, a random and the measured modality. Integer sample limits remain available only for smoke tests; the official configuration uses `None` to select every person.

Run `20261104-071259-h100` (Drive `outputs/qwen3_multimodal/20261104-071259-h100/`) continues minhtb's run `20261104-071259`: 2,275 train cues came from unbatched generation on a T4, the other 24,765 from batch 48 on Colab A100 and Modal H100. The Modal H100 session that generated the last 23,469 cues took about 1.5 hours end to end, 49 minutes of it on the cues. Section 13, mean over seeds 42-44, mAP eval / test: full 40.5 / 28.5, ViT-only 37.5 / 27.0, no-context 40.1 / 29.8, no-person 39.8 / 30.4, affect-free cues 39.7 / 30.0; constant top-26 23.8 / 17.0; best ResNet-50 (context only) 27.35 on test. Affect-free minus ViT-only macro-F1 on seed 42: +2.8 (95% CI 0.7 to 5.2) on eval and +2.7 (1.9 to 3.4) on test; full minus ViT-only on test -0.3 (-1.1 to 0.5). On eval, 1,667 of 1,691 explanations are usable; 99.99% of cited numbers are in the evidence, but the VLM names the person in 1,660 of them and agrees with the largest-drop modality in 38.3% against 40.3% for always naming the most common one; ERASER comprehensiveness / sufficiency 0.072 / 0.123 for the stated modality, 0.078 / 0.120 for the most common one. On test, 7,193 of 7,280 explanations are usable; 99.97% of cited numbers are in the evidence, the VLM names the person in 7,158, and the person is also the most common largest-drop modality, so agreement (38.8%) and ERASER (0.077 / 0.125) match those of always naming the person at this precision. BERT and ViT feature caches stay on the Modal volume. `notebooks/04_Qwen3_Multimodal.ipynb` carries the outputs of the papermill run `modal_executions/20261107-133419`, which retrained the main model to the same dev and eval metrics as the earlier H100 session and drew the figures; the config, prompt, Qwen loading, cue benchmark and explanation cells were edited after that run without changing what it computes.

Run directory `outputs/qwen3_multimodal/<run_id>/`: `config.json`, `samples.json`, `prompts.json`, `environment.txt`, `batch_benchmark.json`, `qwen_cues.jsonl`, `bert_graph_features.npz`, `vit_features.npz`, `model.pt` (a dict with `model_state_dict`, `model_config`, `class_names` and `best_dev_map`), `metrics.json`, `predictions_eval.npz`, `comparison.json`, `evidence_packages.json`, `explanations.jsonl`, `explanation_metrics.json` (keys `eval` and `test`), `qwen_cues_test.jsonl`, `test_features.npz`, `affect_free_features.npz`, `test_results.json`, `predictions_test.npz`, `evidence_packages_test.json`, `explanations_test.jsonl`, and `figures/` (training curves, variant comparison, per-label AP). On Modal, the executed notebook and a timestamped `run.log` go to `outputs/modal_executions/<UTC start>/`. `metrics.json` holds the per-epoch train loss, dev mAP and seconds of the main model; `comparison.json` holds the same history for every variant run.
