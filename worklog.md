# Worklog

## 2026-10-14

### Project initialization and organization

- Initialized the repository for **Explainable Contextual Emotion Recognition via Text-Guided Chain-of-Thought Reasoning**.
- Added the dependencies required for image processing, MATLAB annotation loading, PyTorch, visualization, configuration, and evaluation to `requirements.txt`.
- Configured `.gitignore` for the repository.

### EMOTIC data preparation

- Prepared both versions of the EMOTIC dataset: **PAMI** and **CVPR**.
- Added the PAMI `Annotations.mat` file and the CVPR `CVPR17_Annotations.mat` file.
- Extracted and organized the EMOTIC image sources, including `framesdb`, `mscoco`, `ade20k`, and `emodb_small`.

### Dataset loader

- Implemented `src/datasets/emotic_dataset.py` as a PyTorch `Dataset`.
- Added support for the `train`, `val`, and `test` splits in the EMOTIC annotations.
- Added support for one or more image roots so images can be resolved across different sources.
- Converted every annotated person into an independent sample, including images that contain multiple people.
- Implemented bounding-box handling, coordinate clamping to image dimensions, and optional padding around target-person crops.
- Added both the full context image and the target-person crop to each returned sample.
- Mapped the fixed set of 26 EMOTIC emotion labels to multi-hot tensors.
- Added `union` and `majority` policies for aggregating categorical annotations.
- Aggregated VAD annotations and created masks for samples without VAD values.
- Added strict missing-image validation and an option to skip samples whose images are unavailable during data inspection.

### Data inspection notebook

#### Purpose of `01_Inspect.ipynb`

- Created `01_Inspect.ipynb` as the data-validation stage that must be completed before model development or training.
- Used the notebook to verify the real structure of the original EMOTIC annotations instead of assuming a preprocessed schema.
- Verified that the PAMI and CVPR annotation files expose the expected `train`, `val`, and `test` splits.
- Counted image records and person annotations in each split to confirm that one image may contain multiple annotated target people.
- Checked that the first 100 annotation records of each split resolve to an existing image in the configured image roots.
- Inspected target-person bounding boxes, categorical emotion annotations, and continuous VAD annotations.
- Visualized a source image with every target-person bounding box drawn on it so annotation coordinates can be checked against the actual image.
- Kept this notebook focused on inspection and validation; it does not train a model or generate processed training data.

#### Data and tools used

- Loaded the PAMI `Annotations.mat` file and the CVPR `CVPR17_Annotations.mat` file.
- Resolved images across the `framesdb`, `mscoco`, `ade20k`, and `emodb_small` sources.
- Used `scipy.io.loadmat(..., simplify_cells=True)` to convert MATLAB annotation structures into Python dictionaries, lists, and NumPy values.
- Used `numpy` to normalize MATLAB arrays and parse bounding-box coordinates.
- Used `pathlib.Path` to construct relative image paths and search the configured image roots.
- Used `PIL.Image` to open source images in RGB format.
- Used `matplotlib` and `matplotlib.patches.Rectangle` to display images and draw target-person bounding boxes.
- Added helper functions for normalizing MATLAB values, loading available splits, counting people, constructing relative image paths, and resolving image files.
- Removed the dependency on importing `EmoticDataset`, allowing the notebook to inspect the raw EMOTIC files directly.
- Removed the previous import-error output and syntax-checked every code cell.

### Experimental workflow

- Converted `workflow.md` to a **notebook-first** workflow in which notebooks are the main entry points for training and evaluation.
- Defined the experimental sequence: data inspection, three visual baselines, direct VLM, structured-reasoning VLM, result analysis, and LoRA fine-tuning.
- Defined the three visual baselines: `person_only`, `context_only`, and `person_context`.
- Selected `BCEWithLogitsLoss` for the 26-class multi-label classification task.
- Selected validation mAP as the metric for choosing the best model.
- Designed the per-run output convention covering configuration, logs, metrics, predictions, figures, and checkpoints.
- Defined `last.pt` as the resumable training checkpoint and `best.pt` as the best validation checkpoint.
- Added guidance for resuming training and exporting a finalized model for inference.

## 2026-10-15

### Visual baseline notebook

- Added `02_Train_Visual_Baseline.ipynb`: ResNet-50 `person_only`, `context_only`, and `person_context` baselines trained with `BCEWithLogitsLoss`, AdamW, a cosine schedule, and AMP.
- Each run writes its configuration, environment, per-epoch history, latest validation metrics and predictions, `last.pt`, and `best.pt` (selected by validation mAP) to `outputs/runs/<experiment_name>/<run_id>/`.
- Added resuming from `last.pt` with a configuration compatibility check, and export of the best model to `models/`.

## 2026-10-19

### Faster image loading on Colab

- Stored the EMOTIC images on Drive as `pami_emotic.zip` and `cvpr_emotic.zip`; notebooks extract them to `/content/emotic` once per runtime and check the file counts.

### VLM pilot

- Added `04_VLM_Pilot.ipynb`, comparing direct prompting (D) with structured reasoning (E) on 300 PAMI validation people using `Qwen/Qwen2.5-VL-3B-Instruct`.
- The target person is marked with a red box; both modes share the model, image, label list, token limit, and greedy decoding.
- Answers are saved per sample to JSONL so an interrupted run can continue; metrics include macro/micro/sample F1, parse, schema, and truncation rates, and an image-level paired bootstrap CI for the E − D macro-F1 difference.
- Illustration examples are picked by a fixed rule: the best and worst schema-valid E answers by sample F1, ties broken by the shorter answer.

## 2026-10-31

- Updated `README.md`, `SETUP.md`, `workflow.md`, and `drivesetup.md` to match the notebooks and Drive layout in use.

### Visual baseline evaluation notebook

- Added `03_Evaluate_Visual_Baseline.ipynb`, loading the finalized `model.pt` exports for `person_only`, `context_only`, and `person_context` from either `models/` or the current Drive `checkpoints/` directory.
- Added validation-only per-label threshold selection, union/majority sensitivity analysis, test mAP/AP/F1/precision/recall, a validation-selected constant top-k baseline, evaluation without `Engagement`, and F1 grouped by the number of annotated people in an image.
- Added image-level paired bootstrap comparisons among A/B/C, region ablations for the `person_context` model, and target-swap evaluation on multi-person images with an explicit swap-accuracy definition.
- Added reusable multi-label metrics and bootstrap utilities under `src/evaluation/`; Notebook 03 writes versioned configs, environment details, predictions, metrics, and figures to `outputs/resnet_evaluation/<run_id>/`.
- Completed Notebook 03 on a Colab Tesla T4. Run `20261101-034030` is stored on Drive under `outputs/resnet_evaluation/` with 27 artifacts and source commit `0e4b81fc`.
