# Hướng dẫn viết lại Notebook 05 theo Notebook 04 hiện tại

Tài liệu cho agent (Codex) viết lại `notebooks/05_Compare_Multimodal_Baselines.ipynb` và
`cloud/modal_nb05.py`. Mọi đường dẫn, khóa và số liệu dưới đây đã đối chiếu với volume Modal
`emotic` ngày 2026-11-10. Đọc thêm `AGENTS.md`, `workflow.md`, và code của 04 (`notebooks/04_VLM_Pilot.ipynb`,
`src/evaluation/emotion_metrics.py`) trước khi code.

## 0. Bối cảnh và ràng buộc

- Run nguồn: `outputs/multimodal_pipeline/20261110-074559/` (Notebook 04 mới, ViT-B/16, cue import từ
  run H100, feature mã hóa lại). Team đã quyết định **không cần tái lập run H100**; đây là run chính
  thức mới. Không so số với `outputs/qwen3_multimodal/20261104-071259-h100` nữa.
- Mục 7 của 04 **đã chấm test** cho run này (`RUN_TEST=True`, execution `20261110-093212-local`), kể cả
  bootstrap với ResNet trên test. Vì vậy 05:
  - **không đọc** `test/`, `cues/cues_test.jsonl`, `features/*/test.npz`, `features/bert/*_test.npz`
    hay bất cứ file test nào;
  - **không dùng** kết quả của mình để đổi biến thể chính (`full`), ngưỡng, seed hay cấu hình 04;
  - các mô hình chỉ có trong 05 (`full_ensemble`, late fusion, top-k theo mẫu) là **phân tích chỉ
    trên eval**, không được chấm test sau đó. Ghi rõ trong markdown và `metrics.json["limitations"]`.
- 05 không gọi Qwen, không train, không tính feature, không inference ResNet → **không cần GPU**,
  không cần ảnh, không cần `extract_images`.
- **Chỉ chạy trên Modal** (volume `emotic`, mount ở `/vol`, cùng layout với thư mục Drive cũ). Không
  viết cell `drive.mount`, không clone repo, không hỗ trợ Colab.
- Quy ước `AGENTS.md`: markdown/thông báo tiếng Việt; tên biến, hàm, docstring trong `src/` tiếng Anh;
  output vào `outputs/multimodal_comparison/<run_id>/` kèm `config.json` và `environment.txt` có
  `git_commit`.

## 1. Đầu vào (chỉ đọc)

Nguồn 04, `SOURCE_RUN = "outputs/multimodal_pipeline/20261110-074559"`:

| File | Nội dung dùng |
|---|---|
| `config.json` | `PIPELINE_CONFIG` của 04: `variants` (7 biến thể), `seeds` (`[42, 43, 44]`), `threshold_grids` (`legacy`, `extended`), `category_policy="union"`, `primary_variant="full"`. Đọc danh sách biến thể, seed và lưới từ đây, không hard-code. |
| `samples.json` | `splits["dev"]` (1.643 người), `splits["eval"]` (1.691 người), `counts`, `validation_split_unit="relative_image_path"`. `sample_id` dạng `<split>:<image_index>:<person_index>`. |
| `train/predictions/<variant>_seed<seed>.npz` | 21 file. Khóa: `dev_sample_ids`, `dev_probabilities` (1643×26, float32), `eval_sample_ids`, `eval_probabilities` (1691×26), `thresholds_legacy`, `thresholds_extended` (26, float64). **Không có nhãn.** |
| `train/metrics.json` | `summary[variant][grid][part][metric] = {mean, sd}`; `runs` là list dòng `{variant, seed, grid, split, map, macro_f1, micro_f1, mean_predicted_labels, thresholds_at_grid_min}`. Chỉ dùng để **kiểm tra**, không lấy làm số báo cáo. |
| `features/vit_b_16_imagenet1k_v1/val.npz` | Chỉ đọc `sample_ids` và `labels` (multi-hot union theo `EMOTION_LABELS`, cùng thứ tự `samples.json["sample_ids"]` phần val) để đối chiếu nhãn. Không đọc mảng feature. |
| `code_snapshot.json`, `code_snapshot_*.json`, `environment.txt` | Ghi vào provenance của 05. |

Nguồn ResNet, `RESNET_RUN = "outputs/multimodal_comparison/20261109-090105-378575"` (run 05 cũ, H100,
snapshot `272091817d75…`). Không inference lại:

| File | Khóa |
|---|---|
| `predictions_dev.npz` | `sample_ids`, `image_ids`, `targets` (1643×26), `class_names`, `person_only`, `context_only`, `person_context` (xác suất). |
| `predictions_eval.npz` | như trên cho eval, thêm `<name>_thresholds` (lưới mở rộng, chọn trên dev), `<name>_legacy_thresholds`, `constant_predictions`; `full_original`, `full_thresholds` là của H100 → **bỏ qua**. |
| `selected_thresholds.json`, `metrics.json`, `config.json` | Để kiểm tra số ResNet. |

Đã xác minh: `sample_ids` dev/eval của ResNet và của `train/predictions/full_seed42.npz` **trùng thứ tự**
với `samples.json["splits"]`. Notebook vẫn assert lại cho cả 21 file.

## 2. Cấu trúc notebook

1. **Môi trường.** Cell đầu chỉ đặt `DRIVE_ROOT = Path("/vol")`, `CODE_ROOT` (thư mục runner đặt
   `src/`), thêm `CODE_ROOT` vào `sys.path`, và đọc `GIT_COMMIT` từ biến môi trường runner truyền vào
   (`snapshot-sha256:<sha>`). Assert `DRIVE_ROOT / SOURCE_RUN` tồn tại. Không assert GPU.
2. **Cấu hình và tạo run.** `CONFIG` gồm `source_run`, `resnet_run`, `primary_variant` (đọc từ config 04),
   `bootstrap_resamples=2000`, `bootstrap_seed=42`, `bootstrap_metrics=["map", "macro_f1"]`,
   `threshold_grid = config_04["threshold_grids"]["extended"]` (không tự định nghĩa lại lưới),
   `fusion_weight_grid` (ví dụ `np.linspace(0, 1, 21)`), `top_k_grid` (1..26),
   `error_examples_per_direction=10`, `git_commit`. Tạo `RUN_DIR` mới mỗi lần (05 không resume).
3. **Nạp và kiểm tra nguồn.**
   - SHA-256 mọi file nguồn đã đọc → `source_artifacts.json`.
   - Nhãn: `targets[part]` lấy từ `predictions_{part}.npz` của ResNet, assert bằng `labels` của
     `features/vit_b_16_imagenet1k_v1/val.npz` sau khi lấy đúng hàng theo `sample_id`.
     Assert `class_names == EMOTION_LABELS`.
   - `image_ids[part]` lấy từ ResNet npz; assert dev và eval không chung ảnh.
4. **Tái lập metric của 04 (bắt buộc, lệch là dừng).** Với mọi biến thể × seed × grid × part:
   `multilabel_metrics(targets, probs, thresholds=npz[f"thresholds_{grid}"], class_names=EMOTION_LABELS)`,
   assert `map`, `macro_f1`, `micro_f1` khớp dòng tương ứng trong `train/metrics.json["runs"]`
   (`np.isclose`, atol 1e-9). Chọn lại `select_per_class_thresholds(targets["dev"], dev_probabilities, grid)`
   và assert bằng ngưỡng đã lưu. Ghi kết quả vào `reproduction_checks.json`.
5. **Kiểm tra ResNet.** Tính lại metric eval của `person_only`/`context_only`/`person_context` với
   `<name>_thresholds`, assert khớp `metrics.json` của `RESNET_RUN`; chọn lại ngưỡng trên dev bằng cùng
   lưới và assert bằng ngưỡng đã lưu.
6. **Bảng chính trên eval.** Dòng: ResNet A/B/C; mỗi biến thể 04 (mean ± SD qua 3 seed, kèm từng seed);
   constant top-k (`select_constant_top_k` trên dev, áp lên eval). Cột: `map`, `macro_f1`, `micro_f1`,
   các metric khác `multilabel_metrics` trả về (samples F1, precision, recall nếu có),
   `mean_predicted_labels`, số lớp ở đáy lưới (`threshold_boundary_report`). Lưới chính `extended`,
   `legacy` làm cột đối chiếu. Ghi `metrics.csv`, `metrics.json`, `per_class_metrics.csv` (AP, F1 theo lớp).
7. **Bootstrap ghép cặp theo ảnh trên eval** (`paired_image_bootstrap`, `image_ids` từ mục 3):
   - từng seed: `full` − mỗi biến thể còn lại (`vit_only`, `no_person_image`, `no_context_image`,
     `no_person_cues`, `no_context_cues`, `full_no_affect`) và `full` − ResNet A/B/C;
   - `metric="map"` dùng xác suất; `metric="macro_f1"` dùng nhị phân theo ngưỡng `extended`;
   - tổng hợp: CI từng seed và số seed có CI không chứa 0; không gộp p-value.
   - Eval chỉ có 1.044 ảnh, CI của mAP rộng khoảng ±0,014, còn các ablation chỉ lệch `full`
     0,001–0,008 mAP. Khi CI chứa 0, kết luận viết "không phát hiện đóng góp rõ", không viết
     "không có tác dụng".
8. **Phân tích bổ sung chỉ trên eval** (tham số chọn trên dev, đóng băng trước khi tính eval):
   - `full_ensemble`: trung bình xác suất 3 seed `full`; chọn lại ngưỡng trên dev.
   - late fusion `full_ensemble` + ResNet `person_context`: `w * p_full + (1 - w) * p_resnet`, `w` chọn
     theo dev mAP trên `fusion_weight_grid`, rồi chọn ngưỡng trên dev.
   - top-k theo mẫu: mỗi người dự đoán `k` nhãn có xác suất cao nhất, `k` chọn theo dev macro-F1, áp
     cùng luật cho mọi model.
   - Hàm trộn/chọn `w` và top-k theo mẫu đặt trong `src/evaluation/` (tên tiếng Anh, docstring, thêm
     vào `__all__`), không viết trong notebook.
   - Bootstrap `full_ensemble` − ResNet C và late fusion − ResNet C như mục 7.
9. **Theo lớp và theo số người trong ảnh.** Số người của ảnh = số `sample_id` có cùng `image_id` trong
   dev ∪ eval (đủ chính xác vì val chia theo ảnh, mọi người của một ảnh nằm cùng phần; không cần mở
   annotation hay ảnh). Nhóm 1 / 2 / 3–4 / ≥5; mAP và macro-F1 mỗi nhóm cho mọi dòng mục 6 →
   `people_group_metrics.json`. `per_class_ap_delta.csv`: AP(`full` mean 3 seed) − AP(ResNet C) theo lớp,
   kèm số dương trên dev và eval.
10. **Ví dụ lỗi** theo quy tắc cố định như 05 cũ: `error_examples_per_direction` người mà `full` đúng
    hơn ResNet C và ngược lại, sắp theo chênh lệch F1 từng người rồi `sample_id`. Chỉ ghi `sample_id`,
    `image_id`, nhãn, dự đoán.
11. **Hình** vào `figures/`: mAP và macro-F1 theo dòng (error bar SD seed), forest plot bootstrap,
    AP delta theo lớp, metric theo nhóm số người.
12. **Bàn giao** (markdown tiếng Việt): kết luận, file cần đọc, `LIMITATIONS`: ResNet một seed;
    checkpoint ResNet chọn trên toàn bộ PAMI val (gồm cả eval); feature 04 mã hóa lại nên không
    bằng H100 từng bit; các mô hình mục 8 không được chấm test.

## 3. Output `outputs/multimodal_comparison/<run_id>/`

`config.json`, `environment.txt` (`git_commit` hoặc `snapshot-sha256:…`), `code_snapshot.json`,
`source_artifacts.json` (đường dẫn và SHA-256 nguồn; snapshot run 04: `259a472f…` lúc train,
`e723050c…` lúc test; snapshot ResNet `272091817d75…`), `reproduction_checks.json`, `metrics.csv`,
`metrics.json`, `per_class_metrics.csv`, `per_class_ap_delta.csv`, `bootstrap.json`,
`people_group_metrics.json`, `selected_parameters.json` (ngưỡng, `w`, `k`, constant `k`),
`error_examples.json`, `predictions_dev.npz` và `predictions_eval.npz` (xác suất mọi dòng, kể cả
ensemble và late fusion, cùng `sample_ids`, `image_ids`, `targets`, `class_names`), `figures/`.

## 4. Runner `cloud/modal_nb05.py`

- `SOURCE_RUN` → `outputs/multimodal_pipeline/20261110-074559`. `SOURCE_FILES` → `config.json`,
  `samples.json`, `train/metrics.json`, 21 file `train/predictions/*.npz`,
  `features/vit_b_16_imagenet1k_v1/val.npz`; thêm `predictions_dev.npz`, `predictions_eval.npz`,
  `metrics.json`, `selected_thresholds.json` của `RESNET_RUN`.
- Preflight assert đủ file và không có đường dẫn nào chứa `test` trong danh sách đọc.
- Bỏ GPU (CPU 4–8, RAM 16 GB). Giữ snapshot SHA-256 của `src/`, notebook, `requirements.txt` và
  runner; `status.json`; commit Volume định kỳ và khi kết thúc.
- Notebook đã viết cho Modal nên runner không cần sửa cell; chỉ upload `src/` + notebook, truyền
  snapshot qua biến môi trường, chạy bằng papermill và lưu bản đã chạy vào
  `outputs/modal_executions/<timestamp>-nb05/` như `cloud/modal_nb04_local.py`.
- Lệnh: `modal run cloud/modal_nb05.py --check-only` (preflight), rồi `modal run --detach cloud/modal_nb05.py`.

## 5. Kiểm tra trước khi giao

- Lệnh `ast.parse` trong `AGENTS.md`: `ok`.
- Test nhỏ cho hàm mới trong `src/evaluation/` với dữ liệu giả.
- Chạy 05 trên Modal: mục 4 và 5 qua hết assert; `full` extended eval mAP = 0,4006 ± 0,0030 (khớp
  `train/metrics.json`); ResNet C eval mAP ≈ 0,406 (khớp run 05 cũ).
- Cập nhật `workflow.md`: output của 05, nguồn mới, và câu "Notebook 05 still reads the H100 run".
- Snapshot chỉ lưu hash, không lưu nội dung code. Code 04 mới (`src/models/`, `src/reasoning/`,
  `src/utils/`, `src/datasets/validation_split.py`), thay đổi `src/evaluation/` và code 05 vẫn chưa
  commit; nên commit để snapshot của các run còn truy lại được code.
