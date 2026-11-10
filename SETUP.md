# Setup trên Google Colab

Dữ liệu EMOTIC, checkpoint và model chỉ nằm trên Google Drive, không đưa lên GitHub.

## 1. Chuẩn bị Drive

Đặt dữ liệu đúng các đường dẫn mà notebook đọc, dưới `MyDrive/Colab Notebooks/Contextual Emotion Recognition/`:

```text
data/raw/raw/emotic/PAMI/Annotations/Annotations/Annotations.mat
data/raw/raw/emotic/PAMI/pami_emotic.zip      # thư mục gốc emotic/, 23.185 file
data/raw/raw/emotic/CVPR/CVPR17_Annotations.mat
data/raw/raw/emotic/CVPR/cvpr_emotic.zip      # thư mục gốc cvpr_emotic/, 18.302 file
```

Giữ ảnh ở dạng zip. Notebook tự giải nén vào `/content/emotic` một lần mỗi runtime và kiểm số file; nếu số file khác thì notebook dừng.

## 2. Chạy notebook

1. Mở notebook từ `notebooks/` trên Drive (hoặc từ GitHub qua **File → Open notebook → GitHub**).
2. Chọn **Runtime → Change runtime type → GPU**. Notebook 02, 03 và 04 dừng nếu không có GPU.
3. Chạy từ cell đầu. Cell đầu mount Drive. Ở 02, 03 và 04, cell này còn clone repo vào `/content/contextual-emotion-reasoning` (hoặc `git pull --ff-only` nếu đã có), cài `requirements.txt` và thêm repo vào `sys.path`.

Sau khi runtime reset, chạy lại từ cell đầu; dữ liệu và kết quả trên Drive vẫn còn.

## 2b. Chạy notebook 04 trên Modal

Với tài khoản hiện tại có Volume `emotic` (không có `ecer-nb04`), dùng runner
snapshot local để chạy bản chưa push:

```powershell
$env:PYTHONUTF8 = "1"
python -m modal run --detach cloud/modal_nb04_local.py
```

Runner kiểm tra annotation, zip và artifact H100 trước khi cấp H100; không tự
tạo Volume. Cần thêm `prompts.json`, `qwen_cues.jsonl`, `qwen_cues_test.jsonl`,
`test_results.json`, `evidence_packages.json`, `explanations.jsonl` vào
`outputs/qwen3_multimodal/20261104-071259-h100/` trên Volume, ngoài các file
đã dùng cho notebook 05. Runner này tính lại feature (`feature_import_from=None`),
dùng lại cue, mặc định giữ `RUN_TEST=False`, và vẫn so metric eval với H100.
Notebook Colab gốc không bị sửa. `git_commit` ghi `snapshot-sha256:<hash>`;
manifest hash từng file nằm trong `code_snapshot.json`. Run mới ghi vào
`outputs/multimodal_pipeline/<run_id>/`; notebook đã chạy, input đã chuyển
đường dẫn, `run.log` và `status.json` nằm trong
`outputs/modal_executions/<UTC start>-local/`; `pipeline_run.json` liên kết tới
run ID do notebook tạo. Volume được commit mỗi 5 phút
và khi hàm kết thúc. Runner local không tự retry; khi lỗi cần xem trạng thái
trước khi khởi chạy lại để tránh tạo run mới ngoài ý muốn. Bản đã chạy có thay
đổi cấu hình runtime nên giữ riêng, không dùng `attach_outputs.py` của runner
clone nhánh Git bên dưới.

Để chấm test cho run đã chốt, upload `metrics/thresholds.json` và ba file
`predictions/{person_only,context_only,person_context}_test.npz` của run ResNet
`outputs/resnet_evaluation/20261101-034030/` từ Drive lên đúng đường dẫn trên
Volume. Sau đó tiếp tục run, dùng lại checkpoint, threshold và cache đã có:

```powershell
python -m modal run --detach cloud/modal_nb04_local.py --run-id 20261110-074559 --run-test
```

Preflight kiểm tra đủ checkpoint của mọi biến thể × seed và đầu vào so sánh
ResNet trước khi cấp GPU. Mục test đọc lại kết quả nếu `test/test_results.json`
đã tồn tại; không đổi cấu hình theo kết quả test.

Cần `pip install modal` và token Modal (`modal token new`). Dùng PowerShell; Git Bash đổi đường dẫn `/data/...` thành đường dẫn Windows.

```powershell
modal volume create ecer-nb04
modal volume put ecer-nb04 Annotations.mat /data/raw/raw/emotic/PAMI/Annotations/Annotations/Annotations.mat
modal volume put ecer-nb04 pami_emotic.zip /data/raw/raw/emotic/PAMI/pami_emotic.zip
modal volume put ecer-nb04 cvpr_emotic.zip /data/raw/raw/emotic/CVPR/cvpr_emotic.zip
modal volume put ecer-nb04 <thư mục run> /outputs/multimodal_pipeline/<RUN_ID>
modal run --detach cloud/modal_nb04.py::run --branch main
modal volume get ecer-nb04 /outputs/multimodal_pipeline/<RUN_ID> <thư mục local>
modal volume get ecer-nb04 /outputs/modal_executions/<giờ bắt đầu>/04_VLM_Pilot.ipynb <file local>
python cloud/attach_outputs.py <file local>
```

Mặc định notebook nhập cue và feature từ `/outputs/qwen3_multimodal/20261104-071259-h100`, nên thư mục đó phải có trên volume. Bước `put` thư mục run chỉ cần khi tiếp tục một run có sẵn; `RUN_ID` lấy theo cell cấu hình của notebook. Script chạy chính notebook bằng papermill trên H100: cell clone repo được thay bằng cell trỏ vào volume, ảnh giải nén vào `/tmp/emotic`, notebook đã chạy (kèm output, hình và thời gian từng cell) cùng `run.log` có timestamp nằm ở `/outputs/modal_executions/<giờ bắt đầu UTC>/`. `cloud/attach_outputs.py` chép output đó vào `notebooks/04_VLM_Pilot.ipynb` để commit, giữ nguyên code Colab. Script tự commit volume 5 phút một lần và khi notebook chạy xong; mất kết nối thì Modal chạy lại hàm, notebook resume từ cue, feature và cặp biến thể × seed đã có. Papermill chạy notebook như trong repo, nên `RUN_ID` và `RUN_TEST` phải được đặt trong notebook và đẩy lên nhánh truyền vào `--branch`. Kết quả nằm trên volume, phải tự tải về và đưa lên Drive.

## 3. Thứ tự

1. `01_Inspect.ipynb`: kiểm annotation, đường dẫn ảnh và bbox.
2. `02_Train_Visual_Baseline.ipynb`: lần lượt `person_only`, `context_only`, `person_context`; restart runtime giữa các lần.
3. `03_Evaluate_Visual_Baseline.ipynb`: chấm ba baseline trên test, chọn ngưỡng trên validation.
4. `04_VLM_Pilot.ipynb`: Qwen3-VL 4B + BERT + backbone ảnh + Graph/Multimodal Transformer, chạy tuần tự các mục cue (train, val, test), feature, huấn luyện mọi biến thể × seed, explanation trên eval, và test (chỉ khi `RUN_TEST = True`, sau khi chốt trên dev). Cue, feature và từng cặp biến thể × seed đều resume được với cùng `RUN_ID`. Bản đã chạy của run H100 theo layout cũ nằm ở `outputs/modal_executions/20261107-133419/04_VLM_Pilot.ipynb` trên volume Modal `ecer-nb04` (trong git: commit `e304c0ca`).

Kết quả ghi vào `outputs/` và `models/` trên Drive; cấu trúc chi tiết trong `workflow.md`.
