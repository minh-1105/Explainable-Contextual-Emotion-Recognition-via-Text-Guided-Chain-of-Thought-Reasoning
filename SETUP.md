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

Cần `pip install modal` và token Modal (`modal token new`). Dùng PowerShell; Git Bash đổi đường dẫn `/data/...` thành đường dẫn Windows.

```powershell
modal volume create ecer-nb04
modal volume put ecer-nb04 Annotations.mat /data/raw/raw/emotic/PAMI/Annotations/Annotations/Annotations.mat
modal volume put ecer-nb04 pami_emotic.zip /data/raw/raw/emotic/PAMI/pami_emotic.zip
modal volume put ecer-nb04 cvpr_emotic.zip /data/raw/raw/emotic/CVPR/cvpr_emotic.zip
modal volume put ecer-nb04 <thư mục run> /outputs/qwen3_multimodal/<RUN_ID>
modal run --detach cloud/modal_nb04.py::run --branch main
modal volume get ecer-nb04 /outputs/qwen3_multimodal/<RUN_ID> <thư mục local>
```

Bước `put` thư mục run chỉ cần khi tiếp tục một run có sẵn; `RUN_ID` lấy theo cell cấu hình của notebook. Script chạy trên H100, ghi toàn bộ output kèm timestamp UTC vào `logs/modal_<giờ bắt đầu>.log` trong thư mục run (training log để nộp), tự commit volume 5 phút một lần và sau mỗi cell; mất kết nối thì Modal chạy lại hàm, notebook resume từ cue đã có. Kết quả nằm trên volume, phải tự tải về và đưa lên Drive.

## 3. Thứ tự

1. `01_Inspect.ipynb`: kiểm annotation, đường dẫn ảnh và bbox.
2. `02_Train_Visual_Baseline.ipynb`: lần lượt `person_only`, `context_only`, `person_context`; restart runtime giữa các lần.
3. `03_Evaluate_Visual_Baseline.ipynb`: chấm ba baseline trên test, chọn ngưỡng trên validation.
4. `04_VLM_Pilot.ipynb`: prototype Qwen3-VL 4B + BERT + ViT + Graph/Multimodal Transformer. Qwen sinh cue cho toàn bộ 23.706 người PAMI train và 3.334 người PAMI val; cell cue dừng ngay nếu dự kiến vượt `CUE_BUDGET_HOURS`. Cue ghi JSONL để resume qua nhiều phiên Colab với cùng `RUN_ID`.

Kết quả ghi vào `outputs/` và `models/` trên Drive; cấu trúc chi tiết trong `workflow.md`.
