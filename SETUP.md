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
2. Chọn **Runtime → Change runtime type → GPU**. Notebook 02 và 04 dừng nếu không có GPU.
3. Chạy từ cell đầu. Cell đầu mount Drive, clone repo vào `/content/contextual-emotion-reasoning` (hoặc `git pull --ff-only` nếu đã có), cài `requirements.txt` và thêm repo vào `sys.path`.

Sau khi runtime reset, chạy lại từ cell đầu; dữ liệu và kết quả trên Drive vẫn còn.

## 3. Thứ tự

1. `01_Inspect.ipynb`: kiểm annotation, đường dẫn ảnh và bbox.
2. `02_Train_Visual_Baseline.ipynb`: lần lượt `person_only`, `context_only`, `person_context`; restart runtime giữa các lần.
3. `04_VLM_Pilot.ipynb`: pilot D vs E. GPU T4 chạy được model 3B mặc định.

Kết quả ghi vào `outputs/` và `models/` trên Drive; cấu trúc chi tiết trong `workflow.md`.
