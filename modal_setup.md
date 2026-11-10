# Chạy notebook trên Modal

## Notebook 05: runner đã cấu hình

`cloud/modal_nb05.py` triển khai app `emotic-nb05`, dùng GPU H100, 8 CPU,
64 GiB RAM và Volume có sẵn `emotic`. Có thể chọn Volume khác bằng biến
`NB05_VOLUME` trước khi deploy. Runner không tự tạo Volume.

```powershell
$env:PYTHONUTF8 = "1"  # log Unicode của Modal trên Windows
python cloud/upload_nb05_inputs.py --source "G:\My Drive\Colab Notebooks\Contextual Emotion Recognition" --volume emotic
python -m modal deploy cloud/modal_nb05.py
python -m modal run cloud/modal_nb05.py --check-only
python -m modal run --detach cloud/modal_nb05.py
```

Uploader chỉ đưa bảy artifact cần thiết của run H100 và `model.pt`,
`config.json`, `label_map.json` của ba export ResNet lên Volume, không ghi đè
file có sẵn. Annotation và hai zip ảnh phải có sẵn theo layout bên dưới.
Preflight chạy CPU trước; thiếu đầu vào thì dừng trước khi cấp GPU.

Runner này đóng gói notebook và `src/` từ máy local, thay cell mount/clone bằng
thiết lập Volume và khóa đường dẫn export. Không cần push Git. Các cell đánh
giá còn lại giữ nguyên; notebook gốc không bị sửa. `code_snapshot.json` ghi
SHA-256 từng file; trường `git_commit` trong environment dùng giá trị
`snapshot-sha256:<hash>` để phân biệt rõ code chưa commit với commit Git.
Kết quả vẫn ở `outputs/multimodal_comparison/<run_id>/`; bản notebook đã chạy
và `status.json` nằm ở `outputs/executed_notebooks/nb05-<timestamp>/`.
Volume được commit cả khi notebook báo lỗi. Không có lịch chạy tự động.

Hướng dẫn này chạy các notebook 01–05 trên [Modal](https://modal.com) thay cho Colab, **không sửa notebook**. Ý tưởng: dựng lại đúng môi trường mà notebook đang giả định.

- Dữ liệu đặt trên một Modal Volume. Đường dẫn Drive cứng `/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition` là symlink trỏ vào Volume đó.
- Module `google.colab` được thay bằng một bản giả, trong đó `drive.mount()` không làm gì.
- `/content/emotic` (ảnh giải nén) và `/content/contextual-emotion-reasoning` (repo clone) nằm trên ổ tạm của container, giống runtime Colab.

Cell clone ở đầu 02–05 vẫn clone hoặc `git pull --ff-only` nhánh `main`, nên mọi quy tắc trong `AGENTS.md` về `main`, `git_commit` và resume vẫn áp dụng như trên Colab.

## 1. Cài Modal

Trên máy local:

```bash
pip install modal
modal setup              # mở trình duyệt, đăng nhập và lưu token vào ~/.modal.toml
```

## 2. Tạo Volume và đưa dữ liệu lên

```bash
modal volume create emotic
```

Volume sẽ giữ đúng layout của thư mục project trên Drive:

```text
emotic (Volume)
├── data/raw/raw/emotic/
│   ├── PAMI/Annotations/Annotations/Annotations.mat
│   ├── PAMI/pami_emotic.zip
│   ├── CVPR/CVPR17_Annotations.mat
│   └── CVPR/cvpr_emotic.zip
├── outputs/        # notebook tự tạo
├── models/         # notebook tự tạo
└── .cache/huggingface/   # cache Qwen/BERT/ViT, xem mục 5
```

### Cách 1: upload từ Colab (không phải tải dữ liệu về máy)

Lấy token trong `~/.modal.toml` trên máy local (hoặc tạo token mới ở modal.com → Settings → API Tokens). Sau đó chạy trong một notebook Colab đã mount Drive:

```python
!pip install -q modal
!modal token set --token-id <ID> --token-secret <SECRET>

SRC = "/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition"
!modal volume put emotic "{SRC}/data/raw/raw/emotic/PAMI/Annotations" /data/raw/raw/emotic/PAMI/Annotations
!modal volume put emotic "{SRC}/data/raw/raw/emotic/PAMI/pami_emotic.zip" /data/raw/raw/emotic/PAMI/pami_emotic.zip
!modal volume put emotic "{SRC}/data/raw/raw/emotic/CVPR/CVPR17_Annotations.mat" /data/raw/raw/emotic/CVPR/CVPR17_Annotations.mat
!modal volume put emotic "{SRC}/data/raw/raw/emotic/CVPR/cvpr_emotic.zip" /data/raw/raw/emotic/CVPR/cvpr_emotic.zip
```

Không dán token vào notebook rồi lưu lên Drive hay Git. Chạy xong thì xóa cell đó đi, hoặc dùng Colab Secrets.

### Cách 2: upload từ máy local

Tải 4 file trên từ Drive về, rồi chạy cùng các lệnh `modal volume put` với đường dẫn local.

### Kết quả cũ (chỉ khi cần)

Notebook 03 cần checkpoint của 02. Notebook 05 cần `outputs/qwen3_multimodal/<SOURCE_RUN_ID>` và export của 02. Muốn chạy chúng trên Modal với kết quả đã có trên Drive thì upload thêm các thư mục `outputs/...` và `models/...` tương ứng, cũng theo đúng đường dẫn tương đối như trên.

Kiểm tra:

```bash
modal volume ls emotic /data/raw/raw/emotic/PAMI
```

## 3. Cách A: JupyterLab có GPU (chạy tương tác)

```bash
modal launch jupyter --gpu L40S --volume emotic --timeout 21600
```

- Lệnh in ra URL JupyterLab. Volume được gắn ở `/mnt/emotic`.
- `--timeout` tính bằng giây; ở đây là 6 giờ. Hết timeout thì container bị tắt.
- GPU tính tiền theo giây chạy, nên **tắt container (Ctrl+C ở terminal local) khi xong việc**.

Mở **Terminal** trong JupyterLab và chạy một lần cho mỗi container:

```bash
set -e
apt-get update -qq && apt-get install -y -qq git unzip

# Đường dẫn Drive cứng trong notebook -> Volume
mkdir -p "/content/drive/MyDrive/Colab Notebooks"
ln -sfn /mnt/emotic "/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition"

# google.colab giả: drive.mount() không làm gì
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/google/colab"
touch "$SITE/google/colab/__init__.py"
cat > "$SITE/google/colab/drive.py" <<'EOF'
def mount(*args, **kwargs):
    print("Modal: bỏ qua drive.mount, dữ liệu nằm trên Volume /mnt/emotic")
EOF

# Thư viện của repo
git clone https://github.com/minh-1105/Explainable-Contextual-Emotion-Recognition-via-Text-Guided-Chain-of-Thought-Reasoning.git /content/contextual-emotion-reasoning
pip install -q -r /content/contextual-emotion-reasoning/requirements.txt
```

Sau đó mở notebook cần chạy. Có thể upload file `.ipynb` từ máy, hoặc mở trong `/content/contextual-emotion-reasoning/notebooks/`. Trước khi chạy cell đầu, thêm một cell để model Hugging Face được cache trên Volume:

```python
import os
os.environ["HF_HOME"] = "/mnt/emotic/.cache/huggingface"
```

Rồi chạy notebook như trên Colab.

## 4. Cách B: chạy nền bằng `modal run` (cho run dài: 02, 04)

Với cách này, notebook vẫn chạy khi bạn tắt máy. Tạo file `modal_app.py` ở gốc repo:

```python
import modal

DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition"
REPO_URL = "https://github.com/minh-1105/Explainable-Contextual-Emotion-Recognition-via-Text-Guided-Chain-of-Thought-Reasoning.git"

app = modal.App("emotic")
volume = modal.Volume.from_name("emotic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("papermill", "ipykernel")
    .run_commands(
        'mkdir -p "/content/drive/MyDrive/Colab Notebooks"',
        f'ln -sfn /mnt/emotic "{DRIVE_PATH}"',
        'SITE=$(python -c "import site; print(site.getsitepackages()[0])") '
        '&& mkdir -p "$SITE/google/colab" && touch "$SITE/google/colab/__init__.py" '
        "&& printf 'def mount(*args, **kwargs):\\n    pass\\n' > \"$SITE/google/colab/drive.py\"",
    )
    .env({"HF_HOME": "/mnt/emotic/.cache/huggingface"})
)


@app.function(image=image, gpu="L40S", volumes={"/mnt/emotic": volume}, timeout=24 * 3600)
def run_notebook(name: str):
    import subprocess
    from datetime import datetime
    from pathlib import Path

    import papermill as pm

    repo = Path("/root/notebook-source")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    out_dir = Path("/mnt/emotic/outputs/executed_notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        pm.execute_notebook(
            str(repo / "notebooks" / name),
            str(out_dir / f"{stamp}_{name}"),
            cwd="/content",
            log_output=True,
        )
    finally:
        volume.commit()


@app.local_entrypoint()
def main(name: str):
    run_notebook.remote(name)
```

Chạy:

```bash
modal run --detach modal_app.py --name 02_Train_Visual_Baseline.ipynb
modal run --detach modal_app.py --name 04_VLM_Pilot.ipynb
```

- `--detach`: job vẫn chạy khi đóng terminal. Xem log ở modal.com → Apps.
- Notebook được lấy từ nhánh `main` trên GitHub, không lấy từ máy local. Nên chỉnh cấu hình trong notebook (ví dụ `RUN_ID`, `MODEL_CONFIG`) rồi push lên `main` trước khi chạy.
- Notebook đã chạy (kèm output từng cell) được lưu ở `outputs/executed_notebooks/` trên Volume.
- Đổi GPU ở `gpu="..."` tùy notebook (xem mục 5).

## 5. Chọn GPU

| Notebook | GPU gợi ý | Ghi chú |
|---|---|---|
| 01 | không cần (`modal launch jupyter` bỏ `--gpu`) | Chỉ đọc annotation và vẽ ảnh. |
| 02, 03 | `T4`, `L4` | ResNet; `L4` nhanh hơn rõ rệt. |
| 04 | `L40S` hoặc `A100-80GB`; `H100` nếu muốn nhanh | Qwen3-VL-4B + BERT + ViT. Run full trước đây chạy trên H100. |
| 05 | `H100` (cấu hình trong `cloud/modal_nb05.py`) | Chỉ chạy inference ResNet, nên `L4` cũng đủ nếu muốn đổi. |

`HF_HOME` trỏ vào Volume nên Qwen, BERT và ViT chỉ tải một lần, các container sau dùng lại.

## 6. Lấy kết quả về

```bash
modal volume ls emotic /outputs
modal volume get emotic /outputs/qwen3_multimodal/<run_id> ./outputs/qwen3_multimodal/<run_id>
```

Muốn đưa kết quả về Drive thì làm ngược lại mục 2: chạy `modal volume get` trong một notebook Colab đã mount Drive.

`outputs/` và `models/` không được commit vào Git (xem `.gitignore`).

## 7. Lưu ý

- **Ghi dữ liệu vào Volume.** File ghi vào `/mnt/emotic` được lưu khi gọi `volume.commit()` (cách B đã gọi trong `finally`), và Modal cũng tự commit định kỳ cũng như khi container tắt bình thường. Với cách A, đừng để container bị kill giữa lúc đang ghi checkpoint.
- **Không giải nén ảnh vào Volume.** Notebook giải nén vào `/content/emotic` trên ổ tạm của container, đúng như trên Colab. Đọc hàng chục nghìn ảnh nhỏ từ Volume sẽ chậm. Mỗi container mới phải giải nén lại, mất vài phút.
- **Resume Notebook 04.** Notebook 04 so `MODEL_CONFIG` (gồm `git_commit`) và prompt với run cũ. Một run Colab đang dở chỉ resume được trên Modal nếu đã upload thư mục run đó lên Volume và `main` chưa có commit mới. An toàn nhất là bắt đầu run mới (`RUN_ID = None`).
- **Không push `main` khi đang có run.** Quy tắc này giữ nguyên, vì container clone `main` lúc bắt đầu.
- **Test set.** Quy tắc chọn model, ngưỡng và `k` theo validation không đổi khi chuyển sang Modal.
