# Hướng dẫn cho agent

Đọc `README.md`, rồi `workflow.md` (layout Drive, output của từng notebook, thí nghiệm đã có và còn thiếu). Code là nguồn sự thật: doc lệch code thì sửa doc ngay trong cùng thay đổi.

## Môi trường chạy

- Notebook chạy trên Colab (02, 03 và 04 cần GPU). Đường dẫn cứng là `/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition` (Drive) và `/content` (runtime); máy local không có dữ liệu ở đó. Người trong team chạy trên Colab và gửi lại output hoặc file kết quả.
- 04 còn chạy được trên Modal bằng `cloud/modal_nb04.py` (xem `SETUP.md`): script clone repo, thay hai cell mount Drive bằng cell trỏ vào volume `ecer-nb04` (cùng layout với Drive), chạy notebook bằng papermill và lưu bản đã chạy; `cloud/attach_outputs.py` chép output vào notebook trong repo. Agent có Modal token thì tự chạy được. Đổi thứ tự hoặc nội dung ba code cell đầu của 04 thì sửa script theo.
- Cell đầu của 02, 03 và 04 clone hoặc `git pull --ff-only` nhánh `main` của repo này vào `/content/contextual-emotion-reasoning`, và ghi `git_commit` vào `environment.txt` của run. Thay đổi trong `src/` chưa lên `main` thì Colab không thấy.
- Kiểm tra được ở local, chạy sau mỗi lần sửa:

```bash
python -c "import ast, glob, json; [ast.parse(''.join(c['source'])) for p in glob.glob('notebooks/*.ipynb') for c in json.load(open(p, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']; print('ok')"
```

## Cấu trúc

- `notebooks/`: điểm chạy. `notebooks/runs/` chứa bản đã chạy trên Colab của các run không nằm trong notebook chính (giữ nguyên như lúc chạy, không sửa); `02` mang output của run `person_only` `20261019-091248`. Hàm `extract_images` và các đường dẫn EMOTIC lặp lại ở 01, 02, 03 và 04; cell clone repo lặp lại ở 02, 03 và 04. Sửa ở một notebook thì sửa ở tất cả các notebook còn lại.
- `src/datasets/emotic_dataset.py`: code dùng chung, 02, 03 và 04 import.
- `src/evaluation/`: metric đa nhãn, chọn ngưỡng, baseline hằng số top-k và paired bootstrap; 03 import.

## Không được phá

Kết quả đã lưu trên Drive phụ thuộc vào những thứ sau; đổi chúng là làm hỏng resume hoặc so sánh với run cũ, nên phải báo team trước:

- Thứ tự 26 nhãn trong `EMOTION_LABELS`: `label_map.json`, `class_names` trong checkpoint, vector multi-hot và metric của 04 đều theo thứ tự này.
- `sample_id` dạng `<split>:<image_index>:<person_index>`: 04 dùng nó để resume cue Qwen và để mọi biến thể chạy đúng cùng một tập người.
- Khóa của checkpoint và các trường mà 02 so khi resume (`experiment_name`, `backbone`, `image_size`, `category_policy`, `crop_padding`).
- 04 từ chối resume khi `MODEL_CONFIG` (trừ `git_commit`) hoặc `prompts.json` khác run cũ; commit mới thì ghi thêm toàn bộ môi trường hiện tại vào `environment.txt`. Đổi prompt hoặc cấu hình nghĩa là run mới (`RUN_ID = None`).
- Chọn model, ngưỡng và `k` theo validation. Hiện chỉ `03_Evaluate_Visual_Baseline.ipynb` dùng test set, và đã chạy xong cho ResNet; không chỉnh cấu hình theo số test rồi chạy lại.

## Quy ước

- Không commit `data/`, `outputs/`, `models/`, weights hay file zip (xem `.gitignore`).
- Markdown và thông báo trong notebook viết tiếng Việt; tên biến, hàm, docstring trong `src/` viết tiếng Anh.
- Notebook mới ghi kết quả vào `outputs/<tên>/<run_id>/` (hoặc `outputs/runs/<experiment_name>/<run_id>/` với notebook train), kèm `config.json` và `environment.txt` có `git_commit`, như 02 và 04.
- Notebook import thư viện mà Colab không có sẵn thì thêm vào `requirements.txt`.
- Commit message tiếng Anh, một câu mô tả thay đổi hành vi, như các commit trong `git log`.
