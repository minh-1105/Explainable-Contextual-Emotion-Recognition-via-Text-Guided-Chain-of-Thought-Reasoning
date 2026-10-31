# Hướng dẫn cho agent

Đọc `README.md`, rồi `workflow.md` (layout Drive, output của từng notebook, thí nghiệm đã có và còn thiếu). Code là nguồn sự thật: doc lệch code thì sửa doc ngay trong cùng thay đổi.

## Môi trường chạy

- Notebook chỉ chạy trên Colab (02 và 04 cần GPU). Đường dẫn cứng là `/content/drive/MyDrive/Colab Notebooks/Contextual Emotion Recognition` (Drive) và `/content` (runtime); máy local không có dữ liệu ở đó, nên agent không chạy được notebook. Người trong team chạy trên Colab và gửi lại output hoặc file kết quả.
- Cell đầu của 02 và 04 clone hoặc `git pull --ff-only` nhánh `main` của repo này vào `/content/contextual-emotion-reasoning`, và ghi `git_commit` vào `environment.txt` của run. Thay đổi trong `src/` chưa lên `main` thì Colab không thấy.
- Kiểm tra được ở local, chạy sau mỗi lần sửa:

```bash
python -c "import ast, glob, json; [ast.parse(''.join(c['source'])) for p in glob.glob('notebooks/*.ipynb') for c in json.load(open(p, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']; print('ok')"
```

## Cấu trúc

- `notebooks/`: điểm chạy. Hàm `extract_images` và các đường dẫn EMOTIC lặp lại ở 01, 02 và 04; cell clone repo lặp lại ở 02 và 04. Sửa ở một notebook thì sửa ở tất cả các notebook còn lại.
- `src/datasets/emotic_dataset.py`: code dùng chung, 02 và 04 import.

## Không được phá

Kết quả đã lưu trên Drive phụ thuộc vào những thứ sau; đổi chúng là làm hỏng resume hoặc so sánh với run cũ, nên phải báo team trước:

- Thứ tự 26 nhãn trong `EMOTION_LABELS`: `label_map.json`, `class_names` trong checkpoint, vector multi-hot và nhãn trong JSONL của pilot đều theo thứ tự này.
- `sample_id` dạng `<split>:<image_index>:<person_index>`: pilot dùng nó để resume và để D, E chạy đúng cùng một tập người.
- Khóa của checkpoint và các trường mà 02 so khi resume (`experiment_name`, `backbone`, `image_size`, `category_policy`, `crop_padding`).
- Pilot 04 từ chối resume khi `PILOT_CONFIG` hoặc `PROMPTS` khác run cũ. Đổi prompt hoặc cấu hình nghĩa là run mới (`RUN_ID = None`).
- Chọn model theo validation. Chưa notebook nào dùng test set; test chỉ dùng trong `03_Evaluate_Visual_Baseline.ipynb` (chưa có).

## Quy ước

- Không commit `data/`, `outputs/`, `models/`, weights hay file zip (xem `.gitignore`).
- Markdown và thông báo trong notebook viết tiếng Việt; tên biến, hàm, docstring trong `src/` viết tiếng Anh.
- Notebook mới ghi kết quả vào `outputs/<tên>/<run_id>/` (hoặc `outputs/runs/<experiment_name>/<run_id>/` với notebook train), kèm `config.json` và `environment.txt` có `git_commit`, như 02 và 04.
- Notebook import thư viện mà Colab không có sẵn thì thêm vào `requirements.txt`. Hiện 04 dùng `transformers` có sẵn trên Colab mà chưa khai báo trong file này.
- Commit message tiếng Anh, một câu mô tả thay đổi hành vi, như các commit trong `git log`.
