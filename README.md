# Explainable Contextual Emotion Recognition

Nhận diện cảm xúc theo ngữ cảnh trên EMOTIC: ba visual baseline ResNet-50 và prototype giải thích kết hợp Qwen3-VL, BERT, ViT, Graph Transformer và Multimodal Transformer.

## Bắt đầu

1. Chuẩn bị dữ liệu trên Drive và chạy notebook theo `SETUP.md`.
2. Đọc `workflow.md` để biết từng notebook làm gì, ghi kết quả ở đâu và thí nghiệm nào còn thiếu.
3. Dùng agent (Claude Code, Codex, …) thì agent đọc `AGENTS.md`; `CLAUDE.md` chỉ trỏ tới file đó.

## Nội dung repo

- `notebooks/`: điểm chạy trên Colab (`01_Inspect`, `02_Train_Visual_Baseline`, `03_Evaluate_Visual_Baseline`, `04_Qwen3_Multimodal`). Notebook 04 trích cue bằng Qwen3-VL, mã hóa cue bằng BERT, hợp nhất với feature ViT và cho Qwen3-VL diễn đạt evidence của mô hình thành explanation. `notebooks/runs/` giữ bản đã chạy của các run baseline khác.
- `cloud/`: chạy notebook 04 trên Modal và chép output về notebook trong repo.
- `src/datasets/emotic_dataset.py`: `EmoticDataset`, mỗi người được chú thích là một mẫu; trả ảnh toàn cảnh, crop người, nhãn multi-hot 26 lớp và VAD.
- `src/evaluation/`: metric đa nhãn, chọn ngưỡng, baseline hằng số top-k và paired bootstrap.
- `results/`: file nhỏ của các run mà paper báo cáo (config, metric, training log, hình).
- `ECER-paper/`: paper IEEE (`main.tex`, `main_vi.tex`) và script kiểm số liệu của paper.
- `SETUP.md`: chuẩn bị Drive và chạy notebook.
- `workflow.md`: layout dữ liệu, output của từng notebook, các thí nghiệm đã có và còn lại.
- `AGENTS.md`: môi trường chạy, những thứ không được phá và quy ước khi sửa code.
