# Explainable Contextual Emotion Recognition

Nhận diện cảm xúc theo ngữ cảnh trên EMOTIC: ba visual baseline ResNet-50 và pilot so sánh VLM hỏi thẳng (D) với VLM suy luận có cấu trúc (E).

## Bắt đầu

1. Chuẩn bị dữ liệu trên Drive và chạy notebook theo `SETUP.md`.
2. Đọc `workflow.md` để biết từng notebook làm gì, ghi kết quả ở đâu và thí nghiệm nào còn thiếu.
3. Dùng agent (Claude Code, Codex, …) thì agent đọc `AGENTS.md`; `CLAUDE.md` chỉ trỏ tới file đó.

## Nội dung repo

- `notebooks/`: điểm chạy trên Colab (`01_Inspect`, `02_Train_Visual_Baseline`, `03_Evaluate_Visual_Baseline`, `04_VLM_Pilot`).
- `src/datasets/emotic_dataset.py`: `EmoticDataset`, mỗi người được chú thích là một mẫu; trả ảnh toàn cảnh, crop người, nhãn multi-hot 26 lớp và VAD.
- `SETUP.md`: chuẩn bị Drive và chạy notebook.
- `workflow.md`: layout dữ liệu, output của từng notebook, các thí nghiệm đã có và còn lại.
- `AGENTS.md`: môi trường chạy, những thứ không được phá và quy ước khi sửa code.
- `drivesetup.md`: nối repo trên máy local với Drive.
- `worklog.md`, `worklog_vi.md`: nhật ký công việc.
