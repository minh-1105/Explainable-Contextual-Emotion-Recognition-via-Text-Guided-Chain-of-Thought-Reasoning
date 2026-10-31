# Nhật ký công việc

## 2026-10-14

### Khởi tạo và tổ chức dự án

- Khởi tạo kho mã nguồn cho dự án **Nhận diện cảm xúc theo ngữ cảnh có khả năng giải thích bằng suy luận chuỗi tư duy được định hướng bởi văn bản** (*Explainable Contextual Emotion Recognition via Text-Guided Chain-of-Thought Reasoning*).
- Bổ sung vào `requirements.txt` các thư viện cần thiết cho xử lý ảnh, đọc tệp chú thích MATLAB, PyTorch, trực quan hóa, cấu hình và đánh giá.
- Cấu hình `.gitignore` cho kho mã nguồn.

### Chuẩn bị dữ liệu EMOTIC

- Chuẩn bị cả hai phiên bản của bộ dữ liệu EMOTIC: **PAMI** và **CVPR**.
- Bổ sung tệp `Annotations.mat` của PAMI và tệp `CVPR17_Annotations.mat` của CVPR.
- Giải nén và tổ chức các nguồn ảnh EMOTIC, bao gồm `framesdb`, `mscoco`, `ade20k` và `emodb_small`.

### Bộ nạp dữ liệu

- Triển khai `src/datasets/emotic_dataset.py` dưới dạng một lớp `Dataset` của PyTorch.
- Bổ sung hỗ trợ cho các tập `train`, `val` và `test` trong chú thích EMOTIC.
- Hỗ trợ một hoặc nhiều thư mục gốc chứa ảnh để có thể tìm ảnh trong các nguồn khác nhau.
- Chuyển mỗi người được chú thích thành một mẫu độc lập, kể cả đối với ảnh có nhiều người.
- Triển khai xử lý hộp giới hạn, giới hạn tọa độ theo kích thước ảnh và tùy chọn thêm khoảng đệm xung quanh vùng cắt của người mục tiêu.
- Trả về cả ảnh ngữ cảnh đầy đủ và vùng ảnh cắt của người mục tiêu trong mỗi mẫu.
- Ánh xạ tập cố định gồm 26 nhãn cảm xúc EMOTIC thành các tensor multi-hot.
- Bổ sung hai chính sách `union` và `majority` để tổng hợp các chú thích phân loại.
- Tổng hợp các chú thích VAD và tạo mặt nạ cho những mẫu không có giá trị VAD.
- Bổ sung cơ chế kiểm tra nghiêm ngặt đối với ảnh bị thiếu và tùy chọn bỏ qua các mẫu không có ảnh khi kiểm tra dữ liệu.

### Notebook kiểm tra dữ liệu

#### Mục đích của `01_Inspect.ipynb`

- Tạo `01_Inspect.ipynb` làm bước kiểm định dữ liệu bắt buộc phải hoàn tất trước khi phát triển hoặc huấn luyện mô hình.
- Sử dụng notebook để xác minh cấu trúc thực tế của chú thích EMOTIC gốc thay vì giả định một lược đồ đã được tiền xử lý.
- Xác minh rằng các tệp chú thích PAMI và CVPR có đầy đủ các tập `train`, `val` và `test` như dự kiến.
- Đếm số bản ghi ảnh và số chú thích người trong từng tập để xác nhận rằng một ảnh có thể chứa nhiều người mục tiêu được chú thích.
- Kiểm tra 100 bản ghi chú thích đầu tiên của mỗi tập có ánh xạ được tới một ảnh hiện có trong các thư mục ảnh đã cấu hình.
- Kiểm tra hộp giới hạn của người mục tiêu, chú thích cảm xúc phân loại và chú thích VAD liên tục.
- Trực quan hóa ảnh nguồn cùng toàn bộ hộp giới hạn của người mục tiêu để đối chiếu tọa độ chú thích với ảnh thực tế.
- Giữ notebook tập trung vào việc kiểm tra và xác thực; notebook này không huấn luyện mô hình hoặc tạo dữ liệu huấn luyện đã qua xử lý.

#### Dữ liệu và công cụ được sử dụng

- Đọc tệp `Annotations.mat` của PAMI và tệp `CVPR17_Annotations.mat` của CVPR.
- Tìm ảnh trong các nguồn `framesdb`, `mscoco`, `ade20k` và `emodb_small`.
- Sử dụng `scipy.io.loadmat(..., simplify_cells=True)` để chuyển đổi cấu trúc chú thích MATLAB thành từ điển, danh sách và giá trị NumPy trong Python.
- Sử dụng `numpy` để chuẩn hóa các mảng MATLAB và phân tích tọa độ hộp giới hạn.
- Sử dụng `pathlib.Path` để tạo đường dẫn ảnh tương đối và tìm kiếm trong các thư mục gốc chứa ảnh đã cấu hình.
- Sử dụng `PIL.Image` để mở ảnh nguồn ở định dạng RGB.
- Sử dụng `matplotlib` và `matplotlib.patches.Rectangle` để hiển thị ảnh và vẽ hộp giới hạn của người mục tiêu.
- Bổ sung các hàm hỗ trợ để chuẩn hóa giá trị MATLAB, đọc các tập dữ liệu hiện có, đếm số người, tạo đường dẫn ảnh tương đối và tìm tệp ảnh.
- Loại bỏ sự phụ thuộc vào việc nhập `EmoticDataset`, nhờ đó notebook có thể kiểm tra trực tiếp các tệp EMOTIC thô.
- Xóa kết quả lỗi nhập mô-đun trước đó và kiểm tra cú pháp của tất cả các ô mã.

### Quy trình thực nghiệm

- Chuyển `workflow.md` sang quy trình **ưu tiên notebook**, trong đó notebook là điểm truy cập chính để huấn luyện và đánh giá.
- Xác định trình tự thực nghiệm: kiểm tra dữ liệu, ba mô hình cơ sở dùng đặc trưng thị giác, VLM trực tiếp, VLM suy luận có cấu trúc, phân tích kết quả và tinh chỉnh LoRA.
- Xác định ba mô hình cơ sở dùng đặc trưng thị giác: `person_only`, `context_only` và `person_context`.
- Chọn `BCEWithLogitsLoss` cho bài toán phân loại đa nhãn gồm 26 lớp.
- Chọn mAP trên tập validation làm chỉ số để lựa chọn mô hình tốt nhất.
- Thiết kế quy ước đầu ra cho mỗi lần chạy, bao gồm cấu hình, nhật ký, chỉ số, dự đoán, hình minh họa và checkpoint.
- Quy định `last.pt` là checkpoint dùng để tiếp tục huấn luyện và `best.pt` là checkpoint tốt nhất trên tập validation.
- Bổ sung hướng dẫn tiếp tục huấn luyện và xuất mô hình hoàn chỉnh để suy luận.

## 2026-10-15

### Notebook visual baseline

- Thêm `02_Train_Visual_Baseline.ipynb`: ba mô hình cơ sở ResNet-50 `person_only`, `context_only` và `person_context`, huấn luyện bằng `BCEWithLogitsLoss`, AdamW, lịch cosine và AMP.
- Mỗi lần chạy ghi cấu hình, môi trường, lịch sử theo epoch, chỉ số và dự đoán validation mới nhất, `last.pt` và `best.pt` (chọn theo mAP validation) vào `outputs/runs/<experiment_name>/<run_id>/`.
- Bổ sung tiếp tục huấn luyện từ `last.pt` có kiểm tra tương thích cấu hình, và xuất mô hình tốt nhất vào `models/`.

## 2026-10-19

### Tăng tốc đọc ảnh trên Colab

- Lưu ảnh EMOTIC trên Drive dưới dạng `pami_emotic.zip` và `cvpr_emotic.zip`; notebook giải nén vào `/content/emotic` một lần mỗi runtime và kiểm số file.

### Pilot VLM

- Thêm `04_VLM_Pilot.ipynb`, so sánh hỏi thẳng (D) với suy luận có cấu trúc (E) trên 300 người thuộc tập validation PAMI, dùng `Qwen/Qwen2.5-VL-3B-Instruct`.
- Người mục tiêu được đánh dấu bằng khung đỏ; hai cấu hình dùng chung model, ảnh, danh sách nhãn, giới hạn token và giải mã greedy.
- Câu trả lời được ghi theo từng mẫu vào JSONL để chạy tiếp khi bị ngắt; chỉ số gồm F1 macro/micro/theo mẫu, tỉ lệ parse, đúng schema, bị cắt cụt, và khoảng tin cậy bootstrap ghép cặp theo ảnh cho chênh lệch macro F1 E − D.
- Ví dụ minh họa được chọn theo quy tắc cố định: câu trả lời E đúng schema có F1 theo mẫu cao nhất và thấp nhất, hòa thì lấy câu trả lời ngắn hơn.

## 2026-10-31

- Cập nhật `README.md`, `SETUP.md`, `workflow.md` và `drivesetup.md` cho khớp với notebook và layout Drive đang dùng.
