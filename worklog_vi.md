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

### Hướng phát triển mô hình

- Bổ sung model giải thích
- Tìm model phù hợp: Qwen3VL
- Bổ sung Graph Transformer
- ViT + Multimodal Transformer

### Notebook đánh giá visual baseline

- Thêm `03_Evaluate_Visual_Baseline.ipynb`, nạp các bản export `model.pt` của `person_only`, `context_only` và `person_context` từ `models/` hoặc thư mục `checkpoints/` hiện có trên Drive.
- Bổ sung chọn ngưỡng từng nhãn chỉ trên validation, phân tích độ nhạy union/majority, mAP/AP/F1/precision/recall trên test, baseline hằng số top-k chọn trên validation, đánh giá khi bỏ `Engagement`, và F1 theo số người được chú thích trong ảnh.
- Bổ sung so sánh paired bootstrap theo ảnh giữa A/B/C, che vùng cho model `person_context`, và target swap trên ảnh nhiều người.
- Thêm các hàm metric đa nhãn và bootstrap dùng lại được trong `src/evaluation/`; Notebook 03 ghi cấu hình, môi trường, dự đoán, chỉ số và hình vào `outputs/resnet_evaluation/<run_id>/`.
- Chạy xong Notebook 03 trên Colab Tesla T4. Run `20261101-034030` lưu trên Drive tại `outputs/resnet_evaluation/`, gồm 27 artifact, source commit `0e4b81fc`.

## 2026-11-04

### Prototype đa phương thức có khả năng giải thích

- Chuyển `04_VLM_Pilot.ipynb` thành prototype kết hợp Qwen3-VL 4B, BERT, ViT, Graph Transformer và Multimodal Transformer trên 300 người thuộc validation PAMI.
- Bổ sung chia train/dev/eval theo ảnh, trích cue bằng Qwen không cho thấy danh sách nhãn cảm xúc, cache feature BERT/ViT đóng băng, chọn checkpoint và threshold chỉ trên dev, rồi báo cáo trên phần validation giữ lại.
- Bổ sung evidence package từ modality ablation và bước Qwen3-VL riêng để diễn đạt explanation chỉ từ evidence do emotion model tạo ra.
- Thêm `transformers>=4.57,<6` vào dependency Colab và cập nhật tài liệu workflow. Notebook 04 mới đã được kiểm tra cú pháp ở local nhưng vẫn cần chạy đầy đủ trên Colab GPU.
- Đồng bộ cell clone/cài đặt dùng chung ở Notebook 02 và 04, đồng thời xóa dấu gạch ngang thừa trong URL GitHub của Notebook 02.
- Mở rộng Notebook 04 từ pilot 300 người sang huấn luyện tối đa: toàn bộ 23.706 người của PAMI train; toàn bộ 3.334 người của PAMI val được chia theo ảnh thành dev/eval. Notebook vẫn không dùng test và có thể resume bước sinh Qwen cue qua nhiều phiên Colab.
- Bổ sung giới hạn cue cân bằng cho Notebook 04: 3 cue từ người, 3 cue ngữ cảnh, 2 quan hệ và 1 tình huống; giảm Qwen còn 192 token và mỗi node BERT còn 32 token, đồng thời áp dụng cùng giới hạn trong prompt và bước tạo semantic node.

## 2026-11-07

### Kế hoạch Notebook 05 — so sánh công bằng baseline và mô hình đa phương thức

- Đã triển khai `05_Compare_Multimodal_Baselines.ipynb`, thêm kiến trúc checkpoint-compatible trong `src/models/visual_baseline.py`, kiểm tra cú pháp local; chờ chạy Colab GPU. Full gốc và full ba seed được ghi riêng, bootstrap full-vs-C tính mới còn full-vs-ViT-only kế thừa từ run 04. Xuất thêm `people_group_metrics.json`. Nguồn metadata được gắn cờ provisional vì chưa có hash artifact tại thời điểm tạo model.

- Thêm `05_Compare_Multimodal_Baselines.ipynb` làm notebook đánh giá trên Colab; notebook không được huấn luyện hoặc tinh chỉnh theo tập test PAMI.
- Dùng đúng các `sample_id` của hai phần `dev` và `eval` đã chia theo ảnh và lưu bởi run Notebook 04 hoàn tất, để ba ResNet baseline và các biến thể đa phương thức được đánh giá trên cùng người.
- Dùng `20261104-071259-h100` làm run Notebook 04 chính thức. Giữ `config.json` ở commit tạo config `cbdb25aa`, chuẩn hóa `environment.txt`, sao lưu bản cũ thành `environment.original.txt` và thêm `source_artifacts.json` để ghi riêng các stage T4/H100 cùng commit `d9c147fa`, `93ecf0c5` và `211b6b25`.
- Nạp các bản export ResNet `person_only`, `context_only` và `person_context` đã chốt, không huấn luyện lại. Chọn toàn bộ threshold theo lớp và constant top-k chỉ trên `dev`, sau đó áp dụng lựa chọn đã đóng băng đúng một lần trên `eval`.
- Xuất một bảng chung cho A/B/C, ViT-only, mô hình đa phương thức full và baseline hằng số, gồm mAP, macro/micro/sample F1, precision, recall, AP/F1 theo lớp và số nhãn dự đoán trung bình.
- So sánh mô hình full với `person_context` và ViT-only bằng paired bootstrap theo ảnh, dùng hàm chung trong `src/evaluation/`; lưu chênh lệch điểm, khoảng tin cậy 95%, số ảnh, số lần lấy mẫu lại và seed.
- Tái sử dụng prediction, ba seed và ablation ViT/full/no-person/no-context đã lưu bởi Notebook 04. Notebook 05 không gọi Qwen, không train model, không tính lại feature và không lặp cue/explanation/attribution checks.
- Chỉ bổ sung phân tích so sánh A/B/C với ViT-only/full: mức tăng/giảm theo nhãn, kết quả theo số người trong ảnh và ví dụ lỗi chọn theo quy tắc cố định.
- Ghi output vào `outputs/multimodal_comparison/<run_id>/`, gồm `config.json`, `environment.txt` chỉ có một `git_commit` rõ ràng, `source_artifacts.json`, `metrics.csv`, `metrics.json`, `bootstrap.json`, kết quả theo lớp, prediction nén và các hình so sánh.
- Sau khi Notebook 05 hoàn tất so sánh công bằng và team chấp nhận cấu hình full đã khóa từ Notebook 04, mới đánh giá PAMI test đúng một lần. Không dùng kết quả test để chỉnh model, threshold, seed hoặc biến thể.

## 2026-11-09 — Triển khai Notebook 05 trên Modal
- Thêm cloud/modal_nb05.py: snapshot local, preflight CPU, GPU L4, Volume emotic; lưu notebook đã chạy, trạng thái và hash code; không push main.
- Thêm uploader chỉ tải 16 artifact cần thiết từ Drive (395.675.697 byte), không ghi đè dữ liệu nguồn.
- Đã deploy app emotic-nb05, preflight ready=true và khởi chạy nền fc-01M6ZPQBJ0GE3D1S75AVAECASB. Chưa xác nhận hoàn tất đánh giá.
- Kiểm tra cú pháp tất cả notebook và runner: ok; kiểm tra preflight thiếu file và adapter notebook: ok.


### Notebook 05: lưới threshold mở rộng và bootstrap trên mAP

- Run đầu `20261109-083552-102172` cho thấy full hơn ResNet C +0,137 Macro-F1. Mức chênh này chủ yếu do lưới threshold:
  - threshold chọn trên dev của ResNet kẹt ở đáy lưới 0,05 (A: 26/26 lớp, C: 20/26 lớp);
  - 33–43% mẫu dương trên dev của A và C có xác suất dưới 1e-4.
- Thêm vào `src/evaluation/`:
  - `paired_image_bootstrap(..., metric="map")`; mặc định vẫn là macro-F1;
  - `mean_average_precision`: khớp sklearn, kể cả khi có điểm bằng nhau;
  - `threshold_boundary_report`: đếm số lớp có threshold chạm biên lưới.
- Notebook 05 dùng lưới threshold 1e-8 → 0,95; vẫn tính lưới cũ song song để đối chiếu. Thêm `per_class_ap_delta.csv` và mAP theo số người trong ảnh. Run `20261109-090105-378575` (H100, snapshot `272091817d75…`):
  - mAP: full 0,405, C 0,406. Full − C: mAP −0,001 [−0,014; 0,014], Macro-F1 +0,017 [0,001; 0,033].
  - Full − B: mAP +0,018 [0,002; 0,034].
  - Full hơn C ở 13/26 lớp; chênh lệch AP không tương quan với số mẫu của lớp.
  - Kết luận: full ngang ResNet C. Full chỉ hơn rõ ViT-only (+0,036 Macro-F1).
  - Hạn chế còn lại: threshold của full vẫn theo lưới cũ (11/26 lớp ở đáy) vì run 04 không lưu xác suất dev; checkpoint ResNet chọn trên toàn bộ PAMI val; ResNet chỉ có một seed.

### Phân tích nút thắt

- **ResNet (Notebook 02) overfit:** 30 epoch, không augmentation, `BCEWithLogitsLoss` không `pos_weight`. Val BCE ở checkpoint tốt nhất là 1,65–2,06. Train thêm epoch không giúp.
- **Phần fusion ở Notebook 04 overfit sớm:** dev mAP đạt đỉnh ở epoch 3–7 rồi giảm 0,02–0,03 đến epoch 20, trong khi train loss vẫn giảm (0,63 → 0,24). Giới hạn nằm ở feature ảnh đóng băng (ViT-only khoảng 0,38 mAP).
- **Cue Qwen** (27.040 người):
  - 99,5% có đúng 3 person cue; 38% person cue là quần áo, chỉ 14% về mặt hay biểu cảm; 56% số người không có cue nào về mặt, ánh nhìn hay đầu;
  - 18,4% số người có từ chỉ cảm xúc, chủ yếu "smiling";
  - 32% relation dùng ID "person_N" mà Qwen không được cho biết.
- **Ablation:** `no_person` và `no_context` chỉ bỏ token ảnh; chưa có ablation riêng cho person cue và context cue.
- **CVPR17 là tập con của PAMI:** mọi ảnh CVPR đều nằm trong cùng split của PAMI (12.915/1.705/3.682 ảnh). Ảnh CVPR vẫn được dùng làm image root; annotation CVPR không thêm dữ liệu mới.

### Đồng bộ repo

- Pull `main` lên `d9229b20`. Chuyển local sang nhánh `nb04-batch-cues` (`e304c0ca`). Thay đổi tài liệu local được cất vào stash.
- Áp lại `src/evaluation/` và `.gitignore` (thêm `!src/models/`, vì dòng `models/` vô tình bỏ qua cả `src/models/`).

### Đề xuất: tái cấu trúc Notebook 04, chưa thực hiện

- Viết lại notebook không tự làm điểm tăng. Nút thắt là feature và cue; lần chạy lại `20261107-133419` cho ra đúng số cũ.
- Đề xuất:
  - giữ 04 nguyên làm bản ghi chính thức của run H100;
  - tách thành các phần: sinh cue (Qwen, dùng lại `qwen_cues.jsonl`), feature và train (backbone cấu hình được, ma trận biến thể × seed, lưu xác suất dev và eval từng người), explanation, và test chỉ chạy một lần.
- Thứ tự làm:
  1. Tái lập kết quả H100 bằng code mới.
  2. Đổi backbone sang CLIP hoặc SigLIP, thêm feature khuôn mặt.
  3. Chỉ đổi prompt Qwen khi ablation cho thấy cue có đóng góp.
  