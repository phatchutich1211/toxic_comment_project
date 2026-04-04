# Toxic Comment Filtering for Large Fanpages

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![NLP](https://img.shields.io/badge/NLP-Logistic%20Regression%20%2B%20PhoBERT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Lọc **bình luận độc hại** và **quảng cáo rác** cho Fanpage lớn bằng **Logistic Regression**, **PhoBERT** và **rule-based spam detection**, kèm giao diện **Streamlit** để xem EDA, chạy dự đoán và đánh giá mô hình.

## 1. Mục tiêu đề tài

Đề tài hướng đến việc hỗ trợ kiểm duyệt bình luận trên Fanpage bằng cách:

- phát hiện bình luận **độc hại** như chửi bới, xúc phạm
- phát hiện **quảng cáo rác / spam**
- gợi ý **giữ bình luận** hoặc **ẩn tự động**
- giảm tải cho kiểm duyệt thủ công

## 2. Dữ liệu sử dụng

Dự án sử dụng bộ dữ liệu tiếng Việt **ViCTSD** cho bài toán phát hiện bình luận độc hại.

- GitHub: https://github.com/tarudesu/ViCTSD
- Hugging Face: https://huggingface.co/datasets/tarudesu/ViCTSD

Các file chính trong project:
- `data/ViCTSD_train.csv`
- `data/ViCTSD_valid.csv`
- `data/ViCTSD_test.csv`
## Phương pháp thực hiện

Đề tài được thực hiện theo các bước sau:

1. **Thu thập và chuẩn hóa dữ liệu**  
   Dữ liệu bình luận được lấy từ bộ dữ liệu ViCTSD. Sau đó dữ liệu được chuẩn hóa lại về các cột chính như nội dung bình luận, nhãn, tiêu đề và chủ đề để thuận tiện cho quá trình huấn luyện và đánh giá.

2. **Tiền xử lý văn bản**  
   Bình luận được xử lý trước khi đưa vào mô hình bằng cách:
   - chuyển về chữ thường
   - loại bỏ ký tự nhiễu
   - chuẩn hóa khoảng trắng
   - thay thế link, email và số điện thoại bằng các token đặc biệt
   - giảm bớt các trường hợp lặp ký tự bất thường

3. **Huấn luyện mô hình Logistic Regression**  
   Logistic Regression được sử dụng làm mô hình baseline nhờ ưu điểm nhẹ, tốc độ nhanh, dễ huấn luyện và dễ triển khai trong môi trường thực tế.

4. **Huấn luyện mô hình PhoBERT**  
   PhoBERT được fine-tune trên dữ liệu tiếng Việt để tăng khả năng nhận diện các bình luận độc hại có ngữ cảnh phức tạp hơn, đặc biệt là các câu xúc phạm gián tiếp hoặc biến thể ngôn ngữ mạng.

5. **Kết hợp luật phát hiện spam**  
   Ngoài mô hình phân loại toxic, hệ thống còn sử dụng rule-based spam detection để phát hiện các bình luận quảng cáo rác dựa trên các đặc trưng như:
   - link
   - số điện thoại
   - từ khóa quảng cáo
   - từ khóa tuyển cộng tác viên
   - từ khóa tăng follow, tăng tương tác

6. **Đánh giá mô hình**  
   Các mô hình được đánh giá bằng các chỉ số:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

7. **Triển khai ứng dụng web**  
   Sau khi huấn luyện và lưu model, hệ thống được triển khai bằng Streamlit để minh họa khả năng dự đoán trong thực tế. Ứng dụng hỗ trợ nhập bình luận đơn, tải file CSV để dự đoán hàng loạt và hiển thị kết quả đánh giá mô hình.
## 3. Phương pháp sử dụng

Hệ thống kết hợp 3 thành phần:

### Logistic Regression
- dùng làm mô hình baseline
- nhanh, nhẹ, dễ triển khai

### PhoBERT
- xử lý ngữ cảnh tiếng Việt tốt hơn
- phù hợp với các câu toxic phức tạp hơn

### Rule-based Spam Detection
- bắt các mẫu spam như:
  - link
  - số điện thoại
  - `zalo`, `telegram`, `inbox`, `ib`
  - `sale`, `khuyến mãi`, `tuyển ctv`, `tăng follow`

## 4. Chức năng của ứng dụng

Ứng dụng Streamlit gồm 3 trang:

### 1. Giới thiệu & EDA
- mô tả đề tài
- hiển thị dữ liệu thô
- biểu đồ phân phối nhãn
- biểu đồ độ dài bình luận
- biểu đồ phân bố theo chủ đề

### 2. Triển khai mô hình
- nhập một bình luận để dự đoán
- chọn mô hình Logistic Regression hoặc PhoBERT
- xem xác suất toxic
- dự đoán hàng loạt bằng file CSV

### 3. Đánh giá & Hiệu năng
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Classification Report
- phân tích lỗi dự đoán

## 5. Cấu trúc thư mục

```text
toxic_comment_project/
├── app.py
├── requirements.txt
├── README.md
├── toxic_comment_filter_colab_local_fixed.ipynb
├── data/
├── models/
├── reports/
└── .gitattributes
