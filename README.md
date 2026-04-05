# Toxic Comment Filtering for Large Fanpages

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![NLP](https://img.shields.io/badge/NLP-Linear%20Regression-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Lọc **bình luận độc hại** và **quảng cáo rác** cho Fanpage lớn bằng **Linear Regression** kết hợp **rule-based spam detection**, kèm giao diện **Streamlit** để xem EDA, chạy dự đoán và đánh giá mô hình.

## 1. Mục tiêu đề tài

Đề tài hỗ trợ kiểm duyệt bình luận trên Fanpage bằng cách:

- phát hiện bình luận **độc hại** như chửi bới, xúc phạm
- phát hiện **quảng cáo rác / spam**
- gợi ý **giữ bình luận**, **chuyển kiểm duyệt thủ công** hoặc **ẩn tự động**
- giảm tải cho kiểm duyệt thủ công

## 2. Dữ liệu sử dụng

Dự án sử dụng bộ dữ liệu tiếng Việt **ViCTSD** cho bài toán phát hiện bình luận độc hại.

- GitHub: https://github.com/tarudesu/ViCTSD
- Hugging Face: https://huggingface.co/datasets/tarudesu/ViCTSD

Các file chính:

- `data/ViCTSD_train.csv`
- `data/ViCTSD_valid.csv`
- `data/ViCTSD_test.csv`

## 3. Phương pháp thực hiện

1. **Chuẩn hóa dữ liệu**
   Đổi tên cột, ép kiểu dữ liệu và bổ sung các trường phục vụ EDA.

2. **Tiền xử lý văn bản**
   Văn bản được chuyển về chữ thường, chuẩn hóa khoảng trắng, thay link, email, số điện thoại bằng token đặc biệt và giảm nhiễu ký tự lặp.

3. **Huấn luyện mô hình Linear Regression**
   Sử dụng pipeline `TfidfVectorizer + LinearRegression` để dự đoán điểm toxic trong khoảng liên tục, sau đó chặn giá trị về `[0, 1]` và dùng ngưỡng để suy ra nhãn.

4. **Kết hợp luật phát hiện spam**
   Rule-based spam detection dùng các mẫu như link, số điện thoại, từ khóa bán hàng, tuyển cộng tác viên, tăng follow hoặc tăng tương tác.

5. **Đánh giá mô hình**
   Mô hình được đánh giá bằng:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

6. **Triển khai ứng dụng web**
   Streamlit được dùng để hiển thị EDA, suy luận cho từng bình luận hoặc hàng loạt qua CSV, và trình bày kết quả đánh giá.

## 4. Thành phần hệ thống

### Linear Regression

- mô hình duy nhất cho phân loại toxic
- nhẹ, dễ train và dễ trình bày
- đầu ra là điểm liên tục, cần ngưỡng để đổi sang nhãn

### Rule-based Spam Detection

- bắt các mẫu spam phổ biến
- xử lý tốt các comment quảng cáo rõ ràng
- bổ sung cho mô hình học máy ở tầng quyết định cuối

## 5. Chức năng của ứng dụng

Ứng dụng Streamlit gồm 3 trang:

### 1. Giới thiệu & EDA

- mô tả đề tài
- hiển thị dữ liệu thô
- biểu đồ phân phối nhãn
- biểu đồ độ dài bình luận
- biểu đồ phân bố theo chủ đề

### 2. Triển khai mô hình

- nhập một bình luận để dự đoán
- hiển thị điểm toxic và quyết định gợi ý
- dự đoán hàng loạt bằng file CSV

### 3. Đánh giá & Hiệu năng

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Classification Report
- phân tích lỗi dự đoán

## 6. Cấu trúc thư mục

```text
toxic_comment_project/
├── app.py
├── requirements.txt
├── README.md
├── toxic_comment_filter_colab_local_fixed_synced.ipynb
├── data/
├── models/
├── reports/
└── .gitattributes
```

## 7. Chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 8. Ghi chú

- Phiên bản hiện tại chỉ dùng **một mô hình duy nhất là Linear Regression**.
- PhoBERT đã được loại khỏi luồng huấn luyện, suy luận và tài liệu.
- Nếu chưa có file model hoặc reports mới, hãy chạy notebook để tạo lại artifact trong `models/` và `reports/`.
