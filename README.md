# 🚀 AIC25 Search Fleet - Hạm đội Tìm kiếm Video

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Giới thiệu

**AIC25 Search Fleet** là một công cụ tìm kiếm video thông minh và mạnh mẽ được phát triển cho cuộc thi AI City Challenge 2025. Dự án được xây dựng với kiến trúc "Tấn công Hai Gọng Kìm", cho phép người dùng khai thác thông tin từ video thông qua cả hai phương diện: hình ảnh và lời thoại.

- **👁️ Mắt Thần (Visual Scout):** Tìm kiếm dựa trên mô tả ngữ nghĩa và hình ảnh. Người dùng có thể mô tả một cảnh tượng phức tạp, và hệ thống sẽ phân tích, phân rã truy vấn, sau đó tìm kiếm và tái xếp hạng các keyframe phù hợp nhất bằng một quy trình đa tầng tinh vi.
- **👂 Tai Thính (Transcript Intel):** Điều tra dựa trên nội dung lời thoại. Người dùng có thể lọc qua hàng triệu dòng transcript để nhanh chóng tìm ra những khoảnh khắc mà một từ khóa hoặc cụm từ cụ thể được nhắc đến.

Hệ thống được trang bị giao diện người dùng trực quan xây dựng bằng Gradio, giúp các nhà phân tích dễ dàng tương tác, khám phá kết quả, và quản lý danh sách nộp bài một cách hiệu quả.

## ✨ Tính năng nổi bật

*   **Tìm kiếm Ngữ nghĩa Đa tầng (Phoenix Reranking):**
    *   **Phân rã Truy vấn:** Tự động chia một truy vấn phức tạp thành nhiều truy vấn con đơn giản, tập trung vào các yếu tố hình ảnh cốt lõi.
    *   **Lọc Đa tầng:** Áp dụng nhiều lớp bộ lọc để tái xếp hạng kết quả: điểm tương đồng hình ảnh (CLIP), điểm ngữ nghĩa (Bi-Encoder), điểm phù hợp không gian (Spatial Scoring), và xác thực chi tiết (Fine-grained Verification).
    *   **Đa dạng hóa Kết quả (MMR):** Sử dụng thuật toán Maximal Marginal Relevance để đảm bảo các kết quả trả về không bị trùng lặp và bao quát được nhiều khía cạnh của truy vấn.
*   **Tìm kiếm Lời thoại Nâng cao:**
    *   Lọc lồng nhau với nhiều từ khóa để thu hẹp phạm vi tìm kiếm.
    *   Tự động tô sáng (highlight) từ khóa trong kết quả để dễ dàng xác định.
*   **Giao diện Tương tác Toàn diện:**
    *   Hiển thị kết quả dưới dạng lưới ảnh (gallery) có phân trang.
    *   Khi chọn một kết quả, hệ thống tự động hiển thị keyframe, một đoạn video clip 30 giây xung quanh khoảnh khắc đó, và toàn bộ lời thoại của video.
    *   Cung cấp bảng phân tích điểm số chi tiết cho các kết quả tìm kiếm bằng hình ảnh.
*   **Quản lý Nộp bài Chuyên nghiệp:**
    *   Thêm/bớt kết quả vào danh sách nộp bài từ cả hai tab tìm kiếm (Visual và Transcript).
    *   Chỉnh sửa trực tiếp nội dung file nộp bài ngay trên giao diện.
    *   Tự động tính toán số thứ tự frame (frame index) từ timestamp và FPS của video.
    *   Xuất file `.csv` theo đúng định dạng yêu cầu của ban tổ chức.
*   **Công cụ Hỗ trợ:**
    *   **Frame Calculator:** Một tiện ích nhỏ để nhanh chóng chuyển đổi giữa timestamp (giây hoặc phút:giây) và số thứ tự frame.
    *   **Trình phát Video Gốc:** Cho phép tải và xem toàn bộ video gốc chỉ với một cú nhấp chuột.

## 🏗️ Kiến trúc Hệ thống

Hệ thống được chia thành ba khối chính: **Giao diện người dùng (UI)**, **Backend Logic**, và **Lõi Tìm kiếm (Search Core)**.

1.  **Giao diện Người dùng (Frontend - Gradio):**
    *   `app.py`: Điểm khởi chạy chính của ứng dụng.
    *   `ui_layout.py`: Định nghĩa toàn bộ cấu trúc, bố cục của các thành phần trên giao diện.
    *   `ui_helpers.py`: Chứa các hàm hỗ trợ việc tạo mã HTML động và định dạng hiển thị.
    *   `event_handlers.py`: "Bảng mạch" của ứng dụng, kết nối các hành động của người dùng (nhấn nút, chọn ảnh) với các hàm xử lý ở backend.

2.  **Backend & Tải dữ liệu:**
    *   `backend_loader.py`: Chịu trách nhiệm khởi tạo tất cả các thành-phần-cốt-lõi của hệ thống, bao gồm việc tải các mô hình AI, đọc các file dữ liệu lớn (metadata, index FAISS), và ánh xạ đường dẫn video.
    *   `config.py`: Tệp cấu hình trung tâm, chứa tất cả các đường dẫn file, hằng số và khóa API.

3.  **Lõi Tìm kiếm (Search Core):**
    *   `master_searcher.py`: "Tổng chỉ huy" của chiến dịch tìm kiếm. Nó tiếp nhận truy vấn, điều phối các `searcher` con, và tổng hợp kết quả cuối cùng.
    *   `query_decomposer.py`: Sử dụng mô hình ngôn ngữ lớn (Gemini) để phân rã truy vấn phức tạp.
    *   `semantic_searcher.py`: Chịu trách nhiệm cho quy trình tái xếp hạng đa tầng (Phoenix).
    *   `basic_searcher.py`: Thực hiện bước tìm kiếm cơ bản ban đầu bằng FAISS và CLIP.
    *   `gemini_text_handler.py` & `openai_handler.py`: Các lớp chuyên dụng để tương tác với API của Google Gemini và OpenAI cho các tác vụ phân tích văn bản và hình ảnh.
    *   `transcript_searcher.py`: Động cơ tìm kiếm hiệu năng cao trên dữ liệu lời thoại.
    *   `mmr_builder.py`: Module thực thi thuật toán MMR để đa dạng hóa kết quả.
    *   `spatial_engine.py`: Thư viện chứa các hàm logic để xác minh mối quan hệ không gian giữa các đối tượng (ví dụ: "người A đứng sau xe B").
    *   Các module khác như `task_analyzer.py`, `trake_solver.py`, `vqa_handler.py` cung cấp các chức năng chuyên biệt khác.

4.  **Tiện ích (Utils):**
    *   Thư mục `utils` chứa các hàm tiện ích đa dụng như cắt video (`video_utils.py`), định dạng file nộp bài (`formatting.py`), quản lý cache (`cache_manager.py`), v.v.

## 🚀 Cài đặt và Khởi chạy

1.  **Clone a repository:**
    ```bash
    git clone git clone https://github.com/MonsieurNam/AIC_25.git
    cd Project_AIC_Ver2
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình API Keys:**
    *   Truy cập vào dịch vụ Kaggle và thêm các API Key của bạn vào mục Secrets với các tên sau:
        *   `OPENAI_API_KEY`
        *   `GOOGLE_API_KEY` (dành cho Gemini)
    *   File `config.py` sẽ tự động đọc các secret này.

4.  **Chuẩn bị Dữ liệu:**
    *   Đảm bảo rằng các đường dẫn tới file dữ liệu trong `config.py` (ví dụ: `CLIP_FEATURES_PATH`, `FAISS_INDEX_PATH`, `RERANK_METADATA_PATH`, v.v.) là chính xác và trỏ đến vị trí các file dữ liệu của bạn.
    *   Cập nhật các đường dẫn `VIDEO_BASE_PATHS` và `KEYFRAME_BASE_PATHS` để trỏ đến các thư mục chứa video và keyframe của các batch dữ liệu.

5.  **Chạy ứng dụng:**
    ```bash
    python app.py
    ```
    Ứng dụng sẽ khởi chạy và cung cấp một đường dẫn URL công khai để bạn có thể truy cập giao diện.

## 📂 Cấu trúc Thư mục

```
Project_AIC_Ver2/
 ┣ search_core/
 ┃ ┣ basic_searcher.py         # Lõi tìm kiếm cơ bản (FAISS + CLIP)
 ┃ ┣ gemini_text_handler.py    # Xử lý văn bản bằng Gemini
 ┃ ┣ master_searcher.py        # Điều phối viên chính của hệ thống tìm kiếm
 ┃ ┣ mmr_builder.py            # Module đa dạng hóa kết quả (MMR)
 ┃ ┣ openai_handler.py         # Xử lý hình ảnh/VQA bằng OpenAI
 ┃ ┣ query_decomposer.py       # Phân rã truy vấn phức tạp
 ┃ ┣ semantic_searcher.py      # Lõi tái xếp hạng đa tầng
 ┃ ┣ task_analyzer.py          # Phân loại loại truy vấn
 ┃ ┣ trake_solver.py           # Giải quyết nhiệm vụ tìm chuỗi hành động
 ┃ ┣ transcript_searcher.py    # Lõi tìm kiếm trên lời thoại
 ┃ ┗ vqa_handler.py            # Xử lý hỏi đáp hình ảnh
 ┣ utils/
 ┃ ┣ api_utils.py              # Tiện ích cho việc gọi API (retry logic)
 ┃ ┣ cache_manager.py          # Quản lý cache cho các vector đối tượng
 ┃ ┣ formatting.py             # Định dạng dữ liệu cho hiển thị và nộp bài
 ┃ ┣ image_cropper.py          # Cắt vùng ảnh theo bounding box
 ┃ ┣ spatial_engine.py         # Logic xử lý quan hệ không gian
 ┃ ┣ video_utils.py            # Tiện ích xử lý video (cắt clip)
 ┃ ┗ __init__.py
 ┣ app.py                      # Entry point, khởi chạy ứng dụng Gradio
 ┣ backend_loader.py           # Tải và khởi tạo các thành phần backend
 ┣ config.py                   # Tệp cấu hình trung tâm
 ┣ event_handlers.py           # Xử lý sự kiện từ giao diện người dùng
 ┣ README.md                   # Tài liệu hướng dẫn này
 ┣ requirements.txt            # Danh sách các thư viện Python cần thiết
 ┣ ui_helpers.py               # Hàm hỗ trợ xây dựng giao diện (HTML động)
 ┗ ui_layout.py                # Định nghĩa bố cục giao diện Gradio
```