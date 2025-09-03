import gradio as gr

custom_css = """
/* === CÀI ĐẶT CHUNG & RESET === */
footer {
    display: none !important;
}

/* === STYLING CHO CÁC COMPONENT CHÍNH === */
.gradio-button {
    transition: all 0.2s ease !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
}
.gradio-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}
.gradio-textbox {
    border-radius: 10px !important;
    border: 1px solid #e0e0e0 !important;
    transition: all 0.2s ease !important;
}
.gradio-textbox:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
}
video {
    border-radius: 12px !important;
}

/* === STYLING CHO TAB "MẮT THẦN" (VISUAL SCOUT) === */
.gallery {
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05) !important;
}
#results-gallery > .gradio-gallery {
    height: 700px !important;
    overflow-y: auto !important;
}
.gallery img {
    transition: transform 0.2s ease !important;
    border-radius: 8px !important;
}
.gallery img:hover {
    transform: scale(1.04) !important;
}

/* === STYLING CHO TAB "TAI THÍNH" (TRANSCRIPT INTEL) === */
#transcript-dataframe {
    height: 600px !important; /* Đặt chiều cao cố định cho toàn bộ bảng */
    overflow-y: auto !important; /* Thêm thanh cuộn cho cả bảng nếu cần */
}
/* Sửa lỗi giãn dòng, áp dụng cho các ô chứa text */
#transcript-dataframe table tbody tr td div {
    max-height: 4.5em !important; /* Giới hạn chiều cao tương đương ~3 dòng text */
    overflow-y: auto !important; /* Thêm thanh cuộn BÊN TRONG ô nếu nội dung dài */
    line-height: 1.5em !important; /* Đảm bảo chiều cao dòng nhất quán */
    white-space: normal !important; /* Cho phép text tự xuống dòng */
    padding: 4px 6px !important; /* Thêm một chút đệm cho đẹp */
    text-align: left !important; /* Căn lề trái cho dễ đọc */
}

/* === TÙY CHỈNH THANH CUỘN (SCROLLBAR) === */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}
"""

app_header_html = """
<div style="text-align: center; max-width: 1200px; margin: 0 auto 25px auto;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px 20px; border-radius: 20px; color: white; box-shadow: 0 8px 30px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">🚀 AIC25 Search Fleet - Hạm đội Tìm kiếm</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">Chiến lược Tấn công Hai Gọng Kìm</p>
    </div>
</div>
"""

app_footer_html = """
<div style="text-align: center; margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 12px;">
    <p style="margin: 0; color: #6c757d;">AIC25 Search Fleet - Powered by Visual Scout & Transcript Intelligence</p>
</div>
"""

def build_ui(connect_events_fn):
    """
    Xây dựng toàn bộ giao diện người dùng.
    PHIÊN BẢN CUỐI CÙNG, ĐÃ SỬA LỖI VÀ HỢP NHẤT.
    """
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🚀 AIC25 Search Fleet") as app:
        
        # --- Khai báo States ---
        response_state = gr.State()
        gallery_items_state = gr.State([])
        current_page_state = gr.State(1)
        submission_list_state = gr.State([])
        selected_candidate_for_submission = gr.State()
        transcript_results_state = gr.State()
        transcript_selected_index_state = gr.State()
        video_path_map_state = gr.State()

        gr.HTML(app_header_html)
        
        with gr.Row(variant='panel'):
            # --- CỘT TRÁI (scale=2): KHU VỰC TÌM KIẾM CHÍNH ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    # --- TAB 1: MẮT THẦN (VISUAL SCOUT) ---
                    with gr.TabItem("👁️ Mắt Thần (Visual Scout)"):
                        gr.Markdown("### 1. Tìm kiếm bằng Hình ảnh & Ngữ nghĩa")
                        query_input = gr.Textbox(label="🔍 Nhập mô tả cảnh bạn muốn tìm...", placeholder="Ví dụ: một người phụ nữ mặc váy đỏ...", lines=2, autofocus=True)
                        with gr.Row():
                            search_button = gr.Button("🚀 Quét Visual", variant="primary", size="lg")
                            clear_button = gr.Button("🗑️ Xóa Tất cả", variant="secondary", size="lg")
                        num_results = gr.Slider(minimum=50, maximum=1000, value=200, step=50, label="📊 Số lượng kết quả visual tối đa")
                        with gr.Accordion("⚙️ Tùy chỉnh Reranking Nâng cao", open=False):
                            w_clip_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="w_clip (Thị giác)")
                            w_obj_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_obj (Đối tượng)")
                            w_semantic_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="w_semantic (Ngữ nghĩa)")
                            lambda_mmr_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="λ - MMR (Đa dạng hóa)")
                        status_output = gr.HTML()
                        gr.Markdown("### 2. Kết quả Visual")
                        with gr.Row(equal_height=True, variant='compact'):
                            prev_page_button = gr.Button("◀️ Trang trước")
                            page_info_display = gr.Markdown("Trang 1 / 1", elem_id="page-info")
                            next_page_button = gr.Button("▶️ Trang sau")
                        results_gallery = gr.Gallery(label="Click vào một ảnh để phân tích", show_label=True, elem_id="results-gallery", columns=5, object_fit="contain", height=700, allow_preview=False)

                    # --- TAB 2: TAI THÍNH (TRANSCRIPT INTEL) ---
                    with gr.TabItem("👂 Tai Thính (Transcript Intel)"):
                        gr.Markdown("### 1. Điều tra bằng Lời thoại")
                        transcript_query_1 = gr.Textbox(label="🔍 Tìm kiếm trong toàn bộ transcript...", placeholder="Ví dụ: biến đổi khí hậu")
                        transcript_query_2 = gr.Textbox(label="...và trong kết quả đó, tìm tiếp...", placeholder="Ví dụ: Việt Nam")
                        transcript_query_3 = gr.Textbox(label="...cuối cùng, lọc theo...", placeholder="Ví dụ: giải pháp")
                        with gr.Row():
                            transcript_search_button = gr.Button("🎙️ Bắt đầu Điều tra", variant="primary")
                            transcript_clear_button = gr.Button("🧹 Xóa bộ lọc")
                        gr.Markdown("### 2. Kết quả Điều tra & Nộp bài")
                        transcript_results_count = gr.Markdown("Tìm thấy: 0 kết quả.")
                        with gr.Row():
                             add_transcript_top_button = gr.Button("➕ Thêm kết quả đã chọn vào Top 1", variant="primary")
                             add_transcript_bottom_button = gr.Button("➕ Thêm kết quả đã chọn vào cuối")
                        transcript_results_df = gr.DataFrame(headers=["Video ID", "Timestamp (s)", "Nội dung Lời thoại", "Keyframe Path"], datatype=["str", "number", "str", "str"], row_count=10, col_count=(4, "fixed"), wrap=True, interactive=True, visible=True, column_widths=["15%", "15%", "60%", "0%"], elem_id="transcript-dataframe")
            
            # --- CỘT PHẢI (scale=1): TRẠM PHÂN TÍCH & NỘP BÀI (DÙNG CHUNG) ---
            with gr.Column(scale=1):
                gr.Markdown("### 🔬 Trạm Phân tích Hợp nhất")
                with gr.Accordion("Media Player & Phân tích", open=True):
                    selected_image_display = gr.Image(label="🖼️ Keyframe được chọn", type="filepath")
                    video_player = gr.Video(label="🎬 Media Player", autoplay=False)
                    full_transcript_display = gr.Textbox(label="📜 Transcript (nếu có)", lines=10, interactive=False, placeholder="Nội dung transcript của video sẽ hiện ở đây...")
                    analysis_display_html = gr.HTML(label="📊 Phân tích Điểm số (cho Visual Search)")
                    with gr.Accordion("🎬 Trình phát Video Gốc (Toàn bộ)", open=False):
                        view_full_video_button = gr.Button("▶️ Tải và Xem Toàn bộ Video Gốc (có thể mất vài giây)")
                        full_video_player = gr.Video(label="🎬 Video Gốc", interactive=False)

                    with gr.Row():
                        add_top_button = gr.Button("➕ Thêm (từ Visual) vào Top 1", variant="primary")
                        add_bottom_button = gr.Button("➕ Thêm (từ Visual) vào cuối")
                with gr.Accordion("📋 Bảng điều khiển Nộp bài", open=True):
                    gr.Markdown("Nội dung dưới đây sẽ được lưu vào file CSV. **Bạn có thể chỉnh sửa trực tiếp.**")
                    submission_text_editor = gr.Textbox(label="Nội dung File Nộp bài (Định dạng CSV)", lines=15, interactive=True, placeholder="Thêm kết quả từ các tab tìm kiếm hoặc dán trực tiếp vào đây...")
                    refresh_submission_button = gr.Button("🔄 Cập nhật/Đồng bộ hóa Bảng điều khiển")
                    clear_submission_button = gr.Button("💥 Xóa toàn bộ Danh sách & Bảng điều khiển", variant="stop")
                with gr.Accordion("🧮 Máy tính Thời gian & Frame", open=False):
                    frame_calculator_video_id = gr.Textbox(label="Video ID", placeholder="Tự động điền khi chọn ảnh...")
                    frame_calculator_time_input = gr.Textbox(label="Nhập Thời gian", placeholder="Ví dụ: 123.45 (giây) hoặc 2:03.45 (phút:giây)")
                    frame_calculator_button = gr.Button("Tính toán Frame Index")
                    frame_calculator_output = gr.Textbox(label="✅ Kết quả Frame Index (để copy)", interactive=False, show_copy_button=True)
                with gr.Accordion("💾 Xuất File Nộp bài", open=True):
                    query_id_input = gr.Textbox(label="Nhập Query ID", placeholder="Ví dụ: query_01")
                    submission_button = gr.Button("💾 Tạo File CSV (từ nội dung đã sửa)")
                    submission_file_output = gr.File(label="Tải file nộp bài tại đây")
        
        gr.HTML(app_footer_html)
        
        # --- TẬP TRUNG TOÀN BỘ COMPONENTS VÀO MỘT DICTIONARY ĐỂ QUẢN LÝ ---
        components = {
            # States
            "response_state": response_state, "gallery_items_state": gallery_items_state,
            "current_page_state": current_page_state, "submission_list_state": submission_list_state,
            "selected_candidate_for_submission": selected_candidate_for_submission,
            "transcript_results_state": transcript_results_state,
            "transcript_selected_index_state": transcript_selected_index_state,
            "video_path_map_state": video_path_map_state,
            # Tab Mắt Thần
            "query_input": query_input, "search_button": search_button, "num_results": num_results,
            "w_clip_slider": w_clip_slider, "w_obj_slider": w_obj_slider, "w_semantic_slider": w_semantic_slider,
            "lambda_mmr_slider": lambda_mmr_slider, "clear_button": clear_button,
            "status_output": status_output, "prev_page_button": prev_page_button,
            "page_info_display": page_info_display, "next_page_button": next_page_button,
            "results_gallery": results_gallery,
            # Tab Tai Thính
            "transcript_query_1": transcript_query_1, "transcript_query_2": transcript_query_2,
            "transcript_query_3": transcript_query_3, "transcript_search_button": transcript_search_button,
            "transcript_clear_button": transcript_clear_button, "transcript_results_count": transcript_results_count,
            "add_transcript_top_button": add_transcript_top_button, "add_transcript_bottom_button": add_transcript_bottom_button,
            "transcript_results_df": transcript_results_df,
            # Cột Phải - Trạm Phân tích Hợp nhất
            "selected_image_display": selected_image_display, "video_player": video_player,
            "full_transcript_display": full_transcript_display, "analysis_display_html": analysis_display_html,
            "view_full_video_button": view_full_video_button, "add_top_button": add_top_button,
            "add_bottom_button": add_bottom_button,
            # Cột Phải - Bảng điều khiển Nộp bài
            "submission_text_editor": submission_text_editor,
            "refresh_submission_button": refresh_submission_button,
            "clear_submission_button": clear_submission_button,
            # Cột Phải - Máy tính Thời gian
            "frame_calculator_video_id": frame_calculator_video_id, "frame_calculator_time_input": frame_calculator_time_input,
            "frame_calculator_button": frame_calculator_button, "frame_calculator_output": frame_calculator_output,
            # Cột Phải - Vùng Xuất File
            "query_id_input": query_id_input, "submission_button": submission_button,
            "submission_file_output": submission_file_output,
        }

        connect_events_fn(components)

    return app, components