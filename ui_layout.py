import gradio as gr

# --- CÁC ĐOẠN MÃ GIAO DIỆN TĨNH (Không đổi, chỉ cập nhật tiêu đề) ---
custom_css = """
/* Ẩn footer mặc định của Gradio */
footer {display: none !important}
/* Custom styling cho gallery */
.gallery { border-radius: 12px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.05) !important; }
/* Đảm bảo gallery chính có thể cuộn được */
#results-gallery > .gradio-gallery { height: 700px !important; overflow-y: auto !important; }
/* Animation cho buttons */
.gradio-button { transition: all 0.2s ease !important; border-radius: 20px !important; font-weight: 600 !important; }
.gradio-button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; }
/* Custom textbox styling */
.gradio-textbox { border-radius: 10px !important; border: 1px solid #e0e0e0 !important; transition: all 0.2s ease !important; }
.gradio-textbox:focus { border-color: #667eea !important; box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important; }
/* Video player styling */
video { border-radius: 12px !important; }
/* Hiệu ứng hover cho ảnh trong gallery */
.gallery img { transition: transform 0.2s ease !important; border-radius: 8px !important; }
.gallery img:hover { transform: scale(1.04) !important; }
/* Tùy chỉnh thanh cuộn */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); }
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
    Xây dựng toàn bộ giao diện người dùng và kết nối các sự kiện.
    PHIÊN BẢN V2: Cấu trúc Tabs "Mắt Thần" và "Tai Thính".
    """
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🚀 AIC25 Search Fleet") as app:
        
        # --- Khai báo tất cả States cần thiết cho toàn bộ App ---
        response_state = gr.State()
        gallery_items_state = gr.State([])
        current_page_state = gr.State(1)
        submission_list_state = gr.State([])
        selected_candidate_for_submission = gr.State()
        transcript_results_state = gr.State() # State cho kết quả tìm kiếm transcript

        gr.HTML(app_header_html)
        
        with gr.Row(variant='panel'):
            # --- CỘT TRÁI (scale=2): KHU VỰC TÌM KIẾM CHÍNH ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    # --- TAB 1: MẮT THẦN (VISUAL SCOUT) ---
                    with gr.TabItem("👁️ Mắt Thần (Visual Scout)"):
                        gr.Markdown("### 1. Tìm kiếm bằng Hình ảnh & Ngữ nghĩa")
                        query_input = gr.Textbox(label="🔍 Nhập mô tả cảnh bạn muốn tìm...", placeholder="Ví dụ: một người phụ nữ mặc váy đỏ đang nói về việc bảo tồn rùa biển...", lines=2, autofocus=True)
                        with gr.Row():
                            search_button = gr.Button("🚀 Quét Visual", variant="primary", size="lg")
                            clear_button = gr.Button("🗑️ Xóa Tất cả", variant="secondary", size="lg")
                        num_results = gr.Slider(minimum=50, maximum=1000, value=200, step=50, label="📊 Số lượng kết quả visual tối đa")
                        
                        with gr.Accordion("⚙️ Tùy chỉnh Reranking Nâng cao", open=False):
                            # ... (giữ nguyên các slider cũ)
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
                        gr.Markdown("### 1. Điều tra bằng Lời thoại (Hỗ trợ tìm kiếm lồng)")
                        transcript_query_1 = gr.Textbox(label="🔍 Tìm kiếm trong toàn bộ transcript...", placeholder="Ví dụ: biến đổi khí hậu")
                        transcript_query_2 = gr.Textbox(label="...và trong kết quả đó, tìm tiếp...", placeholder="Ví dụ: Việt Nam")
                        transcript_query_3 = gr.Textbox(label="...cuối cùng, lọc theo...", placeholder="Ví dụ: giải pháp")
                        with gr.Row():
                            transcript_search_button = gr.Button("🎙️ Bắt đầu Điều tra", variant="primary")
                            transcript_clear_button = gr.Button("🧹 Xóa bộ lọc")
                        
                        gr.Markdown("### 2. Kết quả Điều tra")
                        transcript_results_count = gr.Markdown("Tìm thấy: 0 kết quả.")
                        transcript_results_df = gr.DataFrame(
                            headers=["Video ID", "Timestamp (s)", "Nội dung Lời thoại", "Keyframe Path"],
                            datatype=["str", "number", "str", "str"],
                            row_count=10,
                            col_count=(4, "fixed"),
                            wrap=True,
                            interactive=True,
                            visible=True,
                            column_widths=["15%", "15%", "60%", "0%"] # Ẩn cột Keyframe Path
                        )
                        gr.Markdown("### 3. Xem Video từ Lời thoại")
                        transcript_video_player = gr.Video(label="🎬 Video gốc (tua đến thời điểm được chọn)", interactive=False)

            # --- CỘT PHẢI (scale=1): TRẠM PHÂN TÍCH & NỘP BÀI (DÙNG CHUNG) ---
            with gr.Column(scale=1):
                gr.Markdown("### 🔬 Trạm Phân tích & Nộp bài")
                
                with gr.Accordion("Trạm Phân tích Visual", open=True):
                    selected_image_display = gr.Image(label="Ảnh Keyframe Được chọn", type="filepath")
                    video_player = gr.Video(label="🎬 Clip 30 giây", autoplay=True)
                    view_full_video_button = gr.Button("🎬 Mở Video Gốc (Toàn bộ)")

                with gr.Accordion("📋 Vùng Nộp bài", open=True):
                    with gr.Row():
                        add_top_button = gr.Button("➕ Thêm vào Top 1", variant="primary")
                        add_bottom_button = gr.Button("➕ Thêm vào cuối")
                    submission_list_display = gr.Textbox(label="Thứ tự Nộp bài (Top 1 ở trên cùng)", lines=8, interactive=False, value="Chưa có kết quả nào.")
                    submission_list_selector = gr.Dropdown(label="Chọn mục để thao tác", choices=[], interactive=True)
                    with gr.Row():
                        move_up_button = gr.Button("⬆️ Lên")
                        move_down_button = gr.Button("⬇️ Xuống")
                        remove_button = gr.Button("🗑️ Xóa", variant="stop")
                    clear_submission_button = gr.Button("💥 Xóa toàn bộ danh sách")

                with gr.Accordion("🧮 Công cụ Tính toán Frame", open=False):
                    frame_calculator_video_id = gr.Textbox(label="Video ID", placeholder="Tự động điền khi chọn ảnh...")
                    frame_calculator_timestamp = gr.Number(label="Timestamp (giây)", value=0)
                    frame_calculator_button = gr.Button("Tính toán Frame Index")
                    frame_calculator_output = gr.Textbox(label="✅ Frame Index để nộp bài", interactive=False)

                with gr.Accordion("💾 Xuất File Nộp bài", open=True):
                    query_id_input = gr.Textbox(label="Nhập Query ID", placeholder="Ví dụ: query_01")
                    submission_button = gr.Button("💾 Tạo File CSV Nộp bài")
                    submission_file_output = gr.File(label="Tải file nộp bài tại đây")
        
        gr.HTML(app_footer_html)
        
        # --- TẬP TRUNG TOÀN BỘ COMPONENTS VÀO MỘT DICTIONARY ĐỂ QUẢN LÝ ---
        components = {
            # States
            "response_state": response_state, "gallery_items_state": gallery_items_state,
            "current_page_state": current_page_state, "submission_list_state": submission_list_state,
            "selected_candidate_for_submission": selected_candidate_for_submission,
            "transcript_results_state": transcript_results_state,
            
            # Tab Mắt Thần - Inputs
            "query_input": query_input, "search_button": search_button, "num_results": num_results,
            "w_clip_slider": w_clip_slider, "w_obj_slider": w_obj_slider, "w_semantic_slider": w_semantic_slider,
            "lambda_mmr_slider": lambda_mmr_slider, "clear_button": clear_button,
            
            # Tab Mắt Thần - Outputs & Display
            "status_output": status_output, "prev_page_button": prev_page_button,
            "page_info_display": page_info_display, "next_page_button": next_page_button,
            "results_gallery": results_gallery,
            
            # Tab Tai Thính
            "transcript_query_1": transcript_query_1, "transcript_query_2": transcript_query_2,
            "transcript_query_3": transcript_query_3, "transcript_search_button": transcript_search_button,
            "transcript_clear_button": transcript_clear_button, "transcript_results_count": transcript_results_count,
            "transcript_results_df": transcript_results_df, "transcript_video_player": transcript_video_player,
            
            # Cột Phải - Trạm Phân tích
            "selected_image_display": selected_image_display, "video_player": video_player,
            "view_full_video_button": view_full_video_button,
            
            # Cột Phải - Vùng Nộp bài
            "add_top_button": add_top_button, "add_bottom_button": add_bottom_button,
            "submission_list_display": submission_list_display, "submission_list_selector": submission_list_selector,
            "move_up_button": move_up_button, "move_down_button": move_down_button, "remove_button": remove_button,
            "clear_submission_button": clear_submission_button,
            
            # Cột Phải - Công cụ Tính toán
            "frame_calculator_video_id": frame_calculator_video_id, "frame_calculator_timestamp": frame_calculator_timestamp,
            "frame_calculator_button": frame_calculator_button, "frame_calculator_output": frame_calculator_output,
            
            # Cột Phải - Vùng Xuất File
            "query_id_input": query_id_input, "submission_button": submission_button,
            "submission_file_output": submission_file_output,
        }

        connect_events_fn(components)

    return app