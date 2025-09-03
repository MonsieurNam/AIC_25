# ==============================================================================
# === AIC25 SEARCH FLEET - TRUNG TÂM CHỈ HUY CHIẾN DỊCH (app.py) ===
# ==============================================================================
print("--- 🚀 Bắt đầu khởi chạy AIC25 Search Fleet ---")

# --- GIAI ĐOẠN 1: TẢI CÁC THƯ VIỆN CẦN THIẾT ---
print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")
import gradio as gr
import pandas as pd
from functools import partial

# Local imports - Các module cốt lõi của hạm đội
from backend_loader import initialize_backend
from ui_layout import build_ui
import event_handlers as handlers

# --- GIAI ĐOẠN 2: KHỞI TẠO TOÀN BỘ BACKEND ---
print("--- Giai đoạn 2/4: Đang khởi tạo các Động cơ Backend...")
backend_objects = initialize_backend()
master_searcher = backend_objects['master_searcher']
transcript_searcher = backend_objects['transcript_searcher']
fps_map = backend_objects['fps_map']
video_path_map = backend_objects['video_path_map']
print("--- ✅ Toàn bộ Backend đã được nạp và sẵn sàng chiến đấu. ---")


# --- GIAI ĐOẠN 3: XÂY DỰNG GIAO DIỆN & KẾT NỐI MẠCH THẦN KINH ---
print("--- Giai đoạn 3/4: Đang xây dựng giao diện và kết nối sự kiện...")

# --- Chuẩn bị các hàm xử lý sự kiện bằng `partial` ---
# Kỹ thuật này "tiêm" các đối tượng backend cần thiết vào hàm xử lý,
# giúp mã nguồn sạch sẽ và không cần biến toàn cục.

# Handlers cho Mắt Thần (Visual Scout)
search_with_backend = partial(handlers.perform_search, master_searcher=master_searcher)

# Handlers cho Tai Thính (Transcript Intel)
transcript_search_with_backend = partial(handlers.handle_transcript_search, transcript_searcher=transcript_searcher)

# Handlers Hợp nhất cho Trạm Phân tích
on_gallery_select_with_backend = partial(handlers.on_gallery_select, transcript_searcher=transcript_searcher)
on_transcript_select_with_backend = partial(
    handlers.on_transcript_select, 
    video_path_map=video_path_map,
    transcript_searcher=transcript_searcher
)

# Handlers cho các Công cụ Phụ trợ
calculate_frame_with_backend = partial(handlers.calculate_frame_number, fps_map=fps_map)


def connect_event_listeners(ui_components):
    """
    Kết nối TẤT CẢ các sự kiện của component UI với các hàm xử lý tương ứng.
    Đây là "bảng mạch" chính của toàn bộ ứng dụng.
    """
    ui = ui_components # Viết tắt cho gọn

    # === 1. KẾT NỐI SỰ KIỆN CHO TAB "MẮT THẦN" (VISUAL SCOUT) ===
    
    # 1.1. Nút Tìm kiếm chính và ô nhập liệu
    visual_search_inputs = [
        ui["query_input"], ui["num_results"], 
        ui["w_clip_slider"], ui["w_obj_slider"], 
        ui["w_semantic_slider"], ui["lambda_mmr_slider"]
    ]
    visual_search_outputs = [
        ui["results_gallery"], ui["status_output"], ui["response_state"], 
        ui["gallery_items_state"], ui["current_page_state"], ui["page_info_display"]
    ]
    ui["search_button"].click(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)
    ui["query_input"].submit(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)

    # 1.2. Nút Phân trang
    page_outputs = [ui["results_gallery"], ui["current_page_state"], ui["page_info_display"]]
    ui["prev_page_button"].click(
        fn=handlers.update_gallery_page,
        inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("◀️ Trang trước", visible=False)],
        outputs=page_outputs,
        queue=False
    )
    ui["next_page_button"].click(
        fn=handlers.update_gallery_page,
        inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("▶️ Trang sau", visible=False)],
        outputs=page_outputs,
        queue=False
    )
    
    # === 2. KẾT NỐI SỰ KIỆN CHO TAB "TAI THÍNH" (TRANSCRIPT INTEL) ===
    
    # 2.1. Nút Tìm kiếm và Xóa bộ lọc
    transcript_inputs = [ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"]]
    transcript_outputs = [ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]]
    ui["transcript_search_button"].click(fn=transcript_search_with_backend, inputs=transcript_inputs, outputs=transcript_outputs)

    transcript_clear_outputs = [
        ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"],
        ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]
    ]
    ui["transcript_clear_button"].click(fn=handlers.clear_transcript_search, inputs=None, outputs=transcript_clear_outputs, queue=False)

    # === 3. KẾT NỐI SỰ KIỆN HỢP NHẤT CHO TRẠM PHÂN TÍCH (CỘT PHẢI) ===
    
    # Định nghĩa các component output ở cột phải MỘT LẦN và dùng chung
    analysis_panel_outputs = [
        ui["selected_image_display"], ui["video_player"],
        ui["full_transcript_display"], ui["analysis_display_html"],
        ui["view_full_video_html"],
        ui["selected_candidate_for_submission"],
        ui["frame_calculator_video_id"], ui["frame_calculator_time_input"],
        ui["transcript_selected_index_state"]
    ]

    # 3.1. Sự kiện chọn từ Mắt Thần
    ui["results_gallery"].select(
        fn=on_gallery_select_with_backend,
        inputs=[ui["response_state"], ui["current_page_state"]],
        outputs=analysis_panel_outputs
    )
    
    # 3.2. Sự kiện chọn từ Tai Thính
    ui["transcript_results_df"].select(
        fn=on_transcript_select_with_backend,
        inputs=[ui["transcript_results_state"]],
        outputs=analysis_panel_outputs,
    )

    # === 4. KẾT NỐI SỰ KIỆN CHO BẢNG ĐIỀU KHIỂN NỘP BÀI (CỘT PHẢI) ===
    
    # 4.1. Thêm kết quả từ Mắt Thần
    ui["add_top_button"].click(
        fn=handlers.add_to_submission_list,
        inputs=[ui["submission_list_state"], ui["selected_candidate_for_submission"], gr.Textbox("top", visible=False)],
        outputs=[ui["submission_list_state"], ui["submission_text_editor"]]
    )
    ui["add_bottom_button"].click(
        fn=handlers.add_to_submission_list,
        inputs=[ui["submission_list_state"], ui["selected_candidate_for_submission"], gr.Textbox("bottom", visible=False)],
        outputs=[ui["submission_list_state"], ui["submission_text_editor"]]
    )
    
    # 4.2. Thêm kết quả từ Tai Thính
    ui["add_transcript_top_button"].click(
        fn=handlers.add_transcript_result_to_submission,
        inputs=[ui["submission_list_state"], ui["transcript_results_state"], ui["transcript_selected_index_state"], gr.Textbox("top", visible=False)],
        outputs=[ui["submission_list_state"], ui["submission_text_editor"]]
    )
    ui["add_transcript_bottom_button"].click(
        fn=handlers.add_transcript_result_to_submission,
        inputs=[ui["submission_list_state"], ui["transcript_results_state"], ui["transcript_selected_index_state"], gr.Textbox("bottom", visible=False)],
        outputs=[ui["submission_list_state"], ui["submission_text_editor"]]
    )
    
    # 4.3. Cập nhật và Xóa Bảng điều khiển
    ui["refresh_submission_button"].click(
        fn=handlers.sync_submission_state_to_editor,
        inputs=[ui["submission_list_state"]],
        outputs=[ui["submission_text_editor"]],
        queue=False
    )
    ui["clear_submission_button"].click(
        fn=handlers.clear_submission_list,
        inputs=None,
        outputs=[ui["submission_list_state"], ui["submission_text_editor"]],
        queue=False
    )
    
    # === 5. KẾT NỐI SỰ KIỆN CHO CÁC CÔNG CỤ CÒN LẠI (CỘT PHẢI) ===
    
    # 5.1. Máy tính Thời gian & Frame
    ui["frame_calculator_button"].click(
        fn=calculate_frame_with_backend,
        inputs=[ui["frame_calculator_video_id"], ui["frame_calculator_time_input"]],
        outputs=[ui["frame_calculator_output"]],
        queue=False
    )

    # 5.2. Xuất File Nộp bài
    ui["submission_button"].click(
        fn=handlers.handle_submission,
        inputs=[ui["submission_text_editor"], ui["query_id_input"]],
        outputs=[ui["submission_file_output"]]
    )
    
    # 5.3. Nút Xóa Tất cả
    # Đây là nút "reset" toàn bộ hệ thống
    clear_all_outputs = [
        # Mắt Thần
        ui["query_input"], ui["results_gallery"], ui["status_output"], ui["response_state"],
        ui["gallery_items_state"], ui["current_page_state"], ui["page_info_display"],
        # Tai Thính
        ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"],
        ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"],
        # Trạm Phân tích Hợp nhất
        ui["selected_image_display"], ui["video_player"], ui["full_transcript_display"],
        ui["analysis_display_html"], ui["view_full_video_html"], ui["selected_candidate_for_submission"],
        # Bảng điều khiển Nộp bài
        ui["submission_list_state"], ui["submission_text_editor"],
        # Máy tính
        ui["frame_calculator_video_id"], ui["frame_calculator_time_input"], ui["frame_calculator_output"],
        # Vùng Xuất file
        ui["query_id_input"], ui["submission_file_output"]
    ]
    ui["clear_button"].click(fn=handlers.clear_all, inputs=None, outputs=clear_all_outputs, queue=False)


# --- Xây dựng UI và truyền hàm kết nối sự kiện vào ---
app, ui_components = build_ui(connect_event_listeners)

app.load(lambda: video_path_map, inputs=None, outputs=ui_components["video_path_map_state"])

# --- GIAI ĐOẠN 4: KHỞI CHẠY APP SERVER ---
if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server (Hạm đội Gọng Kìm Kép - Phiên bản Hoàn thiện) ---")
    app.launch(
        share=True,
        # Cung cấp các đường dẫn được phép để Gradio có thể phục vụ file video
        allowed_paths=["/kaggle/input/", "/kaggle/working/"],
        debug=True,
        show_error=True
    )