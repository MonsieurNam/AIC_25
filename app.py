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


# --- GIAI ĐOẠN 3: CHUẨN BỊ HANDLER & KẾT NỐI SỰ KIỆN ---
print("--- Giai đoạn 3/4: Đang xây dựng giao diện và kết nối mạch thần kinh...")

# === TẠO WRAPPER `partial` CHO TẤT CẢ CÁC HANDLER CÓ PHỤ THUỘC ===
# Kỹ thuật này "tiêm" các đối tượng backend cần thiết vào hàm xử lý,
# giúp mã nguồn sạch sẽ và không cần biến toàn cục.

# Handlers cho Mắt Thần
search_with_backend = partial(handlers.perform_search, master_searcher=master_searcher)

# Handlers cho Tai Thính
transcript_search_with_backend = partial(handlers.handle_transcript_search, transcript_searcher=transcript_searcher)

# Handlers Hợp nhất cho Trạm Phân tích
on_gallery_select_with_backend = partial(handlers.on_gallery_select, transcript_searcher=transcript_searcher)
on_transcript_select_with_backend = partial(
    handlers.on_transcript_select, 
    video_path_map=video_path_map,
    transcript_searcher=transcript_searcher
)

# Handlers cho Bảng điều khiển Nộp bài
add_to_submission_with_backend = partial(handlers.add_to_submission_list, fps_map=fps_map)
add_transcript_to_submission_with_backend = partial(handlers.add_transcript_result_to_submission, fps_map=fps_map)
sync_editor_with_backend = partial(handlers.sync_submission_state_to_editor, fps_map=fps_map)

# Handlers cho Công cụ Phụ trợ
calculate_frame_with_backend = partial(handlers.calculate_frame_number, fps_map=fps_map)


def connect_event_listeners(ui_components):
    """
    Kết nối TẤT CẢ các sự kiện của component UI với các hàm xử lý tương ứng.
    Đây là "bảng mạch" chính của toàn bộ ứng dụng.
    """
    ui = ui_components # Viết tắt cho gọn

    # === 1. KẾT NỐI SỰ KIỆN CHO TAB "MẮT THẦN" (VISUAL SCOUT) ===
    visual_search_inputs = [
        ui["query_input"], ui["num_results"], ui["w_clip_slider"], 
        ui["w_obj_slider"], ui["w_semantic_slider"], ui["lambda_mmr_slider"]
    ]
    visual_search_outputs = [
        ui["results_gallery"], ui["status_output"], ui["response_state"], 
        ui["gallery_items_state"], ui["current_page_state"], ui["page_info_display"]
    ]
    ui["search_button"].click(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)
    ui["query_input"].submit(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)

    page_outputs = [ui["results_gallery"], ui["current_page_state"], ui["page_info_display"]]
    ui["prev_page_button"].click(fn=handlers.update_gallery_page, inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("◀️ Trang trước", visible=False)], outputs=page_outputs, queue=False)
    ui["next_page_button"].click(fn=handlers.update_gallery_page, inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("▶️ Trang sau", visible=False)], outputs=page_outputs, queue=False)
    
    # === 2. KẾT NỐI SỰ KIỆN CHO TAB "TAI THÍNH" (TRANSCRIPT INTEL) ===
    transcript_inputs = [ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"]]
    transcript_outputs = [ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]]
    ui["transcript_search_button"].click(fn=transcript_search_with_backend, inputs=transcript_inputs, outputs=transcript_outputs)

    transcript_clear_outputs = [ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"], ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]]
    ui["transcript_clear_button"].click(fn=handlers.clear_transcript_search, inputs=None, outputs=transcript_clear_outputs, queue=False)

    # === 3. KẾT NỐI SỰ KIỆN HỢP NHẤT CHO TRẠM PHÂN TÍCH (CỘT PHẢI) ===
    analysis_panel_outputs = [
        ui["selected_image_display"], ui["video_player"], ui["full_transcript_display"],
        ui["analysis_display_html"], ui["view_full_video_html"], ui["selected_candidate_for_submission"],
        ui["frame_calculator_video_id"], ui["frame_calculator_time_input"], ui["transcript_selected_index_state"]
    ]
    ui["results_gallery"].select(fn=on_gallery_select_with_backend, inputs=[ui["response_state"], ui["current_page_state"]], outputs=analysis_panel_outputs)
    ui["transcript_results_df"].select(fn=on_transcript_select_with_backend, inputs=[ui["transcript_results_state"]], outputs=analysis_panel_outputs)

    # === 4. KẾT NỐI SỰ KIỆN CHO BẢNG ĐIỀU KHIỂN NỘP BÀI (CỘT PHẢI) ===
    add_visual_inputs = [ui["submission_list_state"], ui["selected_candidate_for_submission"], ui["response_state"]]
    submission_outputs = [ui["submission_list_state"], ui["submission_text_editor"]]
    
    ui["add_top_button"].click(fn=add_to_submission_with_backend, inputs=add_visual_inputs + [gr.Textbox("top", visible=False)], outputs=submission_outputs)
    ui["add_bottom_button"].click(fn=add_to_submission_with_backend, inputs=add_visual_inputs + [gr.Textbox("bottom", visible=False)], outputs=submission_outputs)
    
    add_transcript_inputs = [ui["submission_list_state"], ui["transcript_results_state"], ui["transcript_selected_index_state"]]
    ui["add_transcript_top_button"].click(fn=add_transcript_to_submission_with_backend, inputs=add_transcript_inputs + [gr.Textbox("top", visible=False)], outputs=submission_outputs)
    ui["add_transcript_bottom_button"].click(fn=add_transcript_to_submission_with_backend, inputs=add_transcript_inputs + [gr.Textbox("bottom", visible=False)], outputs=submission_outputs)
    
    ui["refresh_submission_button"].click(fn=sync_editor_with_backend, inputs=[ui["submission_list_state"]], outputs=[ui["submission_text_editor"]], queue=False)
    ui["clear_submission_button"].click(fn=handlers.clear_submission_list, inputs=None, outputs=submission_outputs, queue=False)
    
    # === 5. KẾT NỐI SỰ KIỆN CHO CÁC CÔNG CỤ CÒN LẠI (CỘT PHẢI) ===
    ui["frame_calculator_button"].click(fn=calculate_frame_with_backend, inputs=[ui["frame_calculator_video_id"], ui["frame_calculator_time_input"]], outputs=[ui["frame_calculator_output"]], queue=False)
    ui["submission_button"].click(fn=handlers.handle_submission, inputs=[ui["submission_text_editor"], ui["query_id_input"]], outputs=[ui["submission_file_output"]])
    
    clear_all_outputs = [
        ui["query_input"], ui["results_gallery"], ui["status_output"], ui["response_state"],
        ui["gallery_items_state"], ui["current_page_state"], ui["page_info_display"],
        ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"],
        ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"],
        ui["selected_image_display"], ui["video_player"], ui["full_transcript_display"],
        ui["analysis_display_html"], ui["view_full_video_html"], ui["selected_candidate_for_submission"],
        ui["submission_list_state"], ui["submission_text_editor"],
        ui["frame_calculator_video_id"], ui["frame_calculator_time_input"], ui["frame_calculator_output"],
        ui["query_id_input"], ui["submission_file_output"]
    ]
    ui["clear_button"].click(fn=handlers.clear_all, inputs=None, outputs=clear_all_outputs, queue=False)

# --- Xây dựng UI và các bước còn lại ---
app, ui_components = build_ui(connect_event_listeners)
# Tải video_path_map vào State để các handler có thể truy cập
app.load(lambda: video_path_map, inputs=None, outputs=ui_components["video_path_map_state"])

# --- GIAI ĐOẠN 4: KHỞI CHẠY APP SERVER ---
if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server (Hạm đội Gọng Kìm Kép - Phiên bản Hoàn thiện) ---")
    app.launch(
        share=True,
        allowed_paths=["/kaggle/input/", "/kaggle/working/"],
        debug=True,
        show_error=True
    )