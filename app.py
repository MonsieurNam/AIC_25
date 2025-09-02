import gradio as gr
from functools import partial
import pandas as pd

# --- Local imports ---
from backend_loader import initialize_backend
from ui_layout import build_ui
import event_handlers as handlers

print("--- 🚀 Bắt đầu khởi chạy AIC25 Search Fleet ---")

# --- Giai đoạn 1 & 2: Khởi tạo Backend và các Động cơ ---
# Hàm này sẽ tải tất cả model và dữ liệu cần thiết một lần duy nhất.
print("--- Giai đoạn 1/4 & 2/4: Đang tải thư viện và khởi tạo Backend... ---")
backend_objects = initialize_backend()

# --- Giai đoạn 3: Xây dựng Giao diện & Kết nối Logic ---
print("--- Giai đoạn 3/4: Đang xây dựng giao diện và kết nối sự kiện... ---")

# --- SỬ DỤNG `partial` ĐỂ "TIÊM" BACKEND VÀO CÁC HÀM XỬ LÝ SỰ KIỆN ---
# Kỹ thuật này giúp giữ cho các kết nối sự kiện (clicks) cực kỳ sạch sẽ.
search_with_backend = partial(handlers.perform_search, master_searcher=backend_objects['master_searcher'])
transcript_search_with_backend = partial(handlers.handle_transcript_search, transcript_searcher=backend_objects['transcript_searcher'])
calculate_frame_with_backend = partial(handlers.calculate_frame_number, fps_map=backend_objects['fps_map'])
add_transcript_to_submission_with_backend = partial(handlers.add_transcript_result_to_submission)
on_transcript_select_with_backend = partial(
    handlers.on_transcript_select, 
    video_path_map=backend_objects['video_path_map']
)

def connect_event_listeners(ui_components):
    """
    Kết nối TẤT CẢ các sự kiện của component UI với các hàm xử lý tương ứng.
    Đây là trung tâm thần kinh của toàn bộ ứng dụng.
    """
    ui = ui_components # Đổi tên cho ngắn gọn
    
    # === 1. SỰ KIỆN TAB "MẮT THẦN" (VISUAL SCOUT) ===
    # 1.1. Nút Tìm kiếm chính
    visual_search_inputs = [ui["query_input"], ui["num_results"], ui["w_clip_slider"], ui["w_obj_slider"], ui["w_semantic_slider"], ui["lambda_mmr_slider"]]
    visual_search_outputs = [ui["results_gallery"], ui["status_output"], ui["response_state"], ui["page_info_display"], ui["gallery_items_state"], ui["current_page_state"]]
    ui["search_button"].click(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)
    ui["query_input"].submit(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)
    
    # 1.2. Phân trang
    page_outputs = [ui["results_gallery"], ui["current_page_state"], ui["page_info_display"]]
    ui["prev_page_button"].click(fn=handlers.update_gallery_page, inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("◀️ Trang trước", visible=False)], outputs=page_outputs)
    ui["next_page_button"].click(fn=handlers.update_gallery_page, inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("▶️ Trang sau", visible=False)], outputs=page_outputs)

    # 1.3. Chọn một ảnh trong Gallery -> Kích hoạt Trạm Phân tích Visual
    full_video_path_state = gr.State()
    analysis_outputs = [
        ui["selected_image_display"], ui["video_player"], ui["analysis_display_html"],
        ui["selected_candidate_for_submission"], ui["frame_calculator_video_id"], 
        ui["frame_calculator_time_input"], full_video_path_state
    ]
    ui["results_gallery"].select(fn=handlers.on_gallery_select, inputs=[ui["response_state"], ui["current_page_state"]], outputs=analysis_outputs)
    
    # 1.4. Nút Mở Video Gốc
    ui["view_full_video_button"].click(fn=handlers.get_full_video_path_for_button, inputs=[full_video_path_state], outputs=[ui["submission_file_output"]])


    # === 2. SỰ KIỆN TAB "TAI THÍNH" (TRANSCRIPT INTEL) ===
    # 2.1. Nút Tìm kiếm Transcript
    transcript_inputs = [ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"]]
    transcript_outputs = [ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]]
    ui["transcript_search_button"].click(fn=transcript_search_with_backend, inputs=transcript_inputs, outputs=transcript_outputs)

    # 2.2. Nút Xóa bộ lọc Transcript
    transcript_clear_outputs = [
        ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"],
        ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"],
        ui["transcript_video_player"], ui["full_transcript_display"], ui["transcript_keyframe_display"]
    ]
    ui["transcript_clear_button"].click(fn=handlers.clear_transcript_search, inputs=None, outputs=transcript_clear_outputs)

    # 2.3. Chọn một dòng kết quả -> Phát video và hiển thị full transcript
    transcript_select_outputs = [
        ui["transcript_video_player"], 
        ui["full_transcript_display"],
        ui["transcript_keyframe_display"], 
        ui["transcript_selected_index_state"] 
    ]
    ui["transcript_results_df"].select(
        fn=on_transcript_select_with_backend, 
        inputs=[ui["transcript_results_state"]], 
        outputs=transcript_select_outputs
    )

    # 2.4. Thêm kết quả vào danh sách nộp bài (SỬA LẠI INPUTS)
    transcript_add_inputs = [
        ui["submission_list_state"], 
        ui["transcript_results_state"], 
        ui["transcript_selected_index_state"] # <-- Input mới: đọc chỉ số từ state
    ]
    transcript_add_outputs = [ui["submission_list_state"], ui["submission_text_editor"]]
    ui["add_transcript_top_button"].click(fn=add_transcript_to_submission_with_backend, inputs=transcript_add_inputs + [gr.Textbox("top", visible=False)], outputs=transcript_add_outputs)
    ui["add_transcript_bottom_button"].click(fn=add_transcript_to_submission_with_backend, inputs=transcript_add_inputs + [gr.Textbox("bottom", visible=False)], outputs=transcript_add_outputs)

    # === 3. SỰ KIỆN DÙNG CHUNG (CỘT PHẢI) ===
    # 3.1. Thêm kết quả từ Visual vào danh sách nộp bài
    add_visual_inputs = [ui["submission_list_state"], ui["selected_candidate_for_submission"], ui["response_state"]]
    add_visual_outputs = [ui["submission_list_state"], ui["submission_text_editor"]]
    ui["add_top_button"].click(fn=handlers.add_to_submission_list, inputs=add_visual_inputs + [gr.Textbox("top", visible=False)], outputs=add_visual_outputs)
    ui["add_bottom_button"].click(fn=handlers.add_to_submission_list, inputs=add_visual_inputs + [gr.Textbox("bottom", visible=False)], outputs=add_visual_outputs)

    # 3.2. Bảng điều khiển Nộp bài
    ui["refresh_submission_button"].click(fn=handlers.prepare_submission_for_edit, inputs=[ui["submission_list_state"]], outputs=[ui["submission_text_editor"]])
    ui["clear_submission_button"].click(fn=handlers.clear_submission_state_and_editor, inputs=None, outputs=[ui["submission_list_state"], ui["submission_text_editor"]])

    # 3.3. Máy tính Thời gian & Frame
    calc_inputs = [ui["frame_calculator_video_id"], ui["frame_calculator_time_input"]]
    ui["frame_calculator_button"].click(fn=calculate_frame_with_backend, inputs=calc_inputs, outputs=[ui["frame_calculator_output"]])

    # 3.4. Nút Xuất File CSV
    ui["submission_button"].click(fn=handlers.handle_submission, inputs=[ui["submission_text_editor"], ui["query_id_input"]], outputs=[ui["submission_file_output"]])

    # 3.5. Nút Xóa Tất cả (Toàn bộ hệ thống)
    clear_all_outputs = [
        # Tab Mắt Thần
        ui["results_gallery"], ui["status_output"], ui["response_state"], ui["page_info_display"], 
        ui["gallery_items_state"], ui["current_page_state"],
        # Tab Tai Thính
        ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"],
        ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_video_player"],
        ui["transcript_results_state"], ui["full_transcript_display"], ui["transcript_keyframe_display"],
        # Cột Phải - Trạm Phân tích Visual
        ui["selected_image_display"], ui["video_player"], ui["analysis_display_html"],
        ui["selected_candidate_for_submission"],
        # Cột Phải - Công cụ tính toán
        ui["frame_calculator_video_id"], ui["frame_calculator_time_input"], ui["frame_calculator_output"],
        # Cột Phải - Bảng điều khiển Nộp bài
        ui["submission_text_editor"], ui["submission_list_state"],
        # Cột Phải - Vùng Xuất File
        ui["query_id_input"], ui["submission_file_output"]
    ]
    ui["clear_button"].click(fn=handlers.clear_all, inputs=None, outputs=clear_all_outputs)


# === Xây dựng UI và truyền hàm kết nối sự kiện vào ===
app = build_ui(connect_event_listeners)

# --- Giai đoạn 4: Khởi chạy App Server ---
if __name__ == "__main__":
    print("--- ✅ Hạm đội đã sẵn sàng! Khởi chạy Gradio App Server... ---")
    app.launch(share=True, allowed_paths=["/kaggle/input/", "/kaggle/working/"], debug=True)