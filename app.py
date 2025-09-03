# ==============================================================================
# === AIC25 SEARCH FLEET - TRUNG TÂM CHỈ HUY CHIẾN DỊCH (app.py) ===
# ==============================================================================
print("--- 🚀 Bắt đầu khởi chạy AIC25 Search Fleet ---")

print("--- Giai đoạn 1/4: Đang tải các thư viện cần thiết...")
import gradio as gr
import pandas as pd
from functools import partial

from backend_loader import initialize_backend
from ui_layout import build_ui
import event_handlers as handlers
from config import VIDEO_BASE_PATH, KEYFRAME_BASE_PATH 

print("--- Giai đoạn 2/4: Đang khởi tạo các Động cơ Backend...")
backend_objects = initialize_backend()
master_searcher = backend_objects['master_searcher']
transcript_searcher = backend_objects['transcript_searcher']
fps_map = backend_objects['fps_map']
video_path_map = backend_objects['video_path_map']
print("--- ✅ Toàn bộ Backend đã được nạp và sẵn sàng chiến đấu. ---")


print("--- Giai đoạn 3/4: Đang xây dựng giao diện và kết nối sự kiện...")
search_with_backend = partial(handlers.perform_search, master_searcher=master_searcher)
transcript_search_with_backend = partial(handlers.handle_transcript_search, transcript_searcher=transcript_searcher, fps_map=fps_map)
# on_gallery_select_with_backend = partial(handlers.on_gallery_select, transcript_searcher=transcript_searcher)
def on_transcript_select_wrapper(results_state, query1, query2, query3, evt: gr.SelectData):
    return handlers.on_transcript_select(
        results_state=results_state, video_path_map=video_path_map,
        transcript_searcher=transcript_searcher,
        query1=query1, query2=query2, query3=query3, # <-- Thêm các query
        evt=evt
    )
    
def on_gallery_select_wrapper(response_state, current_page, query_input, evt: gr.SelectData):
    return handlers.on_gallery_select(
        response_state=response_state,
        current_page=current_page,
        query_text=query_input,
        transcript_searcher=transcript_searcher, # Lấy từ scope ngoài
        evt=evt
    )
calculate_frame_with_backend = partial(handlers.calculate_frame_number, fps_map=fps_map)
add_to_submission_with_backend = partial(handlers.add_to_submission_list)
sync_submission_with_backend = partial(handlers.sync_submission_state_to_editor, fps_map=fps_map)

def connect_event_listeners(ui_components):
    """
    Kết nối TẤT CẢ các sự kiện của component UI với các hàm xử lý tương ứng.
    Đây là "bảng mạch" chính của toàn bộ ứng dụng.
    """
    ui = ui_components # Viết tắt cho gọn
    visual_search_inputs = [
        ui["query_input"], ui["num_results"], 
        ui["w_clip_slider"], ui["w_obj_slider"], 
        ui["w_semantic_slider"], ui["lambda_mmr_slider"], ui["initial_retrieval_slider"]
    ]
    visual_search_outputs = [
        ui["results_gallery"], ui["status_output"], ui["response_state"], 
        ui["gallery_items_state"], ui["current_page_state"], ui["page_info_display"]
    ]
    ui["search_button"].click(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)
    ui["query_input"].submit(fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_outputs)
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
    transcript_inputs = [ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"]]
    transcript_outputs = [ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]]
    ui["transcript_search_button"].click(fn=transcript_search_with_backend, inputs=transcript_inputs, outputs=transcript_outputs)

    transcript_clear_outputs = [
        ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"],
        ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]
    ]
    ui["transcript_clear_button"].click(fn=handlers.clear_transcript_search, inputs=None, outputs=transcript_clear_outputs, queue=False)
    analysis_panel_outputs = [
        ui["selected_image_display"], 
        ui["video_player"],
        ui["full_transcript_display"], 
        ui["analysis_display_html"],
        ui["selected_candidate_for_submission"],
        ui["frame_calculator_video_id"], 
        ui["frame_calculator_time_input"],
        ui["transcript_selected_index_state"]
    ]
    ui["results_gallery"].select(
        fn=on_gallery_select_wrapper,
        inputs=[ui["response_state"], ui["current_page_state"], ui["query_input"]],
        outputs=analysis_panel_outputs
    )
    ui["transcript_results_df"].select(
        fn=on_transcript_select_wrapper,
        inputs=[
            ui["transcript_results_state"],
            ui["transcript_query_1"], # <-- Thêm input
            ui["transcript_query_2"], # <-- Thêm input
            ui["transcript_query_3"]  # <-- Thêm input
        ],
        outputs=analysis_panel_outputs,
    )
    ui["add_top_button"].click(
        fn=add_to_submission_with_backend,
        inputs=[ui["submission_list_state"], ui["selected_candidate_for_submission"], gr.Textbox("top", visible=False)],
        outputs=[ui["submission_list_state"], ui["submission_text_editor"]]
    )
    ui["add_bottom_button"].click(
        fn=add_to_submission_with_backend,
        inputs=[ui["submission_list_state"], ui["selected_candidate_for_submission"], gr.Textbox("bottom", visible=False)],
        outputs=[ui["submission_list_state"], ui["submission_text_editor"]]
    )
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
    ui["refresh_submission_button"].click(
        fn=sync_submission_with_backend,
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
    ui["view_full_video_button"].click(
        fn=handlers.handle_view_full_video,
        inputs=[ui["selected_candidate_for_submission"]],
        outputs=[ui["full_video_player"]],
        queue=True # Sử dụng queue để không block UI trong lúc copy
    )
    ui["frame_calculator_button"].click(
        fn=calculate_frame_with_backend,
        inputs=[ui["frame_calculator_video_id"], ui["frame_calculator_time_input"]],
        outputs=[ui["frame_calculator_output"]],
        queue=False
    )
    ui["submission_button"].click(
        fn=handlers.handle_submission,
        inputs=[ui["submission_text_editor"], ui["query_id_input"]],
        outputs=[ui["submission_file_output"]]
    )
    clear_all_outputs = [
        ui["query_input"], ui["results_gallery"], ui["status_output"], ui["response_state"],
        ui["gallery_items_state"], ui["current_page_state"], ui["page_info_display"],
        ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"],
        ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"],
        ui["selected_image_display"], ui["video_player"], ui["full_transcript_display"],
        ui["analysis_display_html"], ui["selected_candidate_for_submission"],
        ui["submission_list_state"], ui["submission_text_editor"],
        ui["frame_calculator_video_id"], ui["frame_calculator_time_input"], ui["frame_calculator_output"],
        ui["query_id_input"], ui["submission_file_output"]
    ]
    ui["clear_button"].click(fn=handlers.clear_all, inputs=None, outputs=clear_all_outputs, queue=False)


app, ui_components = build_ui(connect_event_listeners)

if __name__ == "__main__":
    print("--- 🚀 Khởi chạy Gradio App Server (Hạm đội Gọng Kìm Kép - Phiên bản Hoàn thiện) ---")
    final_allowed_paths = [VIDEO_BASE_PATH, KEYFRAME_BASE_PATH, "/kaggle/working/"]
    print(f"--- 🔑 Cấp phép truy cập cho các đường dẫn: {final_allowed_paths} ---")

    app.launch(
        share=True,
        allowed_paths=final_allowed_paths, # <-- Sử dụng danh sách đã hoàn thiện
        debug=True,
        show_error=True
    )
