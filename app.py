import gradio as gr
from functools import partial
import pandas as pd

# --- Local imports ---
from backend_loader import initialize_backend
from ui_layout import build_ui
import event_handlers as handlers

def main():
    """
    Hàm chính để khởi tạo backend, xây dựng giao diện,
    kết nối sự kiện và khởi chạy ứng dụng.
    """
    
    print("--- 🚀 Bắt đầu khởi chạy AIC25 Search Fleet ---")
    
    # --- Giai đoạn 1 & 2: Khởi tạo Backend một lần duy nhất ---
    backend_objects = initialize_backend()

    # --- Chuẩn bị các hàm xử lý sự kiện bằng `partial` cho các trường hợp đơn giản ---
    search_with_backend = partial(handlers.perform_search, master_searcher=backend_objects['master_searcher'])
    transcript_search_with_backend = partial(handlers.handle_transcript_search, transcript_searcher=backend_objects['transcript_searcher'])
    calculate_frame_with_backend = partial(handlers.calculate_frame_number, fps_map=backend_objects['fps_map'])

    def connect_event_listeners(ui_components):
        """
        Trung tâm thần kinh: Kết nối TẤT CẢ các sự kiện của UI với logic backend.
        """
        ui = ui_components
        
        # --- Định nghĩa các bộ outputs và states dùng chung ---
        full_video_path_state = gr.State()
        
        unified_analysis_outputs = [
            ui["video_player"], ui["selected_image_display"], ui["full_transcript_display"],
            ui["analysis_display_html"]
        ]
        
        # ==============================================================================
        # === 1. SỰ KIỆN TAB "MẮT THẦN" (VISUAL SCOUT) ===
        # ==============================================================================
        
        visual_search_inputs = [
            ui["query_input"], ui["num_results"], ui["w_clip_slider"], 
            ui["w_obj_slider"], ui["w_semantic_slider"], ui["lambda_mmr_slider"]
        ]
        visual_search_main_outputs = [
            ui["results_gallery"], ui["status_output"], ui["response_state"],
            ui["page_info_display"], ui["gallery_items_state"], ui["current_page_state"]
        ]
        
        ui["search_button"].click(
            fn=handlers.clear_analysis_panel, outputs=unified_analysis_outputs
        ).then(
            fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_main_outputs
        )
        ui["query_input"].submit(
            fn=handlers.clear_analysis_panel, outputs=unified_analysis_outputs
        ).then(
            fn=search_with_backend, inputs=visual_search_inputs, outputs=visual_search_main_outputs
        )

        page_outputs = [ui["results_gallery"], ui["current_page_state"], ui["page_info_display"]]
        ui["prev_page_button"].click(fn=handlers.update_gallery_page, inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("◀️ Trang trước", visible=False)], outputs=page_outputs)
        ui["next_page_button"].click(fn=handlers.update_gallery_page, inputs=[ui["gallery_items_state"], ui["current_page_state"], gr.Textbox("▶️ Trang sau", visible=False)], outputs=page_outputs)
        
        gallery_select_outputs = unified_analysis_outputs + [
            ui["selected_candidate_for_submission"], ui["frame_calculator_video_id"],
            ui["frame_calculator_time_input"], full_video_path_state
        ]
        ui["results_gallery"].select(fn=handlers.on_gallery_select, inputs=[ui["response_state"], ui["current_page_state"]], outputs=gallery_select_outputs)

        # ==============================================================================
        # === 2. SỰ KIỆN TAB "TAI THÍNH" (TRANSCRIPT INTEL) ===
        # ==============================================================================

        transcript_inputs = [ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"]]
        transcript_search_main_outputs = [ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"]]
        
        ui["transcript_search_button"].click(
            fn=handlers.clear_analysis_panel, outputs=unified_analysis_outputs
        ).then(
            fn=transcript_search_with_backend, inputs=transcript_inputs, outputs=transcript_search_main_outputs
        )
        
        transcript_clear_outputs = transcript_search_main_outputs + unified_analysis_outputs + [ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"]]
        ui["transcript_clear_button"].click(fn=handlers.clear_transcript_search, outputs=transcript_clear_outputs)

        # Kết nối sự kiện select một cách trực tiếp và an toàn
        transcript_select_outputs = unified_analysis_outputs + [ui["selected_candidate_for_submission"], ui["frame_calculator_video_id"], ui["frame_calculator_time_input"], full_video_path_state, ui["transcript_selected_index_state"]]
        ui["transcript_results_df"].select(fn=handlers.on_transcript_select, inputs=[ui["transcript_results_state"]], outputs=transcript_select_outputs)
        
        # ==============================================================================
        # === 3. SỰ KIỆN DÙNG CHUNG (CỘT PHẢI) ===
        # ==============================================================================

        ui["view_full_video_button"].click(fn=handlers.get_full_video_path_for_button, inputs=[full_video_path_state], outputs=[ui["submission_file_output"]])
        
        add_outputs = [ui["submission_list_state"], ui["submission_text_editor"]]
        
        add_visual_inputs = [ui["submission_list_state"], ui["selected_candidate_for_submission"], ui["response_state"]]
        ui["add_top_button"].click(fn=partial(handlers.add_to_submission_list, position="top"), inputs=add_visual_inputs, outputs=add_outputs)
        ui["add_bottom_button"].click(fn=partial(handlers.add_to_submission_list, position="bottom"), inputs=add_visual_inputs, outputs=add_outputs)
        
        # Sửa lỗi kết nối: Input phải là state chứa index đã chọn
        transcript_add_inputs = [ui["submission_list_state"], ui["transcript_results_state"], ui["transcript_selected_index_state"]]
        ui["add_transcript_top_button"].click(fn=partial(handlers.add_transcript_result_to_submission, position="top"), inputs=transcript_add_inputs, outputs=add_outputs)
        ui["add_transcript_bottom_button"].click(fn=partial(handlers.add_transcript_result_to_submission, position="bottom"), inputs=transcript_add_inputs, outputs=add_outputs)

        ui["refresh_submission_button"].click(fn=handlers.prepare_submission_for_edit, inputs=[ui["submission_list_state"]], outputs=[ui["submission_text_editor"]])
        ui["clear_submission_button"].click(fn=handlers.clear_submission_state_and_editor, inputs=None, outputs=[ui["submission_list_state"], ui["submission_text_editor"]])
        
        ui["frame_calculator_button"].click(fn=calculate_frame_with_backend, inputs=[ui["frame_calculator_video_id"], ui["frame_calculator_time_input"]], outputs=[ui["frame_calculator_output"]])
        
        ui["submission_button"].click(fn=handlers.handle_submission, inputs=[ui["submission_text_editor"], ui["query_id_input"]], outputs=[ui["submission_file_output"]])
        
        clear_all_outputs = [
            ui["results_gallery"], ui["status_output"], ui["response_state"], ui["page_info_display"], ui["gallery_items_state"], ui["current_page_state"],
            ui["transcript_query_1"], ui["transcript_query_2"], ui["transcript_query_3"], ui["transcript_results_count"], ui["transcript_results_df"], ui["transcript_results_state"],
            *unified_analysis_outputs, ui["submission_text_editor"], ui["submission_list_state"],
            ui["query_id_input"], ui["submission_file_output"]
        ]
        ui["clear_button"].click(fn=handlers.clear_all, inputs=None, outputs=clear_all_outputs, queue=False)

    # === Xây dựng UI và truyền hàm kết nối sự kiện vào ===
    app = build_ui(connect_event_listeners)
    
    # === Khởi chạy App Server ---
    app.launch(share=True, allowed_paths=["/kaggle/input/", "/kaggle/working/"], debug=True)

if __name__ == "__main__":
    main()