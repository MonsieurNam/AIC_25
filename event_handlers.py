import gradio as gr
import pandas as pd
import numpy as np
import time
import os
import traceback
import json
import re
from io import StringIO
from typing import Dict, List

from config import ITEMS_PER_PAGE, MAX_SUBMISSION_RESULTS, VIDEO_BASE_PATH, TRANSCRIPTS_JSON_DIR
from ui_helpers import create_detailed_info_html
from search_core.task_analyzer import TaskType
from utils.formatting import format_list_for_submission, format_results_for_mute_gallery, format_submission_list_to_csv_string
from utils import create_video_segment, generate_submission_file

def perform_search(query_text: str, num_results: int, w_clip: float, w_obj: float, w_semantic: float, lambda_mmr: float, master_searcher):
    analysis_clear_outputs = (None, None, "", "")
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return ([], "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn.</div>", None, "Trang 1 / 1", [], 1, *analysis_clear_outputs)
    try:
        config = {"top_k_final": int(num_results), "w_clip": w_clip, "w_obj": w_obj, "w_semantic": w_semantic, "lambda_mmr": lambda_mmr}
        start_time = time.time()
        full_response = master_searcher.search(query=query_text, config=config)
        search_time = time.time() - start_time
    except Exception as e:
        traceback.print_exc()
        return ([], f"<div style='color: red;'>🔥 Lỗi backend: {e}</div>", None, "Trang 1 / 1", [], 1, *analysis_clear_outputs)
    gallery_paths = format_results_for_mute_gallery(full_response)
    num_found = len(gallery_paths)
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    status_msg = f"<div style='color: {'#166534' if num_found > 0 else '#d97706'};'>{'✅' if num_found > 0 else '😔'} **{task_type_msg}** | Tìm thấy {num_found} kết quả ({search_time:.2f}s).</div>"
    initial_gallery_view = gallery_paths[:ITEMS_PER_PAGE]
    total_pages = int(np.ceil(num_found / ITEMS_PER_PAGE)) or 1
    page_info = f"Trang 1 / {total_pages}"
    return initial_gallery_view, status_msg, full_response, page_info, gallery_paths, 1

def update_gallery_page(gallery_items: list, current_page: int, direction: str):
    if not gallery_items: return [], 1, "Trang 1 / 1"
    total_items = len(gallery_items)
    total_pages = int(np.ceil(total_items / ITEMS_PER_PAGE)) or 1
    new_page = min(total_pages, current_page + 1) if direction == "▶️ Trang sau" else max(1, current_page - 1)
    start_index = (new_page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    return gallery_items[start_index:end_index], new_page, f"Trang {new_page} / {total_pages}"

def clear_analysis_panel():
    """Trả về các giá trị rỗng để dọn dẹp Trạm Phân tích Hợp nhất."""
    return (
        None,   # video_player
        None,   # selected_image_display
        "",     # full_transcript_display
        "",     # analysis_display_html
        None,   # selected_candidate_for_submission
        "",     # frame_calculator_video_id
        "0.0",  # frame_calculator_time_input
        None    # full_video_path_state
    )

def handle_transcript_search(query1: str, query2: str, query3: str, transcript_searcher):
    """
    Chỉ thực hiện tìm kiếm và trả về kết quả. Không còn `yield`.
    """
    gr.Info("Bắt đầu điều tra transcript...") # Vẫn cung cấp phản hồi cho người dùng
    
    results = None
    if query1.strip(): results = transcript_searcher.search(query1, current_results=results)
    if query2.strip(): results = transcript_searcher.search(query2, current_results=results)
    if query3.strip(): results = transcript_searcher.search(query3, current_results=results)

    if results is None:
        return "Nhập truy vấn để bắt đầu điều tra.", pd.DataFrame(), None

    count_str = f"Tìm thấy: {len(results)} kết quả."
    display_df = results[['video_id', 'timestamp', 'transcript_text', 'keyframe_path']]
    
    # Trả về 3 giá trị cho các output của nó
    return count_str, display_df, results

def clear_transcript_search():
    """Xóa các ô tìm kiếm và kết quả của Tab Tai Thính."""
    # 6 outputs cho Tab Tai Thính + 8 outputs cho Trạm Phân tích
    analysis_clear_vals = clear_analysis_panel()
    return (
        "", "", "", # query 1, 2, 3
        "Tìm thấy: 0 kết quả.", 
        pd.DataFrame(columns=["Video ID", "Timestamp (s)", "Nội dung Lời thoại", "Keyframe Path"]), 
        None, # transcript_results_state
        *analysis_clear_vals
    )

def on_gallery_select(response_state: dict, current_page: int, evt: gr.SelectData):
    empty_return = (None, None, "", "", None, "", "0.0", None)
    if not response_state or evt is None: return empty_return
    results = response_state.get("results", [])
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    if not results or global_index >= len(results): return empty_return
    selected_result = results[global_index]
    video_path, timestamp, keyframe_path, video_id = selected_result.get('video_path'), selected_result.get('timestamp', 0.0), selected_result.get('keyframe_path'), selected_result.get('video_id')
    video_clip_path = create_video_segment(video_path, timestamp, duration=30)
    analysis_html = create_detailed_info_html(selected_result, response_state.get("task_type"))
    return (
        video_clip_path,                    # video_player
        keyframe_path,                      # selected_image_display
        "Transcript chỉ hiển thị khi chọn từ Tab 'Tai Thính'.", # full_transcript_display
        analysis_html,                      # analysis_display_html
        selected_result,                    # selected_candidate_for_submission
        video_id,                           # frame_calculator_video_id
        str(timestamp),                     # frame_calculator_time_input
        video_path                          # full_video_path_state
    )

def on_transcript_select(results_state: pd.DataFrame, evt: gr.SelectData, video_path_map: dict):
    """
    Xử lý khi chọn dòng transcript.
    Trả về ĐÚNG SỐ LƯỢNG outputs mà app.py cần.
    """
    empty_return = (
        None, None, "Click vào một dòng kết quả...", "",
        None, "", "0.0", None,
        None
    )
    
    if not isinstance(evt, gr.SelectData) or results_state is None or results_state.empty:
        return empty_return
    try:
        selected_index = evt.index[0]
        selected_row = results_state.iloc[selected_index]
        video_id, timestamp, keyframe_path = selected_row['video_id'], selected_row['timestamp'], selected_row['keyframe_path']
        video_path = video_path_map.get(video_id)
        video_clip_path = create_video_segment(video_path, timestamp, duration=30) if video_path and os.path.exists(video_path) else None
        transcript_json_path = os.path.join(TRANSCRIPTS_JSON_DIR, f"{video_id}.json")
        full_transcript_text = ""
        if os.path.exists(transcript_json_path):
            with open(transcript_json_path, 'r', encoding='utf-8') as f: full_transcript_text = json.load(f).get("text", "").strip()
        candidate = {"video_id": video_id, "timestamp": timestamp, "keyframe_path": keyframe_path, "keyframe_id": f"transcript_{timestamp:.2f}s"}
        return (
            video_clip_path,                    # video_player
            keyframe_path,                      # selected_image_display
            full_transcript_text,               # full_transcript_display
            "",                                 # analysis_display_html (trống)
            candidate,                          # selected_candidate_for_submission
            video_id,                           # frame_calculator_video_id
            str(timestamp),                     # frame_calculator_time_input
            video_path,                         # full_video_path_state
            selected_index                      # transcript_selected_index_state
        )    
    except (IndexError, KeyError, AttributeError) as e: # Bắt thêm AttributeError
        gr.Error(f"Lỗi khi xử lý lựa chọn: {e}")
        return empty_return

def get_full_video_path_for_button(video_path):
    """Cung cấp file video để người dùng tải/xem."""
    if video_path and os.path.exists(video_path):
        # Trả về đường dẫn để component gr.File có thể xử lý
        return video_path
    gr.Warning("Không tìm thấy đường dẫn video gốc để mở.")
    return None

def add_to_submission_list(submission_list: list, candidate: dict, response_state: dict, position: str):
    if not candidate:
        gr.Warning("Chưa có ứng viên nào được chọn để thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list)
    task_type = response_state.get("task_type", TaskType.KIS)
    item_to_add = {**candidate, 'task_type': task_type}
    if len(submission_list) < MAX_SUBMISSION_RESULTS:
        if position == 'top': submission_list.insert(0, item_to_add)
        else: submission_list.append(item_to_add)
        gr.Success(f"Đã thêm kết quả vào {'đầu' if position == 'top' else 'cuối'} danh sách!")
    else:
        gr.Warning(f"Danh sách đã đạt giới hạn {MAX_SUBMISSION_RESULTS} kết quả.")
    return submission_list, format_submission_list_to_csv_string(submission_list)

def add_transcript_result_to_submission(submission_list: list, results_state: pd.DataFrame, selected_index: int, position: str):
    if selected_index is None or results_state is None or results_state.empty:
        gr.Warning("Vui lòng chọn một kết quả từ bảng transcript trước khi thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list)
    try:
        selected_row = results_state.iloc[selected_index]
        candidate = {"video_id": selected_row['video_id'], "timestamp": selected_row['timestamp'], "keyframe_id": f"transcript_{selected_row['timestamp']:.2f}s"}
        return add_to_submission_list(submission_list, candidate, {"task_type": TaskType.KIS}, position)
    except Exception as e:
        gr.Error(f"Lỗi khi thêm kết quả transcript: {e}")
        return submission_list, format_submission_list_to_csv_string(submission_list)

def prepare_submission_for_edit(submission_list: list):
    gr.Info("Đã đồng bộ hóa danh sách vào Bảng điều khiển.")
    return format_submission_list_to_csv_string(submission_list)

def clear_submission_state_and_editor():
    gr.Info("Đã xóa danh sách nộp bài và nội dung trong bảng điều khiển.")
    return [], ""

def calculate_frame_number(video_id: str, time_input: str, fps_map: dict):
    if not video_id or not time_input: return "Vui lòng nhập Video ID và Thời gian."
    try:
        time_input_str = str(time_input).strip()
        match = re.match(r'(\d+)\s*:\s*(\d+(\.\d+)?)', time_input_str)
        if match:
            minutes, seconds = int(match.group(1)), float(match.group(2))
            timestamp = minutes * 60 + seconds
        else:
            timestamp = float(time_input_str)
        fps = fps_map.get(video_id, 30.0)
        return str(round(timestamp * fps))
    except Exception:
        return f"Lỗi: Định dạng thời gian '{time_input}' không hợp lệ."

def handle_submission(submission_csv_text: str, query_id: str):
    if not submission_csv_text or not submission_csv_text.strip():
        gr.Warning("Nội dung nộp bài đang trống.")
        return None
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID.")
        return None
    try:
        pd.read_csv(StringIO(submission_csv_text), header=None)
        file_path = generate_submission_file(submission_csv_text, query_id=query_id)
        gr.Success(f"Đã tạo file nộp bài thành công!")
        return file_path
    except Exception as e:
        gr.Error(f"Lỗi định dạng CSV: {e}.")
        return None

def clear_all():
    """Reset toàn bộ giao diện về trạng thái ban đầu. PHIÊN BẢN ĐỒNG BỘ CUỐI CÙNG."""
    analysis_clear_vals = clear_analysis_panel()
    transcript_clear_main_vals = ("", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None)
    submission_clear_vals = ("", [])
    file_clear_vals = ("", None)
    
    return (
        # Mắt Thần
        [], "", None, "Trang 1 / 1", [], 1,
        # Tai Thính
        *transcript_clear_main_vals,
        # Trạm Phân tích
        *analysis_clear_vals,
        # Bảng điều khiển
        *submission_clear_vals,
        # Xuất File
        *file_clear_vals
    )