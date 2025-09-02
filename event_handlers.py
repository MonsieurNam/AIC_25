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
import glob

# --- Local imports ---
from config import ITEMS_PER_PAGE, MAX_SUBMISSION_RESULTS, VIDEO_BASE_PATH, TRANSCRIPTS_JSON_DIR
from ui_helpers import create_detailed_info_html
from search_core.task_analyzer import TaskType
from utils.formatting import (
    format_list_for_submission, 
    format_results_for_mute_gallery,
    format_submission_list_to_csv_string
)
from utils import create_video_segment

# ==============================================================================
# === BỘ NHỚ CACHE CỤC BỘ "TỰ LỰC CÁNH SINH" ===
# ==============================================================================
print("--- [event_handlers.py] Đang xây dựng Bộ nhớ đệm cho Đường dẫn Video... ---")
try:
    _all_video_files = glob.glob(os.path.join(VIDEO_BASE_PATH, "**", "*.mp4"), recursive=True)
    VIDEO_PATH_MAP_CACHE = {os.path.basename(f).replace('.mp4', ''): f for f in _all_video_files}
    print(f"--- [event_handlers.py] Đã cache {len(VIDEO_PATH_MAP_CACHE)} đường dẫn video. ---")
except Exception as e:
    print(f"--- [event_handlers.py] LỖI khi xây dựng cache đường dẫn video: {e} ---")
    VIDEO_PATH_MAP_CACHE = {}

# ==============================================================================
# === CÁC HÀM XỬ LÝ SỰ KIỆN ===
# ==============================================================================

def clear_analysis_panel():
    """Trả về các giá trị rỗng để dọn dẹp Trạm Phân tích Hợp nhất."""
    return None, None, "", ""

def perform_search(query_text: str, num_results: int, w_clip: float, w_obj: float, w_semantic: float, lambda_mmr: float, master_searcher):
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn.</div>", None, "Trang 1 / 1", [], 1
    gr.Info("Bắt đầu quét visual...")
    try:
        config = {"top_k_final": int(num_results), "w_clip": w_clip, "w_obj": w_obj, "w_semantic": w_semantic, "lambda_mmr": lambda_mmr}
        start_time = time.time()
        full_response = master_searcher.search(query=query_text, config=config)
        search_time = time.time() - start_time
    except Exception as e:
        traceback.print_exc()
        return [], f"<div style='color: red;'>🔥 Lỗi backend: {e}</div>", None, "Trang 1 / 1", [], 1
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

def handle_transcript_search(query1: str, query2: str, query3: str, transcript_searcher):
    gr.Info("Bắt đầu điều tra transcript...")
    results = None
    if query1.strip(): results = transcript_searcher.search(query1, current_results=results)
    if query2.strip(): results = transcript_searcher.search(query2, current_results=results)
    if query3.strip(): results = transcript_searcher.search(query3, current_results=results)
    if results is None:
        return "Nhập truy vấn để bắt đầu điều tra.", pd.DataFrame(), None
    count_str = f"Tìm thấy: {len(results)} kết quả."
    display_df = results[['video_id', 'timestamp', 'transcript_text', 'keyframe_path']]
    return count_str, display_df, results

def clear_transcript_search():
    analysis_clear_vals = clear_analysis_panel()
    return ("", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None, *analysis_clear_vals)

def on_gallery_select(response_state: dict, current_page: int, evt: gr.SelectData):
    empty_return = (None, None, "", "", None, "", "0.0", None)
    if not isinstance(evt, gr.SelectData) or not response_state: return empty_return
    results = response_state.get("results", [])
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    if not results or global_index >= len(results): return empty_return
    selected_result = results[global_index]
    video_path, timestamp, keyframe_path, video_id = selected_result.get('video_path'), selected_result.get('timestamp', 0.0), selected_result.get('keyframe_path'), selected_result.get('video_id')
    video_clip_path = create_video_segment(video_path, timestamp, duration=30)
    analysis_html = create_detailed_info_html(selected_result, response_state.get("task_type"))
    return video_clip_path, keyframe_path, "Transcript chỉ hiển thị khi chọn từ Tab 'Tai Thính'.", analysis_html, selected_result, video_id, str(timestamp), video_path

def on_transcript_select(results_state: pd.DataFrame, evt: gr.SelectData):
    empty_return = (None, None, "Click vào một dòng kết quả...", "", None, "", "0.0", None, None)
    if not isinstance(evt, gr.SelectData) or results_state is None or results_state.empty: return empty_return
    try:
        selected_index = evt.index[0]
        selected_row = results_state.iloc[selected_index]
        video_id, timestamp, keyframe_path = selected_row['video_id'], selected_row['timestamp'], selected_row['keyframe_path']
        video_path = VIDEO_PATH_MAP_CACHE.get(video_id)
        video_clip_path = create_video_segment(video_path, timestamp, duration=30) if video_path and os.path.exists(video_path) else None
        transcript_json_path = os.path.join(TRANSCRIPTS_JSON_DIR, f"{video_id}.json")
        full_transcript_text = f"Thông báo: Không có file transcript cho video '{video_id}'."
        if os.path.exists(transcript_json_path):
            with open(transcript_json_path, 'r', encoding='utf-8') as f: full_transcript_text = json.load(f).get("text", "").strip()
        candidate = {"video_id": video_id, "timestamp": timestamp, "keyframe_id": f"transcript_{timestamp:.2f}s"}
        return video_clip_path, keyframe_path, full_transcript_text, "", candidate, video_id, str(timestamp), video_path, selected_index
    except Exception as e:
        gr.Error(f"Lỗi khi xử lý lựa chọn transcript: {e}")
        traceback.print_exc()
        return empty_return

def get_full_video_path_for_button(video_path):
    if video_path and os.path.exists(video_path): return video_path
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

def add_transcript_result_to_submission(submission_list: list, results_state: pd.DataFrame, evt: gr.SelectData, position: str):
    if not isinstance(evt, gr.SelectData) or results_state is None or results_state.empty:
        gr.Warning("Vui lòng chọn một kết quả từ bảng transcript trước khi thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list)
    try:
        selected_index = evt.index[0]
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
        # Check if text is valid CSV before writing
        pd.read_csv(StringIO(submission_csv_text), header=None)
        
        output_dir = "/kaggle/working/submissions"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{query_id}_submission.csv")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(submission_csv_text.strip())
        
        gr.Success(f"Đã tạo file nộp bài thành công!")
        return file_path
    except Exception as e:
        gr.Error(f"Lỗi định dạng CSV: {e}.")
        return None

def clear_all():
    analysis_clear_vals = clear_analysis_panel()
    transcript_clear_main_vals = ("", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None)
    submission_clear_vals = ("", [])
    file_clear_vals = ("", None)
    return (
        [], "", None, "Trang 1 / 1", [], 1, # Mắt Thần
        *transcript_clear_main_vals,       # Tai Thính
        *analysis_clear_vals,              # Trạm Phân tích
        *submission_clear_vals,            # Bảng điều khiển
        *file_clear_vals                   # Xuất File
    )