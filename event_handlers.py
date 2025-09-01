from typing import Dict, List
import gradio as gr
import pandas as pd
import numpy as np
import time
import os
import traceback
import json
import re
from io import StringIO

# --- Local imports ---
from config import ITEMS_PER_PAGE, MAX_SUBMISSION_RESULTS, VIDEO_BASE_PATH, TRANSCRIPTS_JSON_DIR
from ui_helpers import create_detailed_info_html, format_submission_list_for_display
from search_core.task_analyzer import TaskType
from utils.formatting import (
    format_list_for_submission, 
    format_results_for_mute_gallery,
    format_submission_list_to_csv_string
)
from utils import create_video_segment, generate_submission_file

# ==============================================================================
# === GỌNG KÌM 1: HANDLERS CHO TAB "MẮT THẦN" (VISUAL SCOUT) ===
# ==============================================================================

def perform_search(
    query_text: str, num_results: int, w_clip: float, w_obj: float, 
    w_semantic: float, lambda_mmr: float,
    master_searcher
):
    """Xử lý sự kiện tìm kiếm chính cho Tab Visual."""
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn.</div>", None, "Trang 1 / 1", [], 1

    loading_html = "<div style='color: #4338ca;'>⏳ Đang quét visual... AI đang phân tích và tìm kiếm.</div>"
    yield ([], loading_html, None, "Trang 1 / 1", [], 1)
    
    try:
        config = {
            "top_k_final": int(num_results), "w_clip": w_clip, "w_obj": w_obj, 
            "w_semantic": w_semantic, "lambda_mmr": lambda_mmr
        }
        start_time = time.time()
        full_response = master_searcher.search(query=query_text, config=config)
        search_time = time.time() - start_time
    except Exception as e:
        traceback.print_exc()
        return [], f"<div style='color: red;'>🔥 Lỗi backend: {e}</div>", None, "Trang 1 / 1", [], 1

    gallery_paths = format_results_for_mute_gallery(full_response)
    num_found = len(gallery_paths)
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    
    if num_found == 0:
        status_msg = f"<div style='color: #d97706;'>😔 **{task_type_msg}** | Không tìm thấy kết quả nào ({search_time:.2f}s).</div>"
    else:
        status_msg = f"<div style='color: #166534;'>✅ **{task_type_msg}** | Tìm thấy {num_found} kết quả ({search_time:.2f}s).</div>"

    initial_gallery_view = gallery_paths[:ITEMS_PER_PAGE]
    total_pages = int(np.ceil(num_found / ITEMS_PER_PAGE)) or 1
    page_info = f"Trang 1 / {total_pages}"
    
    yield (initial_gallery_view, status_msg, full_response, page_info, gallery_paths, 1)

def update_gallery_page(gallery_items: List, current_page: int, direction: str):
    """Cập nhật trang cho gallery visual."""
    if not gallery_items:
        return [], 1, "Trang 1 / 1"
    total_items = len(gallery_items)
    total_pages = int(np.ceil(total_items / ITEMS_PER_PAGE)) or 1
    new_page = min(total_pages, current_page + 1) if direction == "▶️ Trang sau" else max(1, current_page - 1)
    start_index = (new_page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    return gallery_items[start_index:end_index], new_page, f"Trang {new_page} / {total_pages}"

# ==============================================================================
# === GỌNG KÌM 2: HANDLERS CHO TAB "TAI THÍNH" (TRANSCRIPT INTEL) ===
# ==============================================================================

def handle_transcript_search(query1: str, query2: str, query3: str, transcript_searcher):
    """Xử lý sự kiện tìm kiếm lồng nhau trên transcript."""
    gr.Info("Bắt đầu điều tra transcript...")
    results = None
    if query1.strip():
        results = transcript_searcher.search(query1, current_results=results)
    if query2.strip():
        results = transcript_searcher.search(query2, current_results=results)
    if query3.strip():
        results = transcript_searcher.search(query3, current_results=results)

    if results is None:
        return "Nhập truy vấn để bắt đầu điều tra.", pd.DataFrame(), None

    count_str = f"Tìm thấy: {len(results)} kết quả."
    display_df = results[['video_id', 'timestamp', 'transcript_text', 'keyframe_path']]
    return count_str, display_df, results

def clear_transcript_search():
    """Xóa các ô tìm kiếm và kết quả của Tab Tai Thính."""
    return "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(columns=["Video ID", "Timestamp (s)", "Nội dung Lời thoại", "Keyframe Path"]), None, None, "", None

def on_transcript_select(results_state: pd.DataFrame, evt: gr.SelectData):
    """Xử lý khi người dùng chọn một dòng trong bảng kết quả transcript."""
    empty_return = None, "Click vào một dòng kết quả để xem chi tiết.", None
    if evt.value is None or results_state is None or results_state.empty:
        return empty_return
    
    try:
        selected_row = results_state.iloc[evt.index[0]]
        video_id = selected_row['video_id']
        timestamp = selected_row['timestamp']
        keyframe_path = selected_row['keyframe_path']
        video_path = os.path.join(VIDEO_BASE_PATH, f"{video_id}.mp4")
        video_output = gr.Video.update(value=None)
        if os.path.exists(video_path):
            video_output = gr.Video(value=video_path, start_time=timestamp)
        else:
            gr.Warning(f"Không tìm thấy file video: {video_path}")
        full_transcript_text = f"Đang tìm transcript cho video {video_id}..."
        transcript_json_path = os.path.join(TRANSCRIPTS_JSON_DIR, f"{video_id}.json")
        if os.path.exists(transcript_json_path):
            with open(transcript_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                full_transcript_text = data.get("text", "Lỗi: File JSON không chứa key 'text'.").strip()
        else:
            full_transcript_text = f"Lỗi: Không tìm thấy file transcript tại: {transcript_json_path}"
        
        return video_output, full_transcript_text, keyframe_path
    except (IndexError, KeyError) as e:
        gr.Error(f"Lỗi khi xử lý lựa chọn: {e}")
        return None, "Có lỗi xảy ra khi xử lý lựa chọn của bạn.", None

# ==============================================================================
# === HANDLERS DÙNG CHUNG (PHÂN TÍCH, NỘP BÀI, CÔNG CỤ) ===
# ==============================================================================

def on_gallery_select(response_state: Dict, current_page: int, evt: gr.SelectData):
    """Xử lý khi click vào ảnh trong gallery."""
    empty_return = None, None, "", None, "", "0.0", None
    if not response_state or evt is None: return empty_return

    results = response_state.get("results", [])
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    if not results or global_index >= len(results): return empty_return

    selected_result = results[global_index]
    video_path = selected_result.get('video_path')
    timestamp = selected_result.get('timestamp', 0.0)

    video_clip_path = create_video_segment(video_path, timestamp, duration=30)
    analysis_html = create_detailed_info_html(selected_result, response_state.get("task_type"))

    return (selected_result.get('keyframe_path'), video_clip_path, analysis_html,
            selected_result, selected_result.get('video_id'), str(timestamp), video_path)

def get_full_video_path_for_button(video_path):
    """Cung cấp file video để người dùng tải/xem."""
    if video_path and os.path.exists(video_path):
        return video_path
    gr.Warning("Không tìm thấy đường dẫn video gốc.")
    return None

def add_to_submission_list(submission_list: list, candidate: Dict, response_state: Dict, position: str):
    """Thêm ứng viên từ Visual Scout và cập nhật editor."""
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

def add_transcript_result_to_submission(
    submission_list: list, 
    results_state: pd.DataFrame, 
    selected_index: int, # <-- NHẬN VÀO CHỈ SỐ TỪ STATE
    position: str
):
    """
    Thêm kết quả từ transcript vào danh sách, sử dụng chỉ số đã được lưu.
    """
    if selected_index is None or results_state is None or results_state.empty:
        gr.Warning("Vui lòng chọn một kết quả từ bảng transcript trước khi thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list)

    try:
        # Sử dụng chỉ số đã lưu để lấy đúng hàng
        selected_row = results_state.iloc[selected_index]
        candidate = {
            "video_id": selected_row['video_id'], "timestamp": selected_row['timestamp'],
            "keyframe_id": f"transcript_{selected_row['timestamp']:.2f}s", "task_type": TaskType.KIS
        }
        return add_to_submission_list(submission_list, candidate, {"task_type": TaskType.KIS}, position)
    except (IndexError, KeyError) as e:
        gr.Error(f"Lỗi khi thêm kết quả transcript: {e}")
        return submission_list, format_submission_list_to_csv_string(submission_list)

def prepare_submission_for_edit(submission_list: list):
    """Đồng bộ hóa state vào text editor."""
    gr.Info("Đã đồng bộ hóa danh sách vào Bảng điều khiển.")
    return format_submission_list_to_csv_string(submission_list)

def clear_submission_state_and_editor():
    """Xóa state và text editor của submission."""
    gr.Info("Đã xóa danh sách nộp bài và nội dung trong bảng điều khiển.")
    return [], ""

def calculate_frame_number(video_id: str, time_input: str, fps_map: dict):
    """Tính toán frame index từ input giây hoặc phút:giây."""
    if not video_id or not time_input: return "Vui lòng nhập Video ID và Thời gian."
    try:
        time_input_str = str(time_input).strip()
        match = re.match(r'(\d+)\s*:\s*(\d+(\.\d+)?)', time_input_str)
        if match:
            minutes, seconds = int(match.group(1)), float(match.group(2))
            timestamp = minutes * 60 + seconds
            gr.Info(f"Đã chuyển đổi '{time_input_str}' thành {timestamp:.2f} giây.")
        else:
            timestamp = float(time_input_str)
        fps = fps_map.get(video_id, 30.0)
        return str(round(timestamp * fps))
    except (ValueError, TypeError):
        return f"Lỗi: Định dạng thời gian '{time_input}' không hợp lệ."

def handle_submission(submission_csv_text: str, query_id: str):
    """Tạo file CSV nộp bài từ nội dung text editor."""
    if not submission_csv_text or not submission_csv_text.strip():
        gr.Warning("Nội dung nộp bài đang trống.")
        return None
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
    try:
        pd.read_csv(StringIO(submission_csv_text), header=None)
        file_path = generate_submission_file(submission_csv_text, query_id=query_id)
        gr.Success(f"Đã tạo file nộp bài thành công từ nội dung đã sửa!")
        return file_path
    except Exception as e:
        gr.Error(f"Lỗi định dạng CSV: {e}. Vui lòng kiểm tra lại nội dung trong Bảng điều khiển.")
        return None

# ==============================================================================
# === HANDLER DỌN DẸP TOÀN BỘ HỆ THỐNG ===
# ==============================================================================

def clear_all():
    """Reset toàn bộ giao diện về trạng thái ban đầu."""
    return (
        # Tab Mắt Thần
        [], "", None, "Trang 1 / 1", [], 1,
        # Tab Tai Thính
        "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(columns=["Video ID", "Timestamp (s)", "Nội dung Lời thoại", "Keyframe Path"]), None, None, "", None,
        # Trạm Phân tích Visual
        None, None, "", None,
        # Công cụ tính toán
        "", "", "",
        # Bảng điều khiển Nộp bài
        "", [],
        # Vùng Xuất File
        "", None,
    )

def generate_submission_file(submission_data, query_id: str, output_dir: str = "/kaggle/working/submissions") -> str:
    """Ghi dữ liệu (DataFrame hoặc string) ra file CSV."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{query_id}_submission.csv")
    if isinstance(submission_data, str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(submission_data.strip())
    elif isinstance(submission_data, pd.DataFrame):
        submission_data.to_csv(file_path, header=False, index=False)
    else:
        raise TypeError("Dữ liệu nộp bài phải là DataFrame hoặc chuỗi CSV.")
    return file_path