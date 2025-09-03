import traceback
import gradio as gr
import pandas as pd
import numpy as np
import time
import os
import re
from typing import Dict, Any, List, Optional

# Local imports
from config import ITEMS_PER_PAGE, MAX_SUBMISSION_RESULTS, VIDEO_BASE_PATH
from ui_helpers import create_detailed_info_html
from search_core.task_analyzer import TaskType
from utils import create_video_segment, generate_submission_file

# ==============================================================================
# === 1. CÁC HÀM TRỢ GIÚP (HELPERS) ===
# ==============================================================================

def generate_full_video_link(video_path: str) -> str:
    """Tạo link HTML để mở video gốc trong tab mới."""
    if not video_path or not os.path.exists(video_path):
        return "<p style='color: #888; text-align: center; padding: 10px;'>Chọn một kết quả để xem link video gốc.</p>"
    file_url = f"/file={video_path}"
    return f"""<div style='text-align: center; margin-top: 10px;'><a href='{file_url}' target='_blank' style='background-color: #4CAF50; color: white; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; border-radius: 8px; font-weight: bold; cursor: pointer;'>🎬 Mở Video Gốc (Toàn bộ) trong Tab mới</a></div>"""

def get_full_transcript_for_video(video_id: str, transcript_searcher) -> str:
    """Trích xuất toàn bộ transcript của một video."""
    if not transcript_searcher or transcript_searcher.full_data is None:
        return "Lỗi: Transcript engine chưa sẵn sàng."
    try:
        video_transcripts = transcript_searcher.full_data[transcript_searcher.full_data['video_id'] == video_id]
        if video_transcripts.empty:
            return "Video này không có lời thoại."
        full_text = " ".join(video_transcripts['transcript_text'].tolist())
        return full_text.strip() if full_text.strip() else "Video này không có lời thoại."
    except Exception:
        return "Không thể tải transcript cho video này."

def parse_time_string(time_input: Any) -> Optional[float]:
    """Chuyển đổi chuỗi thời gian (mm:ss.ms hoặc ss.ms) thành giây."""
    if time_input is None: return None
    time_str = str(time_input).strip()
    if ':' in time_str:
        parts = time_str.split(':')
        try:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except (ValueError, IndexError): return None
    try:
        return float(time_str)
    except ValueError: return None

# ==============================================================================
# === 2. HANDLERS CHO TAB "MẮT THẦN" (VISUAL SCOUT) ===
# ==============================================================================

def perform_search(query_text: str, num_results: int, w_clip: float, w_obj: float, w_semantic: float, lambda_mmr: float, master_searcher):
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn.</div>", None, [], 1, "Trang 1 / 1"

    loading_html = "<div style='color: #4338ca;'>⏳ Đang quét visual... AI đang phân tích và tìm kiếm.</div>"
    yield ([], loading_html, None, [], 1, "Trang 1 / 1")
    
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
        return [], f"<div style='color: red;'>🔥 Lỗi backend: {e}</div>", None, [], 1, "Trang 1 / 1"

    gallery_paths = [item['keyframe_path'] for item in full_response.get("results", [])]
    num_found = len(gallery_paths)
    status_msg = f"<div style='color: {'#166534' if num_found > 0 else '#d97706'};'>{'✅' if num_found > 0 else '😔'} Tìm thấy {num_found} kết quả ({search_time:.2f}s).</div>"
    
    initial_gallery_view = gallery_paths[:ITEMS_PER_PAGE]
    total_pages = int(np.ceil(num_found / ITEMS_PER_PAGE)) or 1
    
    yield (
        initial_gallery_view, status_msg, full_response,
        gallery_paths, 1, f"Trang 1 / {total_pages}"
    )

def update_gallery_page(gallery_items: list, current_page: int, direction: str):
    if not gallery_items: return [], 1, "Trang 1 / 1"
    total_items = len(gallery_items)
    total_pages = int(np.ceil(total_items / ITEMS_PER_PAGE)) or 1
    new_page = min(total_pages, current_page + 1) if direction == "▶️ Trang sau" else max(1, current_page - 1)
    start_index = (new_page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    return gallery_items[start_index:end_index], new_page, f"Trang {new_page} / {total_pages}"

# ==============================================================================
# === 3. HANDLERS CHO TAB "TAI THÍNH" (TRANSCRIPT INTEL) ===
# ==============================================================================

def handle_transcript_search(query1: str, query2: str, query3: str, transcript_searcher):
    gr.Info("Bắt đầu điều tra transcript...")
    results = None
    if query1.strip(): results = transcript_searcher.search(query1, current_results=results)
    if query2.strip(): results = transcript_searcher.search(query2, current_results=results)
    if query3.strip(): results = transcript_searcher.search(query3, current_results=results)

    if results is None: return "Nhập truy vấn để bắt đầu điều tra.", pd.DataFrame(), None

    count_str = f"Tìm thấy: {len(results)} kết quả."
    display_df = results[['video_id', 'timestamp', 'transcript_text', 'keyframe_path']]
    return count_str, display_df, results

def clear_transcript_search():
    return "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None

# ==============================================================================
# === 4. HANDLERS HỢP NHẤT CHO TRẠM PHÂN TÍCH (CỘT PHẢI) ===
# ==============================================================================

def on_gallery_select(response_state: Dict, current_page: int, transcript_searcher, evt: gr.SelectData):
    empty_return = (None, None, "", "", "", None, "", "", None)
    if not response_state or evt is None: return empty_return
    results = response_state.get("results", [])
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    if not results or global_index >= len(results): return empty_return
    
    selected_result = results[global_index]
    video_id = selected_result.get('video_id')
    video_path = selected_result.get('video_path')
    keyframe_path = selected_result.get('keyframe_path')
    timestamp = selected_result.get('timestamp', 0.0)
    
    full_transcript = get_full_transcript_for_video(video_id, transcript_searcher)
    video_clip_path = create_video_segment(video_path, timestamp, duration=30)
    analysis_html = create_detailed_info_html(selected_result, response_state.get("task_type"))
    full_video_link_html = generate_full_video_link(video_path)

    return (
        keyframe_path, gr.Video(value=video_clip_path, label=f"Clip 30s từ @ {timestamp:.2f}s"),
        full_transcript, analysis_html, full_video_link_html,
        selected_result, video_id, f"{timestamp:.2f}", None
    )

def on_transcript_select(results_state: pd.DataFrame, video_path_map: dict, transcript_searcher, evt: gr.SelectData):
    empty_return = (None, None, "", "", "", None, "", "", None)
    if evt.value is None or results_state is None or results_state.empty: return empty_return
    
    try:
        selected_index = evt.index[0]
        selected_row = results_state.iloc[selected_index]
        video_id = selected_row['video_id']
        timestamp = selected_row['timestamp']
        keyframe_path = selected_row['keyframe_path']
        video_path = video_path_map.get(video_id)
        
        if not video_path:
            gr.Error(f"Không tìm thấy đường dẫn cho video ID: {video_id}")
            return empty_return

        full_transcript = get_full_transcript_for_video(video_id, transcript_searcher)
        video_clip_path = create_video_segment(video_path, timestamp, duration=30)
        full_video_link_html = generate_full_video_link(video_path)
        
        candidate_for_submission = {
            "keyframe_id": os.path.basename(keyframe_path).replace('.jpg', ''),
            "video_id": video_id, "timestamp": timestamp, "keyframe_path": keyframe_path,
            "final_score": 0.0, "task_type": TaskType.KIS
        }

        return (
            keyframe_path, gr.Video(value=video_clip_path, label=f"Clip 30s từ @ {timestamp:.2f}s"),
            full_transcript, "", full_video_link_html,
            candidate_for_submission, video_id, f"{timestamp:.2f}", selected_index
        )
    except (IndexError, KeyError): return empty_return

# ==============================================================================
# === 5. HANDLERS CHO BẢNG ĐIỀU KHIỂN NỘP BÀI (CỘT PHẢI) ===
# ==============================================================================

def _format_state_to_csv_text(submission_list: list, fps_map: dict) -> str:
    """Helper: Chuyển state thành chuỗi CSV để hiển thị và sửa."""
    if not submission_list: return "video_id,frame_index\n"
    header = "video_id,frame_index\n"
    lines = []
    for item in submission_list:
        video_id = item.get('video_id')
        timestamp = item.get('timestamp')
        if video_id and timestamp is not None:
            fps = fps_map.get(video_id, 30.0)
            frame_index = round(timestamp * fps)
            lines.append(f"{video_id},{frame_index}")
    return header + "\n".join(lines)

def add_to_submission_list(submission_list: list, candidate: dict, position: str, response_state: dict, fps_map: dict):
    if not candidate:
        gr.Warning("Chưa có ứng viên Visual nào được chọn để thêm!")
        return submission_list, _format_state_to_csv_text(submission_list, fps_map)

    task_type = response_state.get("task_type", TaskType.KIS)
    item_to_add = {**candidate, 'task_type': task_type}
    
    if len(submission_list) >= MAX_SUBMISSION_RESULTS:
        gr.Warning(f"Danh sách đã đạt giới hạn {MAX_SUBMISSION_RESULTS} kết quả.")
    else:
        if position == 'top': submission_list.insert(0, item_to_add)
        else: submission_list.append(item_to_add)
        gr.Success(f"Đã thêm kết quả Visual vào {'đầu' if position == 'top' else 'cuối'} danh sách!")
    
    return submission_list, _format_state_to_csv_text(submission_list, fps_map)

def add_transcript_result_to_submission(submission_list: list, results_state: pd.DataFrame, selected_index: int, position: str, fps_map: dict):
    if selected_index is None or results_state is None or results_state.empty:
        gr.Warning("Chưa có kết quả Transcript nào được chọn để thêm!")
        return submission_list, _format_state_to_csv_text(submission_list, fps_map)
    
    try:
        selected_row = results_state.iloc[selected_index]
        candidate = {
            "video_id": selected_row['video_id'], "timestamp": selected_row['timestamp'],
            "keyframe_id": os.path.basename(selected_row['keyframe_path']).replace('.jpg', '')
        }
        # Tái sử dụng logic của hàm add_to_submission_list
        # Cần một response_state giả để hàm hoạt động
        fake_response_state = {"task_type": TaskType.KIS}
        return add_to_submission_list(submission_list, candidate, position, fake_response_state, fps_map)
    except IndexError:
        gr.Warning("Lựa chọn không hợp lệ. Vui lòng chọn lại một dòng trong bảng.")
        return submission_list, _format_state_to_csv_text(submission_list, fps_map)

def sync_submission_state_to_editor(submission_list: list, fps_map: dict) -> str:
    """Đồng bộ hóa state vào Text Editor cho nút Refresh."""
    gr.Info("Bảng điều khiển đã được đồng bộ hóa với danh sách kết quả.")
    return _format_state_to_csv_text(submission_list, fps_map)

def clear_submission_list():
    gr.Info("Đã xóa danh sách nộp bài.")
    return [], "video_id,frame_index\n"

def handle_submission(submission_text: str, query_id: str):
    if not submission_text.strip() or len(submission_text.strip().split('\n')) <= 1:
        gr.Warning("Bảng điều khiển nộp bài đang trống.")
        return None
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
    
    try:
        lines = submission_text.strip().split('\n')
        header = [h.strip() for h in lines[0].split(',')]
        data = [[item.strip() for item in line.split(',')] for line in lines[1:]]
        df = pd.DataFrame(data, columns=header)
        df['frame_index'] = pd.to_numeric(df['frame_index'])
        
        file_path = generate_submission_file(df, query_id=query_id)
        gr.Success(f"Đã tạo file nộp bài thành công từ nội dung đã sửa: {os.path.basename(file_path)}")
        return file_path
    except Exception as e:
        gr.Error(f"Lỗi khi xử lý nội dung nộp bài: {e}. Hãy kiểm tra lại định dạng CSV.")
        traceback.print_exc()
        return None

# ==============================================================================
# === 6. HANDLERS CHO CÁC CÔNG CỤ PHỤ TRỢ VÀ DỌN DẸP ===
# ==============================================================================

def calculate_frame_number(video_id: str, time_input: str, fps_map: dict):
    if not video_id or not time_input: return "Vui lòng nhập đủ thông tin."
    timestamp = parse_time_string(time_input)
    if timestamp is None: return "Lỗi: Định dạng thời gian không hợp lệ."
    fps = fps_map.get(video_id, 30.0)
    frame_number = round(timestamp * fps)
    gr.Info(f"Đã tính toán: {video_id} @ {timestamp}s, FPS={fps} -> Frame #{frame_number}")
    return str(frame_number)

def clear_all():
    """Reset toàn bộ giao diện về trạng thái ban đầu."""
    # Tuple chứa giá trị reset cho TẤT CẢ các component output
    return (
        # Mắt Thần
        "", pd.DataFrame(), "", None, [], 1, "Trang 1 / 1",
        # Tai Thính
        "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None,
        # Trạm Phân tích Hợp nhất
        None, None, "", "", "", None,
        # Bảng điều khiển Nộp bài
        [], "video_id,frame_index\n",
        # Máy tính
        "", "", "",
        # Vùng Xuất file
        "", None
    )