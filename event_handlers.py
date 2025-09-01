import gradio as gr
import pandas as pd
import numpy as np
import time
import os
import traceback
from typing import Dict, Any, List, Optional

# Local imports from other project files
from config import ITEMS_PER_PAGE, MAX_SUBMISSION_RESULTS, VIDEO_BASE_PATH
from ui_helpers import create_detailed_info_html, format_submission_list_for_display
from search_core.task_analyzer import TaskType
from utils.formatting import format_list_for_submission, format_results_for_mute_gallery
from utils import create_video_segment, generate_submission_file

# ==============================================================================
# === GỌNG KÌM 1: HANDLERS CHO TAB "MẮT THẦN" (VISUAL SCOUT) ===
# ==============================================================================

def perform_search(
    # Inputs from UI
    query_text: str, num_results: int, w_clip: float, w_obj: float, 
    w_semantic: float, lambda_mmr: float,
    # Backend instance
    master_searcher
):
    """
    Xử lý sự kiện tìm kiếm chính cho Tab Visual.
    """
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn.</div>", None, "", [], 1, "Trang 1 / 1"

    loading_html = "<div style='color: #4338ca;'>⏳ Đang quét visual... AI đang phân tích và tìm kiếm.</div>"
    yield ([], loading_html, None, "", [], 1, "Trang 1 / 1")
    
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
        status_msg = f"<div style='color: red;'>🔥 Lỗi backend: {e}</div>"
        return [], status_msg, None, "", [], 1, "Trang 1 / 1"

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
    
    yield (
        initial_gallery_view, status_msg, full_response, str(full_response.get('query_analysis', {})),
        gallery_paths, 1, page_info
    )

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
    """
    Xử lý sự kiện tìm kiếm lồng nhau trên transcript.
    """
    gr.Info("Bắt đầu điều tra transcript...")
    results = None
    if query1.strip():
        results = transcript_searcher.search(query1, current_results=results)
    if query2.strip():
        results = transcript_searcher.search(query2, current_results=results)
    if query3.strip():
        results = transcript_searcher.search(query3, current_results=results)

    if results is None: # Người dùng không nhập gì
        return "Nhập truy vấn để bắt đầu điều tra.", pd.DataFrame(), None

    count_str = f"Tìm thấy: {len(results)} kết quả."
    # Chỉ hiển thị các cột cần thiết trên UI
    display_df = results[['video_id', 'timestamp', 'transcript_text', 'keyframe_path']]
    
    return count_str, display_df, results # Trả về full results cho state

def clear_transcript_search():
    """Xóa các ô tìm kiếm và kết quả của Tab Tai Thính."""
    return "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(columns=["Video ID", "Timestamp (s)", "Nội dung Lời thoại", "Keyframe Path"]), None

def on_transcript_select(results_state: pd.DataFrame, evt: gr.SelectData):
    """
    Xử lý khi người dùng chọn một dòng trong bảng kết quả transcript.
    Sẽ tải và tua video đến đúng thời điểm.
    """
    if evt.value is None or results_state is None or results_state.empty:
        return None
    
    try:
        # evt.index[0] là chỉ số của dòng được chọn
        selected_row = results_state.iloc[evt.index[0]]
        video_id = selected_row['video_id']
        timestamp = selected_row['timestamp']
        
        # Suy ra đường dẫn video từ video_id (cần cấu hình VIDEO_BASE_PATH)
        video_path = os.path.join(VIDEO_BASE_PATH, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            gr.Error(f"Không tìm thấy file video: {video_path}")
            return None
        
        # Sử dụng tính năng start_time của gr.Video để tua đến đúng đoạn
        return gr.Video(value=video_path, start_time=timestamp)
    except (IndexError, KeyError) as e:
        gr.Error(f"Lỗi khi xử lý lựa chọn: {e}")
        return None

# ==============================================================================
# === HANDLERS DÙNG CHUNG (PHÂN TÍCH, NỘP BÀI, CÔNG CỤ) ===
# ==============================================================================

def _get_full_video_path_from_keyframe(keyframe_path: str) -> Optional[str]:
    """Helper: Suy ra đường dẫn video đầy đủ từ đường dẫn keyframe."""
    if not keyframe_path: return None
    try:
        # e.g., /.../keyframes/L21_V001/001.jpg -> L21_V001
        video_id = os.path.basename(os.path.dirname(keyframe_path))
        full_path = os.path.join(VIDEO_BASE_PATH, f"{video_id}.mp4")
        return full_path if os.path.exists(full_path) else None
    except Exception:
        return None

def on_gallery_select(response_state: Dict, current_page: int, evt: gr.SelectData):
    """
    Xử lý khi click vào ảnh trong gallery.
    PHIÊN BẢN V2: Cập nhật thêm công cụ tính toán và nút xem full video.
    """
    empty_return = (None, None, "", None, None, "", "")
    if not response_state or evt is None: return empty_return

    results = response_state.get("results", [])
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    
    if not results or global_index >= len(results): return empty_return

    selected_result = results[global_index]
    keyframe_path = selected_result.get('keyframe_path')
    video_path = selected_result.get('video_path') # Đã có sẵn trong metadata
    timestamp = selected_result.get('timestamp')
    video_id = selected_result.get('video_id')

    # Tạo clip 30 giây
    video_clip_path = create_video_segment(video_path, timestamp, duration=30)
    
    # Tạo HTML hiển thị điểm số
    analysis_html = create_detailed_info_html(selected_result, response_state.get("task_type"))

    return (
        keyframe_path,                      # selected_image_display
        video_clip_path,                    # video_player
        analysis_html,                      # analysis_display_html
        selected_result,                    # selected_candidate_for_submission
        video_id,                           # frame_calculator_video_id
        timestamp,                          # frame_calculator_timestamp
        video_path                          # State ẩn để nút "Mở video gốc" sử dụng
    )
    
def get_full_video_path_for_button(video_path):
    """Tạo ra một file tạm thời để Gradio có thể phục vụ nó."""
    if video_path and os.path.exists(video_path):
        return video_path
    return None

def add_to_submission_list(
    submission_list: list, candidate: Dict, response_state: Dict, position: str
):
    """Thêm ứng viên vào danh sách nộp bài (Không thay đổi logic)."""
    if not candidate:
        gr.Warning("Chưa có ứng viên nào được chọn để thêm!")
        return format_submission_list_for_display(submission_list), submission_list, gr.Dropdown()

    task_type = response_state.get("task_type")
    item_to_add = {**candidate, 'task_type': task_type}
    
    if len(submission_list) >= MAX_SUBMISSION_RESULTS:
        gr.Warning(f"Danh sách đã đạt giới hạn {MAX_SUBMISSION_RESULTS} kết quả.")
        submission_list = submission_list[:MAX_SUBMISSION_RESULTS]
    else:
        if position == 'top':
            submission_list.insert(0, item_to_add)
        else:
            submission_list.append(item_to_add)
        gr.Success(f"Đã thêm kết quả vào {'đầu' if position == 'top' else 'cuối'} danh sách!")

    new_choices = [f"{i+1}. {item.get('keyframe_id') or 'TRAKE (' + str(item.get('video_id')) + ')'}" for i, item in enumerate(submission_list)]
    return format_submission_list_for_display(submission_list), submission_list, gr.Dropdown(choices=new_choices, value=None)

def modify_submission_list(
    submission_list: list, selected_item_index_str: str, action: str
):
    """Modifies the submission list (move up/down, remove)."""
    if not selected_item_index_str:
        gr.Warning("Vui lòng chọn một mục từ danh sách để thao tác.")
        return format_submission_list_for_display(submission_list), submission_list, selected_item_index_str
    try:
        index = int(selected_item_index_str.split('.')[0]) - 1
        if not (0 <= index < len(submission_list)): raise ValueError("Index out of bounds")
    except:
        gr.Error("Lựa chọn không hợp lệ.")
        return format_submission_list_for_display(submission_list), submission_list, None

    if action == 'move_up' and index > 0:
        submission_list[index], submission_list[index-1] = submission_list[index-1], submission_list[index]
    elif action == 'move_down' and index < len(submission_list) - 1:
        submission_list[index], submission_list[index+1] = submission_list[index+1], submission_list[index]
    elif action == 'remove':
        submission_list.pop(index)

    new_choices = [f"{i+1}. {item.get('keyframe_id') or 'TRAKE (' + str(item.get('video_id')) + ')'}" for i, item in enumerate(submission_list)]
    return format_submission_list_for_display(submission_list), submission_list, gr.Dropdown(choices=new_choices, value=None)
    
def calculate_frame_number(video_id: str, timestamp: float, fps_map: dict):
    """
    Tính toán số thứ tự frame dựa trên video_id và timestamp.
    """
    if not video_id or timestamp is None:
        return "Vui lòng nhập đủ thông tin."
    
    fps = fps_map.get(video_id, 30.0) # Sử dụng FPS mặc định nếu không tìm thấy
    frame_number = round(float(timestamp) * fps)
    
    gr.Info(f"Đã tính toán: {video_id} @ {timestamp}s, FPS={fps} -> Frame #{frame_number}")
    return str(frame_number)

def handle_submission(submission_list: list, query_id: str):
    """
    Tạo file CSV nộp bài.
    Logic bên trong đã được nâng cấp ở `formatting.py` nên hàm này không cần đổi.
    """
    if not submission_list:
        gr.Warning("Danh sách nộp bài đang trống.")
        return None
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
        
    submission_df = format_list_for_submission(submission_list, max_results=MAX_SUBMISSION_RESULTS)
    if submission_df.empty:
        gr.Warning("Không thể định dạng kết quả để nộp bài.")
        return None
        
    file_path = generate_submission_file(submission_df, query_id=query_id)
    gr.Info(f"Đã tạo file nộp bài thành công: {os.path.basename(file_path)}")
    return file_path
    
def clear_submission_list():
    """Xóa toàn bộ danh sách nộp bài."""
    gr.Info("Đã xóa danh sách nộp bài.")
    return "Chưa có kết quả nào được thêm vào.", [], gr.Dropdown(choices=[], value=None)
    
# ==============================================================================
# === HANDLER DỌN DẸP TOÀN BỘ HỆ THỐNG ===
# ==============================================================================

def clear_all():
    """
    Reset toàn bộ giao diện về trạng thái ban đầu.
    Trả về một tuple lớn chứa tất cả các giá trị mặc định.
    """
    # Giá trị trả về phải khớp 1-1 với danh sách `clear_all_outputs` trong app.py
    return (
        # --- Tab Mắt Thần (7 outputs) ---
        [],                                         # results_gallery
        "",                                         # status_output
        None,                                       # response_state
        "Trang 1 / 1",                              # page_info_display
        [],                                         # gallery_items_state
        1,                                          # current_page_state
        
        # --- Tab Tai Thính (6 outputs) ---
        "",                                         # transcript_query_1
        "",                                         # transcript_query_2
        "",                                         # transcript_query_3
        "Tìm thấy: 0 kết quả.",                      # transcript_results_count
        pd.DataFrame(columns=["Video ID", "Timestamp (s)", "Nội dung Lời thoại", "Keyframe Path"]), # transcript_results_df
        None,                                       # transcript_video_player
        None,                                       # transcript_results_state

        # --- Cột Phải: Trạm Phân tích (4 outputs) ---
        None,                                       # selected_image_display
        None,                                       # video_player
        None,                                       # selected_candidate_for_submission
        None,                                       # full_video_path_state (State)

        # --- Cột Phải: Công cụ tính toán (3 outputs) ---
        "",                                         # frame_calculator_video_id
        0,                                          # frame_calculator_timestamp
        "",                                         # frame_calculator_output

        # --- Cột Phải: Vùng Nộp bài (5 outputs) ---
        "Chưa có kết quả nào được thêm vào.",        # submission_list_display
        [],                                         # submission_list_state
        gr.Dropdown(choices=[], value=None),        # submission_list_selector
        "",                                         # query_id_input
        None,                                       # submission_file_output
    )