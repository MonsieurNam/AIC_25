# ==============================================================================
# === EVENT HANDLERS - PHIÊN BẢN ĐÃ DỌN DẸP VÀ HỢP NHẤT ===
# ==============================================================================
from io import StringIO
import shutil
import gradio as gr
import pandas as pd
import numpy as np
import time
import os
import re
import traceback
from typing import Dict, Any, List, Optional

# Local imports
from config import ITEMS_PER_PAGE, MAX_SUBMISSION_RESULTS
from ui_helpers import create_detailed_info_html
from search_core.task_analyzer import TaskType
from utils import create_video_segment, generate_submission_file
from utils.formatting import format_submission_list_to_csv_string

# ==============================================================================
# === CÁC HÀM TRỢ GIÚP ===
# ==============================================================================

def generate_full_video_link(video_path: str) -> str:
    # === DEBUG LOG: KIỂM TRA PATH CUỐI CÙNG TRƯỚC KHI TẠO URL ===
    print("\n" + "="*20 + " DEBUG LOG: generate_full_video_link " + "="*20)
    print(f"-> Input video_path: '{video_path}' (Type: {type(video_path)})")
    # === KẾT THÚC DEBUG LOG ===
    
    if not video_path or not os.path.exists(video_path):
        # === DEBUG LOG: PATH KHÔNG HỢP LỆ HOẶC KHÔNG TỒN TẠI ===
        print(f"-> VALIDATION FAILED: Path is None, empty, or does not exist.")
        print("="*73 + "\n")
        # === KẾT THÚC DEBUG LOG ===
        return "<p style='color: #888; text-align: center; padding: 10px;'>Chọn một kết quả để xem link video gốc.</p>"
    
    file_url = f"/file={video_path}"
    
    # === DEBUG LOG: KIỂM TRA URL ĐƯỢC TẠO RA ===
    print(f"-> Generated file_url: '{file_url}'")
    print(f"-> Path exists: {os.path.exists(video_path)}")
    print("="*73 + "\n")
    # === KẾT THÚC DEBUG LOG ===
    
    return f"""<div style='text-align: center; margin-top: 10px;'><a href='{file_url}' target='_blank' style='background-color: #4CAF50; color: white; padding: 10px 15px; text-align: center; text-decoration: none; display: inline-block; border-radius: 8px; font-weight: bold; cursor: pointer;'>🎬 Mở Video Gốc (Toàn bộ) trong Tab mới</a></div>"""

def get_full_transcript_for_video(video_id: str, transcript_searcher) -> str:
    if not transcript_searcher or transcript_searcher.full_data is None: return "Lỗi: Transcript engine chưa sẵn sàng."
    try:
        video_transcripts = transcript_searcher.full_data[transcript_searcher.full_data['video_id'] == video_id]
        full_text = " ".join(video_transcripts['transcript_text'].tolist())
        return full_text if full_text.strip() else "Video này không có lời thoại."
    except Exception: return "Không thể tải transcript cho video này."

def clear_analysis_panel():
    """Helper để xóa các component trong cột phải."""
    return None, None, "", "", "", None, "", "", None

# ==============================================================================
# === HANDLERS CHÍNH CHO CÁC TAB TÌM KIẾM ===
# ==============================================================================

def perform_search(query_text: str, num_results: int, w_clip: float, w_obj: float, w_semantic: float, lambda_mmr: float, master_searcher):
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn.</div>", None, [], 1, "Trang 1 / 1"
    
    gr.Info("Bắt đầu quét visual...")
    try:
        config = {"top_k_final": int(num_results), "w_clip": w_clip, "w_obj": w_obj, "w_semantic": w_semantic, "lambda_mmr": lambda_mmr}
        start_time = time.time()
        full_response = master_searcher.search(query=query_text, config=config)
        search_time = time.time() - start_time
    except Exception as e:
        traceback.print_exc()
        return [], f"<div style='color: red;'>🔥 Lỗi backend: {e}</div>", None, [], 1, "Trang 1 / 1"
    
    gallery_paths = format_results_for_mute_gallery(full_response)
    num_found = len(gallery_paths)
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    status_msg = f"<div style='color: {'#166534' if num_found > 0 else '#d97706'};'>{'✅' if num_found > 0 else '😔'} **{task_type_msg}** | Tìm thấy {num_found} kết quả ({search_time:.2f}s).</div>"
    
    initial_gallery_view = gallery_paths[:ITEMS_PER_PAGE]
    total_pages = int(np.ceil(num_found / ITEMS_PER_PAGE)) or 1
    page_info = f"Trang 1 / {total_pages}"
    
    return initial_gallery_view, status_msg, full_response, gallery_paths, 1, page_info

def handle_transcript_search(query1: str, query2: str, query3: str, transcript_searcher):
    gr.Info("Bắt đầu điều tra transcript...")
    results = None
    if query1.strip(): results = transcript_searcher.search(query1, current_results=results)
    if query2.strip(): results = transcript_searcher.search(query2, current_results=results)
    if query3.strip(): results = transcript_searcher.search(query3, current_results=results)
    
    if results is None or results.empty:
        return "Nhập truy vấn để bắt đầu hoặc không tìm thấy kết quả.", pd.DataFrame(), None
        
    count_str = f"Tìm thấy: {len(results)} kết quả."
    display_df = results[['video_id', 'timestamp', 'transcript_text', 'keyframe_path']].copy()
    display_df.rename(columns={
        'video_id': 'Video ID',
        'timestamp': 'Timestamp (s)',
        'transcript_text': 'Nội dung Lời thoại',
        'keyframe_path': 'Keyframe Path'
    }, inplace=True)
    
    return count_str, display_df, results

def clear_transcript_search():
    return "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None

# ==============================================================================
# === HANDLERS CHO SỰ KIỆN SELECT (CẬP NHẬT TRẠM PHÂN TÍCH) ===
# ==============================================================================

def on_gallery_select(response_state: Dict, current_page: int, transcript_searcher, evt: gr.SelectData):
    empty_return = clear_analysis_panel()
    if not response_state or evt is None: return empty_return
    
    results = response_state.get("results", [])
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    if not results or global_index >= len(results): return empty_return
    
    selected_result = results[global_index]
    video_id = selected_result.get('video_id')
    video_path = selected_result.get('video_path')
    
    print("\n" + "="*20 + " DEBUG LOG: on_gallery_select " + "="*20)
    print(f"-> Selected video_id: {video_id}")
    print(f"-> Retrieved video_path from selected_result: '{video_path}'")
    print("="*65 + "\n")
    
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
    empty_return = clear_analysis_panel()
    if evt.value is None or results_state is None or results_state.empty: return empty_return
    
    try:
        selected_index = evt.index[0]
        selected_row = results_state.iloc[selected_index]
        video_id = selected_row['video_id']
        timestamp = selected_row['timestamp']
        keyframe_path = selected_row['keyframe_path']
        video_path = video_path_map.get(video_id)
        
        print("\n" + "="*20 + " DEBUG LOG: on_transcript_select " + "="*20)
        print(f"-> Selected video_id: {video_id}")
        print(f"-> Retrieved video_path from video_path_map: '{video_path}'")
        print("="*75 + "\n")
        
        if not video_path:
            gr.Error(f"Không tìm thấy đường dẫn cho video ID: {video_id}")
            return empty_return

        full_transcript = get_full_transcript_for_video(video_id, transcript_searcher)
        video_clip_path = create_video_segment(video_path, timestamp, duration=30)
        full_video_link_html = generate_full_video_link(video_path)
        
        candidate_for_submission = {
            "keyframe_id": os.path.basename(keyframe_path).replace('.jpg', ''),
            "video_id": video_id, "timestamp": timestamp, "keyframe_path": keyframe_path,
            "final_score": 0.0, "task_type": TaskType.KIS # Mặc định
        }

        return (
            keyframe_path, gr.Video(value=video_clip_path, label=f"Clip 30s từ @ {timestamp:.2f}s"),
            full_transcript, "", full_video_link_html,
            candidate_for_submission, video_id, f"{timestamp:.2f}", selected_index
        )
    except (IndexError, KeyError) as e:
        gr.Error(f"Lỗi khi xử lý lựa chọn transcript: {e}")
        return empty_return

# ==============================================================================
# === HANDLERS CHO BẢNG ĐIỀU KHIỂN NỘP BÀI ===
# ==============================================================================

def add_to_submission_list(submission_list: list, candidate: dict, position: str):
    if not candidate:
        gr.Warning("Chưa có ứng viên Visual nào được chọn để thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list)

    if len(submission_list) >= MAX_SUBMISSION_RESULTS:
        gr.Warning(f"Danh sách đã đạt giới hạn {MAX_SUBMISSION_RESULTS} kết quả.")
    else:
        # Cần thêm 'task_type' vào candidate trước khi thêm
        item_to_add = candidate.copy()
        if 'task_type' not in item_to_add:
            item_to_add['task_type'] = TaskType.KIS
        
        if position == 'top':
            submission_list.insert(0, item_to_add)
        else:
            submission_list.append(item_to_add)
        gr.Success(f"Đã thêm kết quả Visual vào {'đầu' if position == 'top' else 'cuối'} danh sách!")
    
    return submission_list, format_submission_list_to_csv_string(submission_list)

def add_transcript_result_to_submission(submission_list: list, results_state: pd.DataFrame, selected_index: int, position: str):
    if selected_index is None or results_state is None or results_state.empty:
        gr.Warning("Chưa có kết quả Transcript nào được chọn để thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list)
    
    try:
        selected_row = results_state.iloc[selected_index]
        candidate = {
            "video_id": selected_row['video_id'], "timestamp": selected_row['timestamp'],
            "keyframe_id": os.path.basename(selected_row['keyframe_path']).replace('.jpg', ''),
            "keyframe_path": selected_row['keyframe_path'],
            "task_type": TaskType.KIS # Gán task_type mặc định
        }
        return add_to_submission_list(submission_list, candidate, position)
    except IndexError:
        gr.Warning("Lựa chọn không hợp lệ. Vui lòng chọn lại một dòng trong bảng.")
        return submission_list, format_submission_list_to_csv_string(submission_list)

def sync_submission_state_to_editor(submission_list: list) -> str:
    gr.Info("Bảng điều khiển đã được đồng bộ hóa với danh sách kết quả.")
    return format_submission_list_to_csv_string(submission_list)

def clear_submission_list():
    gr.Info("Đã xóa danh sách nộp bài.")
    return [], ""

def handle_submission(submission_text: str, query_id: str):
    if not submission_text.strip():
        gr.Warning("Bảng điều khiển nộp bài đang trống.")
        return None
    if not query_id.strip():
        gr.Warning("Vui lòng nhập Query ID để tạo file.")
        return None
    
    try:
        # Tái tạo DataFrame từ text người dùng đã sửa
        # Giả định text là CSV không có header
        df = pd.read_csv(StringIO(submission_text.strip()), header=None)
        
        file_path = generate_submission_file(df, query_id=query_id)
        gr.Success(f"Đã tạo file nộp bài thành công từ nội dung đã sửa: {os.path.basename(file_path)}")
        return file_path
    except Exception as e:
        gr.Error(f"Lỗi khi xử lý nội dung nộp bài: {e}. Hãy kiểm tra lại định dạng CSV.")
        return None

# ==============================================================================
# === HANDLERS CHO CÁC CÔNG CỤ PHỤ TRỢ VÀ NÚT TIỆN ÍCH ===
# ==============================================================================

def update_gallery_page(gallery_items: list, current_page: int, direction: str):
    if not gallery_items: return [], 1, "Trang 1 / 1"
    total_items = len(gallery_items)
    total_pages = int(np.ceil(total_items / ITEMS_PER_PAGE)) or 1
    new_page = min(total_pages, current_page + 1) if direction == "▶️ Trang sau" else max(1, current_page - 1)
    start_index = (new_page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    return gallery_items[start_index:end_index], new_page, f"Trang {new_page} / {total_pages}"

def calculate_frame_number(video_id: str, time_input: str, fps_map: dict):
    if not video_id or not time_input: return "Vui lòng nhập Video ID và Thời gian."
    try:
        time_input_str = str(time_input).strip()
        if ':' in time_input_str:
            parts = time_input_str.split(':')
            minutes, seconds = int(parts[0]), float(parts[1])
            timestamp = minutes * 60 + seconds
        else:
            timestamp = float(time_input_str)
            
        fps = fps_map.get(video_id, 30.0)
        return str(round(timestamp * fps))
    except Exception:
        return f"Lỗi: Định dạng thời gian '{time_input}' không hợp lệ."

def clear_all():
    # Tập hợp tất cả các giá trị reset vào một tuple lớn
    return (
        # Mắt Thần
        "", gr.Gallery(value=None), "", None, [], 1, "Trang 1 / 1",
        # Tai Thính
        "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None,
        # Trạm Phân tích Hợp nhất
        None, None, "", "", "", None,
        # Bảng điều khiển Nộp bài
        [], "",
        # Máy tính
        "", "", "",
        # Vùng Xuất file
        "", None
    )
    
def handle_view_full_video(selected_candidate: Dict):
    """
    Sao chép video gốc từ /kaggle/input sang /kaggle/working để phát.
    Đây là giải pháp "Copy-on-Demand".
    """
    if not selected_candidate:
        gr.Warning("Vui lòng chọn một kết quả trước khi xem video gốc.")
        return None

    source_path = selected_candidate.get('video_path')
    if not source_path or not os.path.exists(source_path):
        gr.Error(f"Không tìm thấy file video nguồn tại: {source_path}")
        return None

    # Tạo thư mục đích nếu chưa có
    destination_dir = "/kaggle/working/temp_full_videos"
    os.makedirs(destination_dir, exist_ok=True)
    
    # Tạo đường dẫn đích
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))

    # Sao chép file (chỉ khi nó chưa tồn tại ở đích để tiết kiệm thời gian)
    if not os.path.exists(destination_path):
        gr.Info(f"Đang sao chép video {os.path.basename(source_path)} để chuẩn bị phát...")
        try:
            shutil.copy(source_path, destination_path)
            gr.Success("Sao chép hoàn tất! Bắt đầu phát video.")
        except Exception as e:
            gr.Error(f"Lỗi khi sao chép video: {e}")
            return None
    else:
        gr.Info("Video đã có sẵn, bắt đầu phát.")

    # Trả về đường dẫn mới, an toàn để Gradio phát
    return gr.Video(value=destination_path, label=f"Video Gốc: {os.path.basename(source_path)}")