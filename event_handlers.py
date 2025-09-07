# ==============================================================================
# === EVENT HANDLERS - PHIÊN BẢN ĐÃ DỌN DẸP VÀ HỢP NHẤT ===
# ==============================================================================
import html
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
from utils.formatting import format_submission_list_to_csv_string, format_results_for_mute_gallery 

# ==============================================================================
# === CÁC HÀM TRỢ GIÚP ===
# ==============================================================================
def highlight_keywords(full_text: str, keywords: List[str]) -> str:
    """
    Tô sáng tất cả các từ khóa trong một đoạn văn bản và chuyển nó thành HTML.
    - Xử lý case-insensitive.
    - An toàn với các ký tự đặc biệt trong HTML.
    - Chuyển đổi ký tự xuống dòng thành thẻ <br>.
    """
    valid_keywords = [kw for kw in keywords if kw and kw.strip()]
    
    if not valid_keywords:
        return html.escape(full_text).replace("\n", "<br>")
    pattern = "|".join(re.escape(kw) for kw in valid_keywords)
    highlighted_text = re.sub(
        pattern,
        lambda m: f"<mark>{m.group(0)}</mark>",
        full_text,
        flags=re.IGNORECASE
    )
    return highlighted_text.replace("\n", "<br>")

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

def clear_gallery():
    """
    Hàm trợ giúp siêu nhỏ, chỉ trả về None để xóa sạch nội dung của Gallery.
    Đây là bước đầu tiên trong kỹ thuật "Two-Step Update".
    """
    print("--- 🔄 Clearing gallery for page update... ---")
    return None

def perform_search(
    # --- Các tham số cũ ---
    query_text: str, num_results: int, 
    w_clip: float, w_obj: float, w_semantic: float, 
    lambda_mmr: float, initial_retrieval_count: int,
    # --- ✅ Các tham số mới từ slider ---
    w_spatial: float, w_fine_grained: float,
    # --- Backend object từ partial ---
    master_searcher
):
    """
    Hàm xử lý sự kiện tìm kiếm chính - Phiên bản PHOENIX hoàn thiện.
    """
    if not query_text.strip():
        gr.Warning("Vui lòng nhập truy vấn tìm kiếm!")
        return [], "<div style='color: orange;'>⚠️ Vui lòng nhập truy vấn.</div>", None, [], 1, "Trang 1 / 1"
    
    gr.Info("🚀 Kích hoạt quy trình tìm kiếm đa tầng PHOENIX...")
    
    try:
        # Đóng gói TOÀN BỘ cấu hình vào một dictionary duy nhất
        config = {
            "top_k_final": int(num_results),
            "kis_retrieval": int(initial_retrieval_count),
            "lambda_mmr": lambda_mmr,
            "weights": {
                'w_clip': w_clip,
                'w_obj': w_obj, # w_obj vẫn được gửi xuống, dù có thể không dùng trong PHOENIX
                'w_semantic': w_semantic,
                'w_spatial': w_spatial,
                'w_fine_grained': w_fine_grained
            }
        }
        
        start_time = time.time()
        full_response = master_searcher.search(query=query_text, config=config)
        search_time = time.time() - start_time
        
    except Exception as e:
        traceback.print_exc()
        return [], f"<div style='color: red;'>🔥 Lỗi backend: {e}</div>", None, [], 1, "Trang 1 / 1"
    
    # --- Phần xử lý kết quả và trả về cho UI giữ nguyên ---
    gallery_paths = format_results_for_mute_gallery(full_response)
    num_found = len(gallery_paths)
    task_type_msg = full_response.get('task_type', TaskType.KIS).value
    status_msg = f"<div style='color: {'#166534' if num_found > 0 else '#d97706'};'>{'✅' if num_found > 0 else '😔'} **{task_type_msg}** | Tìm thấy {num_found} kết quả ({search_time:.2f}s).</div>"
    
    initial_gallery_view = gallery_paths[:ITEMS_PER_PAGE]
    total_pages = int(np.ceil(num_found / ITEMS_PER_PAGE)) or 1
    page_info = f"Trang 1 / {total_pages}"
    
    return initial_gallery_view, status_msg, full_response, gallery_paths, 1, page_info

def handle_transcript_search(query1: str, query2: str, query3: str, transcript_searcher, fps_map: dict):
    gr.Info("Bắt đầu điều tra transcript...")
    results = None
    if query1.strip(): results = transcript_searcher.search(query1, current_results=results)
    if query2.strip(): results = transcript_searcher.search(query2, current_results=results)
    if query3.strip(): results = transcript_searcher.search(query3, current_results=results)
    
    if results is None or results.empty:
        return "Nhập truy vấn để bắt đầu hoặc không tìm thấy kết quả.", pd.DataFrame(), None
        
    count_str = f"Tìm thấy: {len(results)} kết quả."
    results['fps'] = results['video_id'].map(fps_map).fillna('N/A')

    keywords_to_highlight = [q for q in [query1, query2, query3] if q and q.strip()]
    if keywords_to_highlight:
        results['highlighted_text'] = results['transcript_text'].apply(
            lambda text: highlight_keywords(text, keywords_to_highlight)
        )
    else:
        results['highlighted_text'] = results['transcript_text']
        
    display_df = results[['video_id', 'fps', 'timestamp', 'highlighted_text', 'keyframe_path']].copy()
    
    display_df.rename(columns={
        'video_id': 'Video ID',
        'fps': 'FPS', # <-- Thêm tên cột mới
        'timestamp': 'Timestamp (s)',
        'highlighted_text': 'Nội dung Lời thoại',
        'keyframe_path': 'Keyframe Path'
    }, inplace=True)
    
    return count_str, display_df, results

def clear_transcript_search():
    return "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None

# ==============================================================================
# === HANDLERS CHO SỰ KIỆN SELECT (CẬP NHẬT TRẠM PHÂN TÍCH) ===
# ==============================================================================

def on_gallery_select(response_state: Dict, current_page: int, query_text: str, transcript_searcher, evt: gr.SelectData):
    empty_return = clear_analysis_panel()
    if not response_state or evt is None: return empty_return
    
    results = response_state.get("results", [])
    global_index = (current_page - 1) * ITEMS_PER_PAGE + evt.index
    if not results or global_index >= len(results): return empty_return
    
    selected_result = results[global_index]
    video_id = selected_result.get('video_id')
    
    # Lấy lại video_path từ selected_result vì nó đã có sẵn
    video_path = selected_result.get('video_path')
    
    print("\n" + "="*20 + " DEBUG LOG: on_gallery_select " + "="*20)
    print(f"-> Selected video_id: {video_id}")
    print(f"-> Retrieved video_path from selected_result: '{video_path}'")
    print("="*65 + "\n")
    
    keyframe_path = selected_result.get('keyframe_path')
    timestamp = selected_result.get('timestamp', 0.0)
    
    full_transcript = get_full_transcript_for_video(video_id, transcript_searcher)
    
    highlighted_transcript = highlight_keywords(full_transcript, [query_text])
    
    video_clip_path = create_video_segment(video_path, timestamp, duration=30)
    analysis_html = create_detailed_info_html(selected_result, response_state.get("task_type"))

    candidate_for_submission = selected_result

    return (
        keyframe_path, gr.Video(value=video_clip_path, label=f"Clip 30s từ @ {timestamp:.2f}s"),
        gr.Markdown(value=highlighted_transcript),
        analysis_html,
        candidate_for_submission, 
        video_id, f"{timestamp:.2f}", None
    )

def on_transcript_select(results_state: pd.DataFrame, video_path_map: dict, transcript_searcher, query1: str, query2: str, query3: str, evt: gr.SelectData):
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
        keywords_to_highlight = [q for q in [query1, query2, query3] if q and q.strip()]
        highlighted_transcript = highlight_keywords(full_transcript, keywords_to_highlight)
        video_clip_path = create_video_segment(video_path, timestamp, duration=30)
        
        candidate_for_submission = {
            "keyframe_id": os.path.basename(keyframe_path).replace('.jpg', ''),
            "video_id": video_id,
            "timestamp": timestamp,
            "keyframe_path": keyframe_path,
            "video_path": video_path,  
            "final_score": 0.0,
            "task_type": TaskType.KIS
        }

        return (
            keyframe_path, gr.Video(value=video_clip_path, label=f"Clip 30s từ @ {timestamp:.2f}s"),
            highlighted_transcript,
            "", 
            candidate_for_submission, video_id, f"{timestamp:.2f}", selected_index
        )
    except (IndexError, KeyError) as e:
        gr.Error(f"Lỗi khi xử lý lựa chọn transcript: {e}")
        return empty_return

# ==============================================================================
# === HANDLERS CHO BẢNG ĐIỀU KHIỂN NỘP BÀI ===
# ==============================================================================

def add_to_submission_list(submission_list: list, candidate: dict, position: str, fps_map: dict):
    if not candidate:
        gr.Warning("Chưa có ứng viên Visual nào được chọn để thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list, fps_map)

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
    
    return submission_list, format_submission_list_to_csv_string(submission_list, fps_map)

def add_transcript_result_to_submission(submission_list: list, results_state: pd.DataFrame, selected_index: int, position: str, fps_map: dict):
    if selected_index is None or results_state is None or results_state.empty:
        gr.Warning("Chưa có kết quả Transcript nào được chọn để thêm!")
        return submission_list, format_submission_list_to_csv_string(submission_list, fps_map)
    
    try:
        selected_row = results_state.iloc[selected_index]
        candidate = {
            "video_id": selected_row['video_id'], "timestamp": selected_row['timestamp'],
            "keyframe_id": os.path.basename(selected_row['keyframe_path']).replace('.jpg', ''),
            "keyframe_path": selected_row['keyframe_path'],
            "task_type": TaskType.KIS # Gán task_type mặc định
        }
        return add_to_submission_list(submission_list, candidate, position, fps_map)
    except IndexError:
        gr.Warning("Lựa chọn không hợp lệ. Vui lòng chọn lại một dòng trong bảng.")
        return submission_list, format_submission_list_to_csv_string(submission_list, fps_map)

def sync_submission_state_to_editor(submission_list: list, fps_map: dict) -> str:
    gr.Info("Bảng điều khiển đã được đồng bộ hóa với danh sách kết quả.")
    return format_submission_list_to_csv_string(submission_list, fps_map)

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

# Sửa lại event_handlers.py

def clear_all():
    return (
        # Mắt Thần (7)
        "", gr.Gallery(value=None), "", None, [], 1, "Trang 1 / 1",
        # Tai Thính (6)
        "", "", "", "Tìm thấy: 0 kết quả.", pd.DataFrame(), None,
        # Trạm Phân tích Hợp nhất (5)
        None, None, "", "", None, #<-- SỬA LẠI THÀNH 5 GIÁ TRỊ
        # Bảng điều khiển Nộp bài (2)
        [], "",
        # Máy tính (3)
        "", "", "",
        # Vùng Xuất file (2)
        "", None
    )
    
def handle_view_full_video(selected_candidate: Dict):
    """
    Sao chép video gốc từ /kaggle/input sang /kaggle/working để phát.
    Phiên bản này có log chi tiết để theo dõi quá trình.
    """
    # === LOG: BẮT ĐẦU QUY TRÌNH ===
    print("\n" + "="*20 + " LOG: Tải Video Gốc " + "="*20)
    
    # 1. Kiểm tra đầu vào
    if not selected_candidate or not isinstance(selected_candidate, dict):
        gr.Warning("Vui lòng chọn một kết quả hợp lệ trước khi xem video gốc.")
        print("-> [VALIDATION FAILED] selected_candidate không hợp lệ hoặc không phải dict.")
        print("="*60 + "\n")
        return None
    
    video_id = selected_candidate.get('video_id', 'N/A')
    print(f"-> Nhận lệnh tải video cho: '{video_id}'")

    # 2. Lấy và kiểm tra đường dẫn nguồn
    source_path = selected_candidate.get('video_path')
    print(f"   -> Đường dẫn nguồn (source): '{source_path}'")
    if not source_path or not os.path.exists(source_path):
        gr.Error(f"Không tìm thấy file video nguồn tại: {source_path}")
        print(f"-> [VALIDATION FAILED] Đường dẫn nguồn không tồn tại.")
        print("="*60 + "\n")
        return None

    # 3. Chuẩn bị đường dẫn đích
    destination_dir = "/kaggle/working/temp_full_videos"
    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))
    print(f"   -> Đường dẫn đích (destination): '{destination_path}'")

    # 4. Logic sao chép chính
    if not os.path.exists(destination_path):
        gr.Info(f"Đang sao chép video '{os.path.basename(source_path)}'...")
        print(f"   -> File chưa tồn tại ở đích. Bắt đầu sao chép...")
        
        start_time = time.time() # Bắt đầu đếm giờ
        try:
            shutil.copy(source_path, destination_path)
            end_time = time.time() # Kết thúc đếm giờ
            elapsed_time = end_time - start_time
            
            gr.Success("Sao chép hoàn tất! Bắt đầu phát video.")
            print(f"   -> ✅ Sao chép thành công sau {elapsed_time:.2f} giây.")

        except Exception as e:
            gr.Error(f"Lỗi khi sao chép video: {e}")
            print(f"   -> ❌ LỖI trong quá trình sao chép: {e}")
            print("="*60 + "\n")
            return None
    else:
        gr.Info("Video đã có sẵn trong cache, bắt đầu phát.")
        print("   -> File đã tồn tại ở đích. Bỏ qua bước sao chép.")

    # 5. Trả kết quả về cho Gradio
    print(f"-> Hoàn tất. Trả về đường dẫn '{destination_path}' cho Gradio.")
    print("="*60 + "\n")
    
    return gr.Video(value=destination_path, label=f"Video Gốc: {os.path.basename(source_path)}")

