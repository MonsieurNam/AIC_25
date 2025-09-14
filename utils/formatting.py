from search_core.task_analyzer import TaskType
import pandas as pd
from typing import List, Dict, Any
import os
import json
def format_submission_list_to_csv_string(submission_list: List[Dict], fps_map: dict) -> str:
    """
    Chuyển danh sách nộp bài thành một chuỗi CSV để hiển thị và chỉnh sửa.
    """
    if not submission_list:
        return "" 
    
    df = format_list_for_submission(submission_list, fps_map=fps_map)
    
    if df.empty:
        return ""
        
    csv_string = df.to_csv(header=False, index=False)
    return csv_string

def _load_fps_map(path="/kaggle/input/stage1/video_fps_map.json") -> Dict[str, float]:
    """
    Tải và cache bản đồ FPS. Chỉ được gọi một lần.
    """
    if not os.path.exists(path):
        print(f"--- ⚠️ CẢNH BÁO NỘP BÀI: Không tìm thấy file '{path}'. Sẽ dùng FPS mặc định là 30.0 ---")
        return {}
    try:
        with open(path, 'r') as f:
            fps_map = json.load(f)
            print(f"--- ✅ Tải thành công bản đồ FPS cho {len(fps_map)} video. ---")
            return fps_map
    except Exception as e:
        print(f"--- ❌ LỖI NỘP BÀI: Không thể đọc file FPS map. Lỗi: {e}. Sẽ dùng FPS mặc định là 30.0 ---")
        return {}

FPS_MAP = _load_fps_map()
DEFAULT_FPS = 30.0

def format_results_for_gallery(response: Dict[str, Any]) -> List[str]:
    """
    Định dạng kết quả thô thành định dạng cho gr.Gallery (chỉ trả về đường dẫn ảnh).
    PHIÊN BẢN "COCKPIT V3.3"
    """
    results = response.get("results", [])
    task_type = response.get("task_type")
    
    gallery_paths = []
    if not results:
        return []

    for res in results:
        keyframe_path = None
        if task_type == TaskType.TRAKE:
            sequence = res.get('sequence', [])
            if sequence:
                keyframe_path = sequence[0].get('keyframe_path')
        else:
            keyframe_path = res.get('keyframe_path')

        if keyframe_path and os.path.isfile(keyframe_path):
            gallery_paths.append(keyframe_path)
            
    return gallery_paths

def format_results_for_mute_gallery(response: Dict[str, Any]) -> List[str]:
    """
    Định dạng kết quả thô CHỈ LẤY ĐƯỜNG DẪN ẢNH cho "Lưới ảnh câm" (Cockpit v3.3).
    """
    print("\n" + "="*20 + " DEBUG LOG: format_results_for_mute_gallery " + "="*20)
    print(f"-> Nhận được response với các key: {response.keys() if isinstance(response, dict) else 'Không phải dict'}")
    results = response.get("results", [])
    task_type = response.get("task_type")
    print(f"-> Task Type: {task_type}")
    print(f"-> Số lượng 'results' nhận được: {len(results)}")
    if results:
        print(f"-> Cấu trúc của result đầu tiên: {results[0].keys() if isinstance(results[0], dict) else 'Không phải dict'}")
        if 'keyframe_path' in results[0]:
             print(f"  -> Key 'keyframe_path' tồn tại. Giá trị: {results[0]['keyframe_path']}")
        else:
             print("  -> 🚨 CẢNH BÁO: Key 'keyframe_path' KHÔNG TỒN TẠI trong result đầu tiên!")
    print("="*75 + "\n")

    if not results:
        return []
        
    task_type = response.get("task_type")
    
    keyframe_paths = []

    if task_type == TaskType.TRAKE:
        for sequence_result in results:
            sequence = sequence_result.get('sequence', [])
            if sequence: 
                first_frame = sequence[0]
                path = first_frame.get('keyframe_path')
                if path and os.path.isfile(path):
                    keyframe_paths.append(path)
    
    else:
        for single_frame_result in results:
            path = single_frame_result.get('keyframe_path')
            if path and os.path.isfile(path):
                keyframe_paths.append(path)

    return keyframe_paths

def format_for_submission(response: Dict[str, Any], max_results: int = 100) -> pd.DataFrame:
    """
    Định dạng kết quả thô thành một DataFrame sẵn sàng để lưu ra file CSV nộp bài.

    Args:
        response (Dict[str, Any]): Dictionary kết quả trả về từ MasterSearcher.search().
        max_results (int): Số lượng dòng tối đa trong file nộp bài.

    Returns:
        pd.DataFrame: DataFrame có các cột phù hợp với yêu cầu của ban tổ chức.
    """
    task_type = response.get("task_type")
    results = response.get("results", [])
        
    submission_data = []

    if task_type == TaskType.KIS:
        for res in results:
            try:
                frame_index = int(res.get('keyframe_id', '').split('_')[-1])
                submission_data.append({
                    'video_id': res.get('video_id'),
                    'frame_index': frame_index
                })
            except (ValueError, IndexError):
                continue

    elif task_type == TaskType.QNA:
        for res in results:
            try:
                frame_index = int(res.get('keyframe_id', '').split('_')[-1])
                submission_data.append({
                    'video_id': res.get('video_id'),
                    'frame_index': frame_index,
                    'answer': res.get('answer', '')
                })
            except (ValueError, IndexError):
                continue
    
    elif task_type == TaskType.TRAKE:
        for seq_res in results:
            sequence = seq_res.get('sequence', [])
            if not sequence:
                continue
            
            row = {'video_id': seq_res.get('video_id')}
            for i, frame in enumerate(sequence):
                try:
                    frame_index = int(frame.get('keyframe_id', '').split('_')[-1])
                    row[f'frame_moment_{i+1}'] = frame_index
                except (ValueError, IndexError):
                    row[f'frame_moment_{i+1}'] = -1 
            submission_data.append(row)

    if not submission_data:
        return pd.DataFrame()

    df = pd.DataFrame(submission_data)
    
    return df.head(max_results)

def generate_submission_file(df: pd.DataFrame, query_id: str, output_dir: str = "/kaggle/working/submissions") -> str:
    """
    Lưu DataFrame thành file CSV theo đúng định dạng tên file.

    Args:
        df (pd.DataFrame): DataFrame đã được định dạng để nộp bài.
        query_id (str): ID của câu truy vấn (ví dụ: 'query_01').
        output_dir (str): Thư mục để lưu file.

    Returns:
        str: Đường dẫn đến file CSV đã được tạo.
    """
    if df.empty:
        return "Không có dữ liệu để tạo file."

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{query_id}.csv")
    
    df.to_csv(file_path, header=False, index=False)
    
    print(f"--- ✅ Đã tạo file nộp bài tại: {file_path} ---")
    return file_path

def format_list_for_submission(submission_list: List[Dict], fps_map: dict, max_results: int = 100) -> pd.DataFrame:
    """
    Định dạng một danh sách các dictionary kết quả thành DataFrame để nộp bài.
    PHIÊN BẢN V2: Tính toán số thứ tự frame chính xác bằng FPS_MAP.
    """
    if not submission_list:
        return pd.DataFrame()
        
    submission_data = []
    task_type = submission_list[0].get('task_type')
    
    if task_type in [TaskType.KIS, TaskType.QNA]:
        for res in submission_list:
            video_id = res.get('video_id')
            timestamp = res.get('timestamp')
            
            if video_id is None or timestamp is None:
                continue

            fps = fps_map.get(video_id, DEFAULT_FPS)
            frame_number = round(timestamp * fps)
            
            row = {'video_id': video_id, 'frame_number': frame_number}
            if task_type == TaskType.QNA:
                row['answer'] = res.get('answer', '')
            
            submission_data.append(row)

    elif task_type == TaskType.TRAKE:
        for seq_res in submission_list:
            video_id = seq_res.get('video_id')
            sequence = seq_res.get('sequence', [])
            if not video_id or not sequence:
                continue
            
            fps = fps_map.get(video_id, DEFAULT_FPS)
            row = {'video_id': video_id}
            
            for i, frame in enumerate(sequence):
                timestamp = frame.get('timestamp')
                if timestamp is not None:
                    frame_number = round(timestamp * fps)
                    row[f'frame_moment_{i+1}'] = frame_number
                else:
                    row[f'frame_moment_{i+1}'] = -1
            submission_data.append(row)

    if not submission_data:
        return pd.DataFrame()

    df = pd.DataFrame(submission_data)
    if 'frame_number' in df.columns:
        df.rename(columns={'frame_number': 'frame_index'}, inplace=True)
        
    return df.head(max_results)