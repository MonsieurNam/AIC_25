# /AIC25_Video_Search_Engine/ui_helpers.py

import os
import base64
from typing import Dict, Any, List
from search_core.task_analyzer import TaskType

def encode_image_to_base64(image_path: str) -> str:
    """Mã hóa một file ảnh thành chuỗi base64 để nhúng vào HTML."""
    if not image_path or not os.path.isfile(image_path):
        return ""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"--- ⚠️ Lỗi khi mã hóa ảnh {image_path}: {e} ---")
        return ""

def create_detailed_info_html(result: Dict[str, Any], task_type: TaskType) -> str:
    """
    Tạo mã HTML hiển thị thông tin chi tiết của một kết quả được chọn.
    *** PHIÊN BẢN PHOENIX REBORN: HIỂN THỊ BẰNG CHỨNG KHỚP ***
    """
    # Hàm phụ trợ tạo thanh tiến trình (không đổi)
    def create_progress_bar(score, color):
        percentage = max(0, min(100, score * 100))
        # Dùng điểm log để thanh tiến trình trông đẹp hơn với các giá trị điểm rất lớn
        # tuy nhiên, để đơn giản, ta sẽ chuẩn hóa tạm thời
        normalized_score = min(1.0, score / 10000 if score > 1 else score) # Giả định một ngưỡng chuẩn hóa
        percentage = max(0, min(100, normalized_score * 100))
        return f"""<div style='background: #e9ecef; border-radius: 5px; overflow: hidden; height: 8px;'><div style='background: {color}; width: {percentage}%; height: 100%;'></div></div>"""

    # Lấy dữ liệu cơ bản (không đổi)
    video_id = result.get('video_id', 'N/A')
    keyframe_id = result.get('keyframe_id', 'N/A')
    timestamp = result.get('timestamp', 0)
    final_score = result.get('final_score', 0)

    # Bảng thông tin cơ bản (không đổi)
    info_html = f"""
    <div style='font-size: 14px; line-height: 1.6; background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e5e7eb; color: #374151;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
            <span style='color: #6b7280;'>📹 Video ID:</span>
            <code style='background-color: #e9ecef; padding: 2px 6px; border-radius: 4px; color: #4338ca; font-size: 13px;'>{video_id}</code>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
            <span style='color: #6b7280;'>🖼️ Keyframe ID:</span>
            <code style='background-color: #e9ecef; padding: 2px 6px; border-radius: 4px; color: #4338ca; font-size: 13px;'>{keyframe_id}</code>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #6b7280;'>⏰ Timestamp:</span>
            <code style='background-color: #e9ecef; padding: 2px 6px; border-radius: 4px; color: #4338ca; font-size: 13px;'>{timestamp:.2f}s</code>
        </div>
    </div>
    """

    # Bảng điểm số (thay đổi nhỏ để hiển thị điểm mới)
    scores_html = f"""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #e5e7eb;'>
        <h4 style='margin: 0 0 15px 0; color: #111827; text-align: center;'>🏆 Phân tích Điểm</h4>
        <div style='margin-bottom: 12px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; color: #374151;'>
                <span><strong>📊 Điểm Đồng xuất hiện:</strong></span>
                <span style='font-weight: bold; font-size: 16px;'>{final_score:.2f}</span>
            </div>
            {create_progress_bar(final_score, '#10b981')}
        </div>
    </div>
    """
    
    # ==============================================================================
    # === NÂNG CẤP GIAI ĐOẠN 3: HIỂN THỊ BẰNG CHỨNG KHỚP ===
    # ==============================================================================
    matched_queries = result.get('matched_queries', [])
    evidence_html = ""
    if matched_queries:
        evidence_html = """
        <div style='background-color: #f0fdf4; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #bbf7d0;'>
            <h4 style='margin: 0 0 10px 0; color: #166534; text-align: center;'>✅ Bằng chứng Khớp</h4>
            <p style='text-align: center; font-size: 13px; color: #15803d; margin-top: -5px; margin-bottom: 10px;'>Kết quả này được đề xuất vì khớp với các mô tả sau:</p>
            <ul style='margin: 0; padding-left: 20px; list-style-type: disc;'>
        """
        for mq in matched_queries:
            evidence_html += f"<li style='margin-bottom: 5px; color: #14532d; font-size: 14px;'>{mq}</li>"
        evidence_html += "</ul></div>"
    # ==============================================================================
    
    # Trả về chuỗi HTML hoàn chỉnh
    return info_html + scores_html + evidence_html

def format_submission_list_for_display(submission_list: List[Dict[str, Any]]) -> str:
    """Biến danh sách submission thành một chuỗi text đẹp mắt để hiển thị."""
    if not submission_list:
        return "Chưa có kết quả nào được thêm vào."
    
    display_text = []
    for i, item in enumerate(submission_list):
        task_type = item.get('task_type')
        item_info = ""
        if task_type == TaskType.TRAKE:
            item_info = f"TRAKE Seq | Vid: {item.get('video_id')} | Score: {item.get('final_score', 0):.3f}"
        else:  # KIS, QNA
            item_info = f"Frame | {item.get('keyframe_id')} | Score: {item.get('final_score', 0):.3f}"
        
        display_text.append(f"{i+1:02d}. {item_info}")
    
    return "\n".join(display_text)

def build_selected_preview(gallery_items, selected_indices):
    """Tạo danh sách đường dẫn ảnh cho khu vực 'Ảnh đã chọn'."""
    imgs = []
    # Helper function to normalize item
    def normalize_item_to_path(item):
        return item[0] if isinstance(item, (list, tuple)) else item

    for i in sorted(selected_indices or []):
        if 0 <= i < len(gallery_items or []):
            path = normalize_item_to_path(gallery_items[i])
            if path:
                imgs.append(path)
    return imgs