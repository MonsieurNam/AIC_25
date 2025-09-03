# /utils/spatial_engine.py

from typing import List, Tuple

# Định dạng bounding box mà chúng ta nhận được là [y1, x1, y2, x2]

def get_center(box: List[float]) -> Tuple[float, float]:
    """
    Tính toán tọa độ tâm của một bounding box.
    
    Args:
        box: Một list float theo định dạng [y1, x1, y2, x2].
    
    Returns:
        Một tuple (center_x, center_y).
    """
    center_y = (box[0] + box[2]) / 2
    center_x = (box[1] + box[3]) / 2
    return center_x, center_y

def is_between(box_a: List[float], box_b: List[float], box_c: List[float]) -> bool:
    """
    Kiểm tra xem box_a có nằm giữa box_b và box_c theo chiều ngang hay không.
    
    Args:
        box_a: Box cần kiểm tra.
        box_b: Một trong hai box mốc.
        box_c: Box mốc còn lại.
        
    Returns:
        True nếu tâm của A nằm giữa tâm của B và C trên trục X.
    """
    center_a_x, _ = get_center(box_a)
    center_b_x, _ = get_center(box_b)
    center_c_x, _ = get_center(box_c)
    
    return min(center_b_x, center_c_x) <= center_a_x <= max(center_b_x, center_c_x)

def is_behind(box_a: List[float], box_b: List[float]) -> bool:
    """
    Kiểm tra xem box_a có nằm phía sau (phía trên trong ảnh) box_b hay không.
    
    Args:
        box_a: Box được cho là ở phía sau.
        box_b: Box ở phía trước.
        
    Returns:
        True nếu tâm của A có tọa độ Y nhỏ hơn tâm của B.
    """
    _, center_a_y = get_center(box_a)
    _, center_b_y = get_center(box_b)
    
    return center_a_y < center_b_y

# Có thể thêm các hàm khác ở đây trong tương lai (is_left_of, is_above, etc.)