# /utils/spatial_engine.py

from typing import List, Tuple

# ==============================================================================
# === HỆ THỐNG NHẬN THỨC KHÔNG GIAN (SPATIAL AWARENESS SYSTEM) - V2.0 ===
# ==============================================================================
#
# Cung cấp các hàm toán học để phân tích mối quan hệ không gian giữa các
# bounding box. Tất cả các hàm đều hoạt động với tọa độ tương đối [y1, x1, y2, x2].
#
# (0,0) là góc trên cùng bên trái của ảnh.
# Trục Y tăng dần xuống dưới.
# Trục X tăng dần sang phải.
#
# ==============================================================================


# --- HÀM CƠ SỞ ---

def get_center(box: List[float]) -> Tuple[float, float]:
    """
    Tính toán tọa độ tâm của một bounding box.
    
    Args:
        box: Một list float theo định dạng [y1, x1, y2, x2].
    
    Returns:
        Một tuple (center_x, center_y).
    """
    if not (isinstance(box, list) and len(box) == 4):
        return (0.5, 0.5) # Trả về tâm ảnh nếu box không hợp lệ
    center_y = (box[0] + box[2]) / 2
    center_x = (box[1] + box[3]) / 2
    return center_x, center_y

def get_area(box: List[float]) -> float:
    """Tính diện tích của một bounding box."""
    if not (isinstance(box, list) and len(box) == 4):
        return 0.0
    height = box[2] - box[0]
    width = box[3] - box[1]
    return height * width

def get_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Tính chỉ số Intersection over Union (IoU) giữa hai box.
    IoU = Diện tích phần giao / Diện tích phần hợp.
    Trả về giá trị từ 0.0 (không giao) đến 1.0 (trùng khớp hoàn toàn).
    """
    # Xác định tọa độ của vùng giao nhau
    xA = max(box_a[1], box_b[1])
    yA = max(box_a[0], box_b[0])
    xB = min(box_a[3], box_b[3])
    yB = min(box_a[2], box_b[2])

    # Tính diện tích vùng giao
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Tính diện tích của từng box
    box_a_area = (box_a[3] - box_a[1]) * (box_a[2] - box_a[0])
    box_b_area = (box_b[3] - box_b[1]) * (box_b[2] - box_b[0])

    # Tính diện tích hợp
    union_area = box_a_area + box_b_area - intersection_area

    # Tính IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


# --- CÁC HÀM QUAN HỆ KHÔNG GIAN ---

def is_between(box_a: List[float], box_b: List[float], box_c: List[float], tolerance: float = 0.1) -> bool:
    """
    Kiểm tra xem box_a có nằm giữa box_b và box_c theo chiều ngang hay không,
    cho phép một khoảng sai số về chiều dọc.
    """
    center_a_x, center_a_y = get_center(box_a)
    center_b_x, center_b_y = get_center(box_b)
    center_c_x, center_c_y = get_center(box_c)
    
    # Điều kiện 1: Tâm của A phải nằm giữa B và C theo trục X.
    horizontal_check = min(center_b_x, center_c_x) <= center_a_x <= max(center_b_x, center_c_x)
    
    # Điều kiện 2: Tâm của A phải gần với đường thẳng nối B và C theo trục Y.
    # Tính trung bình tọa độ Y của B và C.
    avg_y = (center_b_y + center_c_y) / 2
    vertical_check = abs(center_a_y - avg_y) < tolerance
    
    return horizontal_check and vertical_check

def is_behind(box_a: List[float], box_b: List[float]) -> bool:
    """
    Kiểm tra xem box_a có nằm phía sau (phía trên trong ảnh) box_b hay không.
    Điều này hữu ích cho các cảnh có phối cảnh.
    "Phía sau" thường tương ứng với tọa độ Y nhỏ hơn.
    """
    _, center_a_y = get_center(box_a)
    _, center_b_y = get_center(box_b)
    
    # A được coi là "phía sau" B nếu tâm của nó cao hơn (Y nhỏ hơn) tâm của B.
    return center_a_y < center_b_y

def is_on(box_a: List[float], box_b: List[float]) -> bool:
    """
    Kiểm tra xem box_a có nằm "trên" box_b hay không.
    Logic tương tự 'is_behind' nhưng có thể được tinh chỉnh sau này.
    Ví dụ, 'cái ly trên cái bàn' cũng có thể hiểu là is_behind trong phối cảnh 2D.
    """
    return is_behind(box_a, box_b)

def is_above(box_a: List[float], box_b: List[float]) -> bool:
    """
    Kiểm tra xem box_a có nằm hoàn toàn phía trên box_b hay không (không chồng lấn).
    Cạnh dưới của A phải cao hơn cạnh trên của B.
    """
    return box_a[2] < box_b[0]

def is_below(box_a: List[float], box_b: List[float]) -> bool:
    """
    Kiểm tra xem box_a có nằm hoàn toàn phía dưới box_b hay không (không chồng lấn).
    Cạnh trên của A phải thấp hơn cạnh dưới của B.
    """
    return box_a[0] > box_b[2]

def is_next_to(box_a: List[float], box_b: List[float], tolerance_ratio: float = 0.5) -> bool:
    """
    Kiểm tra xem box_a có nằm cạnh box_b hay không.
    "Cạnh nhau" có nghĩa là chúng gần nhau theo chiều ngang và có sự chồng lấn đáng kể theo chiều dọc.
    
    Args:
        tolerance_ratio: Tỷ lệ chiều cao mà hai box phải chồng lấn để được coi là "cạnh nhau".
    """
    center_a_x, center_a_y = get_center(box_a)
    center_b_x, center_b_y = get_center(box_b)

    # 1. Tính khoảng cách ngang giữa hai tâm
    horizontal_distance = abs(center_a_x - center_b_x)
    # Tính tổng một nửa chiều rộng của hai box
    sum_half_widths = (box_a[3] - box_a[1]) / 2 + (box_b[3] - box_b[1]) / 2
    # Điều kiện gần nhau theo chiều ngang: khoảng cách tâm < tổng nửa chiều rộng
    is_horizontally_close = horizontal_distance < sum_half_widths

    # 2. Tính độ chồng lấn theo chiều dọc
    y_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    # Chiều cao của box nhỏ hơn
    min_height = min(box_a[2] - box_a[0], box_b[2] - box_b[0])
    # Điều kiện chồng lấn dọc: độ chồng lấn phải lớn hơn một ngưỡng
    is_vertically_aligned = y_overlap > min_height * tolerance_ratio

    return is_horizontally_close and is_vertically_aligned

def is_inside(box_a: List[float], box_b: List[float]) -> bool:
    """
    Kiểm tra xem box_a (nhỏ) có nằm hoàn toàn bên trong box_b (lớn) hay không.
    """
    return box_a[1] >= box_b[1] and box_a[3] <= box_b[3] and \
           box_a[0] >= box_b[0] and box_a[2] <= box_b[2]