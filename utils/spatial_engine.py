# /utils/spatial_engine.py

from typing import List, Tuple

def get_center(box: List[float]) -> Tuple[float, float]:
    """
    Tính toán tọa độ tâm của một bounding box.
    
    Args:
        box: Một list float theo định dạng [y1, x1, y2, x2].
    
    Returns:
        Một tuple (center_x, center_y).
    """
    if not (isinstance(box, list) and len(box) == 4):
        return (0.5, 0.5) 
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
    xA = max(box_a[1], box_b[1])
    yA = max(box_a[0], box_b[0])
    xB = min(box_a[3], box_b[3])
    yB = min(box_a[2], box_b[2])

    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[3] - box_a[1]) * (box_a[2] - box_a[0])
    box_b_area = (box_b[3] - box_b[1]) * (box_b[2] - box_b[0])
    union_area = box_a_area + box_b_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou



def is_between(box_a: List[float], box_b: List[float], box_c: List[float], tolerance: float = 0.1) -> bool:
    """
    Kiểm tra xem box_a có nằm giữa box_b và box_c theo chiều ngang hay không,
    cho phép một khoảng sai số về chiều dọc.
    """
    center_a_x, center_a_y = get_center(box_a)
    center_b_x, center_b_y = get_center(box_b)
    center_c_x, center_c_y = get_center(box_c)
    
    horizontal_check = min(center_b_x, center_c_x) <= center_a_x <= max(center_b_x, center_c_x)
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

    horizontal_distance = abs(center_a_x - center_b_x)
    sum_half_widths = (box_a[3] - box_a[1]) / 2 + (box_b[3] - box_b[1]) / 2
    is_horizontally_close = horizontal_distance < sum_half_widths
    y_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    min_height = min(box_a[2] - box_a[0], box_b[2] - box_b[0])
    is_vertically_aligned = y_overlap > min_height * tolerance_ratio

    return is_horizontally_close and is_vertically_aligned

def is_inside(box_a: List[float], box_b: List[float]) -> bool:
    """
    Kiểm tra xem box_a (nhỏ) có nằm hoàn toàn bên trong box_b (lớn) hay không.
    """
    return box_a[1] >= box_b[1] and box_a[3] <= box_b[3] and \
           box_a[0] >= box_b[0] and box_a[2] <= box_b[2]