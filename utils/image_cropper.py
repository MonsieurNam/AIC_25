# /utils/image_cropper.py
from PIL import Image
from typing import List

def crop_image_by_box(image_path: str, box: List[float]) -> Image.Image:
    """
    Mở một ảnh và cắt ra một vùng dựa trên bounding box.
    
    Args:
        image_path: Đường dẫn đến file ảnh.
        box: Bounding box theo định dạng [y1, x1, y2, x2] với tọa độ tương đối (0-1).
        
    Returns:
        Một đối tượng PIL.Image đã được cắt.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        # Chuyển tọa độ tương đối thành tọa độ pixel tuyệt đối
        left = box * width
        top = box * height
        right = box * width
        bottom = box * height
        
        return img.crop((left, top, right, bottom))