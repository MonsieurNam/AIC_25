# /utils/cache_manager.py

import os
import pickle
import numpy as np
from typing import Dict, Optional

class ObjectVectorCache:
    """
    Quản lý việc cache các vector của object.
    - Tải cache từ đĩa khi khởi tạo.
    - Cung cấp truy cập nhanh trong bộ nhớ (dictionary).
    - Tự động lưu lại vào đĩa mỗi khi có một vector mới được thêm vào.
    """
    def __init__(self, cache_path: str = "/kaggle/working/object_vector_cache.pkl"):
        self.cache_path = cache_path
        self.cache: Dict[str, np.ndarray] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Tải cache từ file pickle nếu tồn tại."""
        if os.path.exists(self.cache_path):
            print(f"--- 🧠 Phát hiện cache vector object. Đang tải từ: {self.cache_path} ---")
            try:
                with open(self.cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"--- ✅ Tải thành công {len(self.cache)} vector đã tính toán trước. ---")
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"--- ⚠️ Lỗi khi tải cache: {e}. Bắt đầu với cache rỗng. ---")
                self.cache = {}
        else:
            print("--- 🧠 Không tìm thấy cache vector object. Sẽ tạo mới khi cần. ---")

    def get(self, key: str) -> Optional[np.ndarray]:
        """Lấy một vector từ cache. Trả về None nếu không tìm thấy."""
        return self.cache.get(key)

    def set(self, key: str, vector: np.ndarray):
        """Thêm một vector mới vào cache và lưu xuống đĩa."""
        self.cache[key] = vector
        self._save_to_disk()

    def _save_to_disk(self):
        """Lưu toàn bộ cache hiện tại xuống file pickle."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi lưu cache xuống đĩa: {e} ---")
            
    def __len__(self):
        return len(self.cache)