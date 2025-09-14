# /search_core/basic_searcher.py

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class BasicSearcher:
    """
    BasicSearcher: Động cơ Retrieval Nền tảng (Core Retrieval Engine).
    
    Chịu trách nhiệm cho tầng tìm kiếm đầu tiên và nhanh nhất.
    - Tải và quản lý FAISS index để tìm kiếm vector tốc độ cao.
    - Tải và quản lý metadata tương ứng.
    - Tải và quản lý model CLIP để mã hóa các truy vấn văn bản thành vector.
    
    Kiến trúc PHOENIX.
    """
    def __init__(self, 
                 faiss_index_path: str, 
                 metadata_path: str, 
                 clip_model_name: str = 'clip-ViT-B-32',
                 device: str = "cuda"):
        """
        Khởi tạo BasicSearcher.
        Tải tất cả các tài nguyên cần thiết vào bộ nhớ.
        """
        print("--- 🔍 Khởi tạo BasicSearcher (Core Retrieval Engine - Phoenix Edition)... ---")
        self.device = device
        try:
            print(f"   -> Đang tải FAISS index từ: {faiss_index_path}")
            self.index = faiss.read_index(faiss_index_path)
            print(f"   -> Đang tải metadata từ: {metadata_path}")
            self.metadata = pd.read_parquet(
                metadata_path, 
                columns=['keyframe_id', 'video_id', 'timestamp', 'keyframe_path']
            )
            print(f"--- ✅ Tải thành công {self.index.ntotal} vector và metadata tương ứng. ---")
            print(f"   -> Đang tải CLIP model: {clip_model_name} lên {self.device}")
            self.model = SentenceTransformer(clip_model_name, device=self.device)
            print("--- ✅ Tải CLIP model thành công. BasicSearcher sẵn sàng hoạt động! ---")
        except FileNotFoundError as e:
            print(f"--- ❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file cần thiết: {e}. ---")
            print("    -> Hãy chắc chắn rằng các đường dẫn trong config.py là chính xác.")
            raise e
        except Exception as e:
            print(f"--- ❌ Lỗi không xác định khi khởi tạo BasicSearcher: {e} ---")
            raise e

    def search(self, query_text: str, top_k: int) -> List[Dict]:
        """
        Thực hiện tìm kiếm vector trên FAISS index.

        Args:
            query_text (str): Chuỗi văn bản truy vấn.
            top_k (int): Số lượng kết quả gần nhất cần tìm.

        Returns:
            List[Dict]: Một danh sách các dictionary, mỗi cái chứa thông tin của một keyframe.
        """
        if not query_text or not query_text.strip():
            return []
        query_embedding = self.model.encode(
            query_text, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False 
        )
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(query_embedding_np)
        distances, indices = self.index.search(query_embedding_np, top_k)
        results = []
        result_indices = indices[0]
        result_distances = distances[0]
        for i in range(len(result_indices)):
            idx = result_indices[i]
            meta_info = self.metadata.iloc[idx].to_dict()
            meta_info['clip_score'] = float(result_distances[i])
            meta_info['original_index'] = int(idx)
            results.append(meta_info)
        return results