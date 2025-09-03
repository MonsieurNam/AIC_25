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
            # --- 1. Tải FAISS Index ---
            print(f"   -> Đang tải FAISS index từ: {faiss_index_path}")
            self.index = faiss.read_index(faiss_index_path)
            
            # --- 2. Tải Metadata ---
            print(f"   -> Đang tải metadata từ: {metadata_path}")
            # Chỉ tải các cột cần thiết để tiết kiệm RAM tối đa
            self.metadata = pd.read_parquet(
                metadata_path, 
                columns=['keyframe_id', 'video_id', 'timestamp', 'keyframe_path']
            )
            print(f"--- ✅ Tải thành công {self.index.ntotal} vector và metadata tương ứng. ---")
            
            # --- 3. Tải CLIP Model ---
            print(f"   -> Đang tải CLIP model: {clip_model_name} lên {self.device}")
            # Sử dụng SentenceTransformer để có API tiện lợi cho cả model và processor
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

        # 1. Mã hóa truy vấn văn bản thành vector embedding
        #    Sử dụng model đã được tải sẵn trong __init__
        query_embedding = self.model.encode(
            query_text, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False # Tắt progress bar để log gọn gàng hơn
        )
        
        # Chuyển về numpy và reshape để phù hợp với input của FAISS
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        
        # 2. Chuẩn hóa L2 (bắt buộc cho tìm kiếm tương đồng cosine trên FAISS)
        faiss.normalize_L2(query_embedding_np)

        # 3. Tìm kiếm trên FAISS index
        #    `search` trả về (distances, indices)
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        # 4. Lấy thông tin metadata và định dạng kết quả trả về
        results = []
        result_indices = indices[0]
        result_distances = distances[0]
        
        for i in range(len(result_indices)):
            idx = result_indices[i]
            
            # Lấy thông tin từ metadata bằng index (iloc rất nhanh)
            meta_info = self.metadata.iloc[idx].to_dict()
            
            # Thêm điểm số và index gốc vào kết quả
            meta_info['clip_score'] = float(result_distances[i])
            meta_info['original_index'] = int(idx) # Dùng cho MMR
            
            results.append(meta_info)
            
        return results