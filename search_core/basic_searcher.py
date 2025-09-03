import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
class BasicSearcher:
    def __init__(self, faiss_index_path, metadata_path, clip_model_name='clip-ViT-L-14'):
        """
        Khởi tạo BasicSearcher - Tầng retrieval nền tảng.
        PHIÊN BẢN NÂNG CẤP: Tải và lưu trữ model/processor.
        """
        print("--- 🔍 Khởi tạo BasicSearcher (Core Retrieval Engine)... ---")
        try:
            print(f"   -> Đang tải FAISS index từ: {faiss_index_path}")
            self.index = faiss.read_index(faiss_index_path)
            print(f"   -> Đang tải metadata từ: {metadata_path}")
            self.metadata = pd.read_parquet(metadata_path, columns=['keyframe_id', 'video_id', 'timestamp', 'keyframe_path', 'video_path'])
            print(f"--- ✅ Tải thành công {self.index.ntotal} vector và metadata. ---")
            
            print(f"   -> Đang tải CLIP model: {clip_model_name}")
            self.model = SentenceTransformer(clip_model_name, device='cuda')
            print("--- ✅ Tải CLIP model thành công. ---")
            
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi khởi tạo BasicSearcher: {e} ---")
            raise e

    def get_all_clip_features(self) -> np.ndarray: # <-- PHƯƠNG THỨC MỚI
        """Trả về ma trận NumPy của tất cả các vector CLIP."""
        return self.clip_features_numpy
    
    def search(self, query_text: str, top_k: int) -> list:
        """
        Thực hiện tìm kiếm vector trên FAISS.
        """
        if not query_text:
            return []

        # 1. Mã hóa query text thành vector sử dụng model đã được tải sẵn
        query_embedding = self.model.encode(query_text, convert_to_tensor=True, device=self.device)
        query_embedding = query_embedding.cpu().numpy().reshape(1, -1)
        
        # Chuẩn hóa L2 (quan trọng để so sánh cosine)
        faiss.normalize_L2(query_embedding)

        # 2. Tìm kiếm trên FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 3. Lấy thông tin metadata và trả về kết quả
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            # Lấy thông tin từ metadata bằng index
            meta_info = self.metadata.iloc[idx].to_dict()
            meta_info['clip_score'] = float(distances[0][i])
            # Thêm index gốc để MMR có thể dùng nếu cần
            meta_info['original_index'] = idx
            results.append(meta_info)
            
        return results