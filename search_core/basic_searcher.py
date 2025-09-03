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
    
    def search(self, query_text: str, top_k: int = 10):
        query_vector = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vector.astype('float32'))
        distances, indices = self.index.search(query_vector.astype('float32'), k=top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                info = self.metadata.iloc[idx].to_dict()
                scale_factor = 0.5 
                info['clip_score'] = np.exp(-scale_factor * dist)
                results.append(info)
        return results
