from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import util
import faiss 

class MMRResultBuilder:
    """
    Xây dựng lại danh sách kết quả cuối cùng bằng thuật toán Maximal Marginal Relevance (MMR)
    để tăng cường sự đa dạng.
    PHIÊN BẢN V2: Tối ưu hóa tốc độ bằng cách tính toán tương đồng hàng loạt (batched).
    """
    def __init__(self, clip_features: np.ndarray, device: str = "cuda"):
        """
        Khởi tạo MMRResultBuilder. (Logic không đổi)
        """
        print("--- 🎨 Khởi tạo MMR Result Builder (Diversity Engine) ---")
        self.device = device
        try:
            print(f"   -> Đang chuyển ma trận vector CLIP sang tensor trên {self.device}...")
            features_copy = np.ascontiguousarray(clip_features.astype('float32'))
            faiss.normalize_L2(features_copy)
            self.clip_features_tensor = torch.from_numpy(features_copy).to(self.device)
            print(f"--- ✅ Chuyển đổi thành công {self.clip_features_tensor.shape[0]} vector CLIP. ---")
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi xử lý vector CLIP: {e}. MMR sẽ bị vô hiệu hóa. ---")
            import traceback
            traceback.print_exc()
            self.clip_features_tensor = None

    def build_diverse_list(self, 
                           candidates: List[Dict], 
                           target_size: int, 
                           lambda_val: float = 0.7
                          ) -> List[Dict]:
        """
        Xây dựng danh sách kết quả đa dạng bằng thuật toán MMR.
        PHIÊN BẢN TỐI ƯU HÓA.
        """
        if not candidates or self.clip_features_tensor is None:
            return candidates[:target_size]

        print(f"--- Bắt đầu xây dựng danh sách đa dạng bằng MMR (λ={lambda_val}, Chế độ Tối ưu) ---")
        candidates_pool = {i: cand for i, cand in enumerate(candidates)}
        for i, cand in enumerate(candidates):
            candidates_pool[i]['original_index'] = cand.get('index')
        final_results_indices = []
        if not candidates_pool: return []
        best_initial_idx = max(candidates_pool, key=lambda idx: candidates_pool[idx]['final_score'])
        final_results_indices.append(best_initial_idx)
        while len(final_results_indices) < min(target_size, len(candidates)):
            best_mmr_score = -np.inf
            best_candidate_idx = -1
            remaining_indices = set(candidates_pool.keys()) - set(final_results_indices)
            if not remaining_indices: break
            selected_original_indices = [
                candidates_pool[idx]['original_index'] for idx in final_results_indices
                if candidates_pool[idx].get('original_index') is not None
            ]
            if not selected_original_indices: 
                break
            selected_vectors_tensor = self.clip_features_tensor[selected_original_indices]
            remaining_original_indices = [
                candidates_pool[idx]['original_index'] for idx in remaining_indices
                if candidates_pool[idx].get('original_index') is not None
            ]
            if not remaining_original_indices:
                break
            remaining_vectors_tensor = self.clip_features_tensor[remaining_original_indices]
            similarity_matrix = util.pytorch_cos_sim(remaining_vectors_tensor, selected_vectors_tensor)
            max_similarity_per_candidate = torch.max(similarity_matrix, dim=1).values
            for i, cand_idx in enumerate(remaining_indices):
                relevance_score = candidates_pool[cand_idx]['final_score']
                max_sim = max_similarity_per_candidate[i].item()
                mmr_score = (lambda_val * relevance_score) - ((1 - lambda_val) * max_sim)
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate_idx = cand_idx
            if best_candidate_idx != -1:
                final_results_indices.append(best_candidate_idx)
            else:
                break
        final_diverse_list = [candidates_pool[idx] for idx in final_results_indices]
        print(f"--- ✅ Xây dựng danh sách MMR hoàn tất với {len(final_diverse_list)} kết quả. ---")
        
        return final_diverse_list