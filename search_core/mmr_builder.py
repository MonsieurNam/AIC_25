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
        if not candidates or self.clip_features_tensor is None:
            return candidates[:target_size]

        print(f"--- Bắt đầu xây dựng danh sách đa dạng bằng MMR (λ={lambda_val}, Chế độ Tối ưu & An toàn) ---")

        # === BƯỚC 1: TIỀN XỬ LÝ VÀ LỌC AN TOÀN ===
        # Chỉ giữ lại các ứng viên có 'original_index' hợp lệ
        max_index = self.clip_features_tensor.shape[0] - 1
        valid_candidates = [
            cand for cand in candidates
            if cand.get('original_index') is not None and 0 <= cand['original_index'] <= max_index
        ]
        if not valid_candidates:
             print("--- ⚠️ [MMR] Không có ứng viên nào có original_index hợp lệ. Trả về danh sách gốc.")
             return candidates[:target_size]
        
        # Chuyển đổi thành dictionary để truy cập nhanh
        candidates_pool = {i: cand for i, cand in enumerate(valid_candidates)}
        
        # === BƯỚC 2: KHỞI TẠO MMR ===
        final_pool_indices = []
        if not candidates_pool: return []
        
        # Chọn ứng viên đầu tiên dựa trên điểm số (từ pool đã lọc)
        best_initial_idx = max(candidates_pool, key=lambda idx: candidates_pool[idx]['final_score'])
        final_pool_indices.append(best_initial_idx)
        
        # === BƯỚC 3: VÒNG LẶP MMR TỐI ƯU HÓA ===
        while len(final_pool_indices) < min(target_size, len(valid_candidates)):
            remaining_pool_indices = list(set(candidates_pool.keys()) - set(final_pool_indices))
            if not remaining_pool_indices: break

            # Lấy vector của các kết quả đã chọn
            selected_original_indices = [candidates_pool[idx]['original_index'] for idx in final_pool_indices]
            selected_vectors_tensor = self.clip_features_tensor[selected_original_indices]

            # Lấy vector của các ứng viên còn lại
            remaining_original_indices = [candidates_pool[idx]['original_index'] for idx in remaining_pool_indices]
            remaining_vectors_tensor = self.clip_features_tensor[remaining_original_indices]
            
            # Tính toán tương đồng hàng loạt
            similarity_matrix = util.pytorch_cos_sim(remaining_vectors_tensor, selected_vectors_tensor)
            max_similarity_per_candidate, _ = torch.max(similarity_matrix, dim=1)
            
            # === BƯỚC 4: TÌM ỨNG VIÊN TỐT NHẤT TIẾP THEO ===
            best_mmr_score = -np.inf
            best_candidate_idx_to_add = -1
            
            for i, pool_idx in enumerate(remaining_pool_indices):
                relevance_score = candidates_pool[pool_idx]['final_score']
                max_sim = max_similarity_per_candidate[i].item()
                mmr_score = (lambda_val * relevance_score) - ((1 - lambda_val) * max_sim)
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate_idx_to_add = pool_idx
            
            if best_candidate_idx_to_add != -1:
                final_pool_indices.append(best_candidate_idx_to_add)
            else:
                # Không tìm thấy ứng viên nào tốt hơn, dừng lại
                break
        
        final_diverse_list = [candidates_pool[idx] for idx in final_pool_indices]
        print(f"--- ✅ Xây dựng danh sách MMR hoàn tất với {len(final_diverse_list)} kết quả. ---")
        
        return final_diverse_list