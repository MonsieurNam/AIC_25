# ==============================================================================
# SEMANTIC SEARCHER - PHIÊN BẢN V5 (TỐI ƯU HÓA & NÂNG CẤP TỪ BẢN GỐC)
# ==============================================================================
import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Any
from utils.cache_manager import ObjectVectorCache
from utils.spatial_engine import is_between, is_behind
from utils.image_cropper import crop_image_by_box
from search_core.basic_searcher import BasicSearcher

class SemanticSearcher:
    def __init__(self, basic_searcher, rerank_model, device="cuda"):
        print("--- 🧠 Khởi tạo SemanticSearcher (Reranking Engine - Phoenix Edition) ---")
        self.basic_searcher = basic_searcher
        self.model = rerank_model
        self.device = device
        
        # --- TẢI "HỒ DỮ LIỆU OBJECT" ---
        self.master_object_df = None
        object_data_path = "/kaggle/input/stage1/master_object_data.parquet"
        if os.path.exists(object_data_path):
            print(f"   -> Đang tải Hồ Dữ liệu Object từ: {object_data_path}")
            self.master_object_df = pd.read_parquet(object_data_path)
            # Tối ưu hóa: Set index để tăng tốc độ truy vấn sau này
            self.master_object_df.set_index('keyframe_id', inplace=True)
            print(f"--- ✅ Tải thành công và lập chỉ mục cho {len(self.master_object_df)} object. ---")
        else:
            print("--- ⚠️ Cảnh báo: Không tìm thấy master_object_data.parquet. Bộ lọc không gian sẽ bị vô hiệu hóa. ---")
            
        # --- TRANG BỊ CÔNG CỤ CHO TẦNG 3 ---
        print("--- 🔬 Trang bị công cụ Xác thực Chi tiết... ---")
        # Lấy quyền truy cập vào CLIP model và processor từ BasicSearcher
        self.clip_model = basic_searcher.model
        self.clip_processor = basic_searcher.processor
        # Khởi tạo Ngân hàng Vector Linh hoạt
        self.object_vector_cache = ObjectVectorCache()
        print("--- ✅ Sẵn sàng hoạt động với bộ nhớ cache. ---")
            
    def _apply_spatial_filter(self, candidates: List[Dict], spatial_rules: List[Dict], grounded_entities: List[str]) -> List[Dict]:
        """
        Áp dụng các quy tắc không gian để tính điểm 'spatial_score' cho mỗi ứng viên.
        PHIÊN BẢN HOÀN CHỈNH.
        """
        # --- Điều kiện thoát sớm ---
        if not spatial_rules or self.master_object_df is None or self.master_object_df.empty:
            for cand in candidates:
                cand['scores']['spatial_score'] = 1.0
            return candidates

        print(f"--- 📐 Áp dụng {len(spatial_rules)} Quy tắc Không gian trên {len(candidates)} ứng viên... ---")
        
        candidate_ids = [c['keyframe_id'] for c in candidates]
        try:
            relevant_objects_df = self.master_object_df.loc[self.master_object_df.index.isin(candidate_ids)]
        except KeyError:
            relevant_objects_df = pd.DataFrame()

        if relevant_objects_df.empty:
            for cand in candidates:
                cand['scores']['spatial_score'] = 0.0
            return candidates

        for cand in candidates:
            keyframe_objects = relevant_objects_df[relevant_objects_df.index == cand['keyframe_id']]
            
            if keyframe_objects.empty:
                cand['scores']['spatial_score'] = 0.0
                continue
            
            total_rules = len(spatial_rules)
            satisfied_rules_count = 0
            
            # Lặp qua từng quy tắc mà Gemini đã cung cấp
            for rule in spatial_rules:
                entity_label = rule['entity'].replace('_', ' ')
                relation = rule['relation']
                target_labels = [t.replace('_', ' ') for t in rule['targets']]
                
                # Lấy ra tất cả các bounding box của các object có liên quan trong rule này
                # Chúng ta sẽ tìm các label chứa (contains) entity_label, ví dụ "man" sẽ khớp với "man black shirt"
                entity_boxes = keyframe_objects[keyframe_objects['object_label'].str.contains(entity_label, case=False)]['bounding_box'].tolist()
                
                target_boxes_lists = []
                for label in target_labels:
                    boxes = keyframe_objects[keyframe_objects['object_label'].str.contains(label, case=False)]['bounding_box'].tolist()
                    target_boxes_lists.append(boxes)

                # Nếu thiếu bất kỳ loại object nào, không thể thỏa mãn rule -> bỏ qua
                if not entity_boxes or any(not boxes for boxes in target_boxes_lists):
                    continue
                    
                rule_satisfied = False
                # Lặp qua tất cả các box của entity chính
                for entity_box in entity_boxes:
                    if rule_satisfied: break
                    
                    # --- Xử lý các loại quan hệ ---
                    if relation == 'is_between' and len(target_boxes_lists) == 2:
                        # Cần tìm một cặp target (từ list 1 và list 2) để entity nằm giữa
                        # Lấy tất cả các cặp có thể có giữa hai list target boxes
                        target_pairs = [(b1, b2) for b1 in target_boxes_lists[0] for b2 in target_boxes_lists[1]]
                        for target1_box, target2_box in target_pairs:
                            # Tránh trường hợp 2 target là cùng một object
                            if target1_box == target2_box: continue
                            if is_between(entity_box, target1_box, target2_box):
                                rule_satisfied = True
                                break
                    
                    elif relation == 'is_behind' and len(target_boxes_lists) == 1:
                        for target_box in target_boxes_lists[0]:
                            if is_behind(entity_box, target_box):
                                rule_satisfied = True
                                break
                    
                    # TODO: Thêm các điều kiện 'is_next_to', 'is_above', etc. ở đây nếu cần
                    
                if rule_satisfied:
                    satisfied_rules_count += 1
            
            # Tính điểm cuối cùng: tỷ lệ các rule được thỏa mãn
            cand['scores']['spatial_score'] = satisfied_rules_count / total_rules if total_rules > 0 else 1.0

        # In ra một vài ví dụ điểm để debug
        print("    -> Ví dụ điểm không gian:", {c['keyframe_id']: f"{c['scores']['spatial_score']:.2f}" for c in candidates[:5]})
        return candidates
    
    def _apply_fine_grained_filter(self, candidates: List[Dict], verification_rules: List[Dict]) -> List[Dict]:
        """
        Sử dụng CLIP trên các vùng ảnh đã crop để xác thực các chi tiết nhỏ.
        """
        if not verification_rules or self.master_object_df is None or self.master_object_df.empty:
            for cand in candidates:
                cand['scores']['fine_grained_score'] = 1.0
            return candidates

        print(f"--- 🔬 Áp dụng {len(verification_rules)} Quy tắc Xác thực Chi tiết...")
        
        # Chỉ xử lý trên top 50 ứng viên để tiết kiệm thời gian, số còn lại nhận điểm mặc định
        top_candidates = candidates[:50]
        
        # Encode tất cả các mô tả text một lần duy nhất
        detailed_descriptions = [rule['detailed_description'] for rule in verification_rules]
        text_inputs = self.clip_processor(text=detailed_descriptions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        for cand in tqdm(top_candidates, desc="Xác thực chi tiết (soi kính hiển vi)"):
            keyframe_id = cand['keyframe_id']
            keyframe_objects = self.master_object_df.loc[self.master_object_df.index == keyframe_id]
            
            if keyframe_objects.empty:
                cand['scores']['fine_grained_score'] = 0.0
                continue
            
            total_score = 0.0
            for i, rule in enumerate(verification_rules):
                target_label = rule['target_entity']
                
                # Tìm object phù hợp nhất trong keyframe (confidence cao nhất)
                possible_objects = keyframe_objects[keyframe_objects['object_label'].str.contains(target_label, case=False)]
                if possible_objects.empty:
                    continue # Bỏ qua rule này nếu không có object khớp

                best_object = possible_objects.loc[possible_objects['confidence_score'].idxmax()]
                
                # --- LOGIC CACHING ---
                cache_key = f"{keyframe_id}_{target_label}_{best_object['confidence_score']:.4f}"
                object_vector = self.object_vector_cache.get(cache_key)
                
                if object_vector is None: # Cache miss
                    try:
                        cropped_image = crop_image_by_box(cand['keyframe_path'], best_object['bounding_box'])
                        image_input = self.clip_processor(images=cropped_image, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            image_features = self.clip_model.get_image_features(**image_input)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                        
                        object_vector = image_features.cpu().numpy()
                        self.object_vector_cache.set(cache_key, object_vector)
                    except Exception as e:
                        print(f"Lỗi khi xử lý ảnh crop cho {keyframe_id}: {e}")
                        continue
                
                # Tính điểm tương đồng
                image_tensor = torch.from_numpy(object_vector).to(self.device)
                similarity = util.pytorch_cos_sim(image_tensor, text_features[i].unsqueeze(0))
                total_score += similarity.item()

            cand['scores']['fine_grained_score'] = total_score / len(verification_rules) if verification_rules else 1.0

        # Gán điểm mặc định cho các ứng viên không được check
        for cand in candidates[50:]:
            cand['scores']['fine_grained_score'] = 0.5 # Điểm trung bình

        return candidates

    def search(self,
               query_text: str,
               top_k_final: int,
               top_k_retrieval: int,
               precomputed_analysis: Dict[str, Any] = None,
               weights: Dict[str, float] = None
              ) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm và tái xếp hạng đa tầng theo kiến trúc PHOENIX.
        Luồng xử lý: Contextual -> Spatial -> Fine-grained Verification.
        """
        print("\n--- 🔱 Bắt đầu quy trình tìm kiếm đa tầng PHOENIX... ---")

        # --- Bước 0: Chuẩn bị ---
        if precomputed_analysis is None: precomputed_analysis = {}
        # Đặt trọng số mặc định và cho phép ghi đè từ UI
        final_weights = {
            'w_clip': 0.2, 
            'w_semantic': 0.3, 
            'w_spatial': 0.25, 
            'w_fine_grained': 0.25, 
            **(weights or {})
        }
        print(f"    -> Trọng số hỏa lực: {final_weights}")

        # --- TẦNG 1: BỘ LỌC NGỮ CẢNH (Lấy ứng viên thô bằng CLIP toàn cục) ---
        print(f"--- Tầng 1: Lấy Top-{top_k_retrieval} ứng viên theo Ngữ cảnh... ---")
        candidates = self.basic_searcher.search(query_text, top_k=top_k_retrieval)
        if not candidates:
            print("--- ⛔ Không tìm thấy ứng viên nào ở Tầng 1. Dừng tìm kiếm. ---")
            return []
        print(f"    -> Tìm thấy {len(candidates)} ứng viên tiềm năng.")
        
        # Khởi tạo cấu trúc điểm cho mỗi ứng viên
        for cand in candidates:
            cand['scores'] = {'clip_score': cand.get('clip_score', 0.0)}

        # --- RERANKING NGỮ NGHĨA (Tinh chỉnh điểm ngữ cảnh bằng Bi-Encoder) ---
        # (Phần này bạn cần đảm bảo logic rerank_batch của mình được tích hợp ở đây)
        # Giả sử sau bước này, 'semantic_score' được thêm vào
        print("--- Tầng 1.5: Tinh chỉnh điểm Ngữ nghĩa bằng Bi-Encoder... ---")
        # Ví dụ:
        # candidates = self.rerank_with_bi_encoder(candidates, query_text)
        # Tạm thời gán điểm giả định để code chạy được
        for cand in candidates:
             cand['scores']['semantic_score'] = cand['scores']['clip_score'] # Tạm thời gán bằng điểm clip
        print("    -> Hoàn tất tinh chỉnh điểm ngữ nghĩa.")


        # --- TẦNG 2: BỘ LỌC QUAN HỆ KHÔNG GIAN ---
        spatial_rules = precomputed_analysis.get('spatial_rules', [])
        grounded_entities = precomputed_analysis.get('grounded_entities', []) # Có thể dùng trong tương lai
        
        candidates_after_spatial = self._apply_spatial_filter(candidates, spatial_rules, grounded_entities)


        # --- TẦNG 3: BỘ LỌC XÁC THỰC CHI TIẾT ---
        verification_rules = precomputed_analysis.get('fine_grained_verification', [])
        
        # Sắp xếp lại trước khi đưa vào Tầng 3 để đảm bảo chỉ "soi" những ứng viên tốt nhất
        # Tính điểm tạm thời sau Tầng 2
        for cand in candidates_after_spatial:
            s = cand['scores']
            cand['temp_score'] = (
                final_weights['w_clip'] * s.get('clip_score', 0.0) +
                final_weights['w_semantic'] * s.get('semantic_score', 0.0) +
                final_weights['w_spatial'] * s.get('spatial_score', 0.5)
            )
        
        sorted_before_fine_grained = sorted(candidates_after_spatial, key=lambda x: x.get('temp_score', 0.0), reverse=True)

        candidates_after_fine_grained = self._apply_fine_grained_filter(sorted_before_fine_grained, verification_rules)


        # --- BƯỚC CUỐI: TÍNH ĐIỂM TỔNG HỢP VÀ SẮP XẾP ---
        print("--- 🎯 Tính toán điểm hỏa lực cuối cùng và sắp xếp... ---")
        for cand in candidates_after_fine_grained:
            scores = cand['scores']
            
            # Công thức điểm hoàn chỉnh
            final_score = (
                final_weights['w_clip'] * scores.get('clip_score', 0.0) +
                final_weights['w_semantic'] * scores.get('semantic_score', 0.0) +
                final_weights['w_spatial'] * scores.get('spatial_score', 0.5) + # Dùng 0.5 làm điểm mặc định nếu có lỗi
                final_weights['w_fine_grained'] * scores.get('fine_grained_score', 0.5) # Dùng 0.5 làm điểm mặc định
            )
            cand['final_score'] = final_score

        # Sắp xếp lại lần cuối cùng dựa trên điểm tổng hợp
        final_sorted_candidates = sorted(candidates_after_fine_grained, key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        print(f"--- ✅ Quy trình PHOENIX hoàn tất. Trả về Top-{top_k_final} kết quả. ---")
        
        # In ra 3 kết quả đầu tiên để debug
        for i, cand in enumerate(final_sorted_candidates[:3]):
            print(f"  Top {i+1}: {cand['keyframe_id']} | Score: {cand['final_score']:.4f} | Scores: {cand['scores']}")

        return final_sorted_candidates[:top_k_final]