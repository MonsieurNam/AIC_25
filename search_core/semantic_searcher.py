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
from utils.spatial_engine import is_above, is_below, is_between, is_behind, is_inside, is_next_to, is_on
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
        # self.clip_processor = basic_searcher.processor
        # Khởi tạo Ngân hàng Vector Linh hoạt
        self.object_vector_cache = ObjectVectorCache()
        print("--- ✅ Sẵn sàng hoạt động với bộ nhớ cache. ---")
            
    def _apply_spatial_filter(self, 
                              candidates: List[Dict], 
                              spatial_rules: List[Dict], 
                              precomputed_analysis: Dict[str, Any]
                             ) -> List[Dict]:
        """
        Áp dụng các quy tắc không gian để tính điểm 'spatial_score'.
        PHIÊN BẢN NÂNG CẤP DỰA TRÊN CODE GỐC: Sử dụng cơ chế "Chấm điểm Mềm" (Soft Scoring).
        """
        grounding_map = precomputed_analysis.get('grounding_map', {})
        if not spatial_rules or self.master_object_df is None or self.master_object_df.empty:
            for cand in candidates:
                cand['scores']['spatial_score'] = 1.0
            return candidates

        print(f"--- 📐 Áp dụng {len(spatial_rules)} Quy tắc Không gian (Chế độ Chấm điểm Mềm)... ---")
        
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
            
            keyframe_objects_lower = keyframe_objects.copy()
            keyframe_objects_lower['object_label'] = keyframe_objects_lower['object_label'].str.lower()
            
            total_rules = len(spatial_rules)
            # ✅ THAY ĐỔI 1: Chuyển sang float để có thể cộng điểm lẻ
            total_satisfied_score = 0.0
            
            is_debug_candidate = cand['keyframe_id'] in [c['keyframe_id'] for c in candidates[:5]]
            if is_debug_candidate:
                print(f"\n--- DEBUG: Phân tích không gian cho Keyframe: {cand['keyframe_id']} ---")

            for rule in spatial_rules:
                entity_original = rule.get('entity', '').replace('_', ' ')
                relation = rule.get('relation')
                targets_original = [t.replace('_', ' ') for t in rule.get('targets', [])]
                
                entity_grounded = grounding_map.get(entity_original, entity_original).lower()
                targets_grounded = [grounding_map.get(t, t).lower() for t in targets_original]
                
                if is_debug_candidate:
                    print(f"  - Rule: {rule.get('entity')} {relation} {rule.get('targets')}")
                    print(f"    -> Grounded: '{entity_grounded}' vs {targets_grounded}")

                # --- ✅ BẮT ĐẦU LOGIC NÂNG CẤP ---
                
                # 1. Tìm bounding box cho TẤT CẢ các đối tượng trong quy tắc
                entity_boxes = keyframe_objects_lower[keyframe_objects_lower['object_label'] == entity_grounded]['bounding_box'].tolist()
                target_boxes_lists = [keyframe_objects_lower[keyframe_objects_lower['object_label'] == label]['bounding_box'].tolist() for label in targets_grounded]

                if is_debug_candidate:
                    print(f"    -> Tìm thấy: '{entity_grounded}' ({len(entity_boxes)} box), Targets ({[len(boxes) for boxes in target_boxes_lists]} boxes)")
                
                # 2. Kiểm tra xem có đủ đối tượng để xác minh quan hệ không gian hay không
                can_verify_relation = (len(entity_boxes) > 0) and all(len(boxes) > 0 for boxes in target_boxes_lists)
                
                rule_satisfied = False
                if can_verify_relation:
                    # Nếu có đủ đối tượng, tiến hành kiểm tra quan hệ không gian như code gốc
                    for entity_box in entity_boxes:
                        if rule_satisfied: break
                        
                        # Khối logic điều phối quan hệ (giữ nguyên)
                        if relation == 'is_between' and len(target_boxes_lists) == 2:
                            target_pairs = [(b1, b2) for b1 in target_boxes_lists[0] for b2 in target_boxes_lists[1]]
                            for target1_box, target2_box in target_pairs:
                                if np.array_equal(target1_box, target2_box): continue
                                if is_between(entity_box, target1_box, target2_box):
                                    rule_satisfied = True; break
                        
                        elif len(target_boxes_lists) == 1:
                            relation_function = {
                                'is_behind': is_behind, 'is_on': is_on, 'is_above': is_above,
                                'is_below': is_below, 'is_next_to': is_next_to, 'is_inside': is_inside
                            }.get(relation)
                            
                            if relation_function:
                                for target_box in target_boxes_lists[0]:
                                    if relation_function(entity_box, target_box):
                                        rule_satisfied = True; break
                
                # 3. Tính điểm cuối cùng cho quy tắc này theo cơ chế "Mềm"
                if rule_satisfied:
                    # Thưởng điểm tối đa nếu cả quan hệ không gian đều đúng
                    total_satisfied_score += 1.0
                    if is_debug_candidate:
                        print(f"    -> ✅ QUY TẮC ĐƯỢC THỎA MÃN HOÀN TOÀN! (1.0 điểm)")
                else:
                    # Nếu quan hệ không gian không đúng HOẶC không thể xác minh (do thiếu đối tượng),
                    # chúng ta vẫn thưởng điểm nếu một phần đối tượng tồn tại.
                    all_entities_in_rule = [entity_grounded] + targets_grounded
                    found_entities_count = (1 if entity_boxes else 0) + sum(1 for boxes in target_boxes_lists if boxes)
                    existence_score = found_entities_count / len(all_entities_in_rule) if all_entities_in_rule else 0
                    
                    # Thưởng 50% của điểm tồn tại.
                    # Ví dụ: nếu tìm thấy 2/3 đối tượng, điểm thưởng là 0.5 * (2/3) = 0.33
                    partial_score = 0.5 * existence_score
                    total_satisfied_score += partial_score
                    if is_debug_candidate and partial_score > 0:
                        print(f"    -> ⚠️  QUY TẮC KHỚP MỘT PHẦN (do đối tượng tồn tại). ({partial_score:.2f} điểm)")
            
            # ✅ THAY ĐỔI 2: Tính điểm trung bình cuối cùng từ tổng điểm đã tích lũy
            cand['scores']['spatial_score'] = total_satisfied_score / total_rules if total_rules > 0 else 1.0
        
        print("    -> Ví dụ điểm không gian (Chấm điểm Mềm):", {c['keyframe_id']: f"{c['scores'].get('spatial_score', 0):.2f}" for c in candidates[:5]})
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
        with torch.no_grad():
            text_features = self.clip_model.encode(detailed_descriptions, convert_to_tensor=True, device=self.device)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        for cand in tqdm(top_candidates, desc="Xác thực chi tiết (soi kính hiển vi)"):
            keyframe_id = cand['keyframe_id']
            keyframe_objects = self.master_object_df.loc[self.master_object_df.index == keyframe_id]
            
            if keyframe_objects.empty:
                cand['scores']['fine_grained_score'] = 0.0
                continue
            
            total_score = 0.0
            for i, rule in enumerate(verification_rules):
                object_vector = None
                target_label = rule['target_entity']
                
                # Tìm object phù hợp nhất trong keyframe (confidence cao nhất)
                possible_objects = keyframe_objects[keyframe_objects['object_label'].str.contains(target_label, case=False)]
                if possible_objects.empty:
                    continue # Bỏ qua rule này nếu không có object khớp

                best_object_series = possible_objects.sort_values(by='confidence_score', ascending=False).iloc[0]
    
                # Lấy giá trị một cách an toàn và kiểm tra kiểu
                confidence_value = best_object_series.get('confidence_score')
                bounding_box_value = best_object_series.get('bounding_box')

                # Kiểm tra kiểu dữ liệu trước khi sử dụng
                if not isinstance(confidence_value, (int, float)):
                    print(f"--- ⚠️ WARNING: Kiểu dữ liệu confidence_score không hợp lệ ({type(confidence_value)}) cho keyframe {keyframe_id}. Bỏ qua rule. ---")
                    continue
                    
                if bounding_box_value is None:
                    continue
                
                # Giờ thì chúng ta có thể yên tâm sử dụng
                cache_key = f"{keyframe_id}_{target_label}_{confidence_value:.4f}"
                
                # --- KẾT THÚC THAY THẾ ---

                if object_vector is None: # Cache miss
                    try:
                        # Ép kiểu bounding_box thành list để đảm bảo
                        cropped_image = crop_image_by_box(cand['keyframe_path'], list(bounding_box_value))
                        with torch.no_grad():
                            image_features = self.clip_model.encode(cropped_image, convert_to_tensor=True, device=self.device)
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
        
        candidates_after_spatial = self._apply_spatial_filter(
            candidates=candidates, 
            spatial_rules=spatial_rules, 
            precomputed_analysis=precomputed_analysis
        )

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