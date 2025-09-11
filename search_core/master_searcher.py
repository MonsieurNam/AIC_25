# search_core/master_searcher.py

from collections import defaultdict
from typing import Dict, Any, Optional, List
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# Import các module cốt lõi của hệ thống
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.trake_solver import TRAKESolver
from search_core.gemini_text_handler import GeminiTextHandler
from search_core.openai_handler import OpenAIHandler
from search_core.task_analyzer import TaskType
from search_core.mmr_builder import MMRResultBuilder 
from search_core.query_decomposer import QueryDecomposer

class MasterSearcher:
    """
    Lớp điều phối chính của hệ thống tìm kiếm (Hybrid AI Edition).
    Nó quản lý và điều phối các AI Handler khác nhau (Gemini cho text, OpenAI cho vision)
    để giải quyết các loại truy vấn phức tạp.
    """

    def __init__(self, 
                 basic_searcher: BasicSearcher, 
                 rerank_model,
                 gemini_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 entities_path: str = None,
                 clip_features_path: str = None,
                 video_path_map: dict = None):
        """
        Khởi tạo MasterSearcher và hệ sinh thái AI lai.
        """
        print("--- 🧠 Khởi tạo Master Searcher (Hybrid AI Edition) ---")
        
        self.semantic_searcher = SemanticSearcher(basic_searcher=basic_searcher, rerank_model=rerank_model)
        self.mmr_builder: Optional[MMRResultBuilder] = None
        if clip_features_path and os.path.exists(clip_features_path):
            try:
                print(f"--- 🚚 Đang tải toàn bộ CLIP features cho MMR từ: {clip_features_path} ---")
                all_clip_features = np.load(clip_features_path)
                self.mmr_builder = MMRResultBuilder(clip_features=all_clip_features)
            except Exception as e:
                 print(f"--- ⚠️ Lỗi khi khởi tạo MMR Builder: {e}. MMR sẽ bị vô hiệu hóa. ---")
        else:
            print("--- ⚠️ Không tìm thấy file CLIP features, MMR sẽ không hoạt động. ---")
        self.video_path_map = video_path_map
        self.gemini_handler: Optional[GeminiTextHandler] = None
        self.openai_handler: Optional[OpenAIHandler] = None
        self.trake_solver: Optional[TRAKESolver] = None
        self.query_decomposer: Optional[QueryDecomposer] = None
        self.ai_enabled = False
        self.known_entities: set = set()
        print(f"--- ✅ Master Searcher đã sẵn sàng! (AI Enabled: {self.ai_enabled}) ---")
        
        if entities_path and os.path.exists(entities_path):
            try:
                print(f"--- 📚 Đang tải Từ điển Đối tượng từ: {entities_path} ---")
                with open(entities_path, 'r') as f:
                    entities_list = [entity.lower() for entity in json.load(f)]
                    self.known_entities = set(entities_list)
                print(f"--- ✅ Tải thành công {len(self.known_entities)} thực thể đã biết. ---")
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi tải Từ điển Đối tượng: {e}. Semantic Grounding sẽ bị vô hiệu hóa. ---")
                
        # --- Khởi tạo và xác thực Gemini Handler cho các tác vụ TEXT ---
        if gemini_api_key:
            try:
                self.gemini_handler = GeminiTextHandler(api_key=gemini_api_key)
                if self.known_entities and self.gemini_handler:
                    self.gemini_handler.load_known_entities(self.known_entities)
                self.query_decomposer = QueryDecomposer(self.gemini_handler)
                self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo Gemini Handler: {e}. Các tính năng text AI sẽ bị hạn chế. ---")

        # --- Khởi tạo và xác thực OpenAI Handler cho các tác vụ VISION ---
        if openai_api_key:
            try:
                self.openai_handler = OpenAIHandler(api_key=openai_api_key)
                if not self.openai_handler.check_api_health():
                    self.openai_handler = None
                else:
                    self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo OpenAI Handler: {e}. Các tính năng vision AI sẽ bị hạn chế. ---")
        
        # --- Khởi tạo các Solver phức tạp nếu các handler cần thiết đã sẵn sàng ---
        if self.gemini_handler:
            self.trake_solver = TRAKESolver(ai_handler=self.gemini_handler)

        print(f"--- ✅ Master Searcher đã sẵn sàng! (AI Enabled: {self.ai_enabled}) ---")
        
    def perform_semantic_grounding(self, entities_to_ground: List[str]) -> Dict[str, str]:
        """
        Dịch các nhãn entity tự do về các nhãn chuẩn có trong từ điển.
        Trả về một dictionary mapping: {entity_gốc: entity_đã_dịch}.
        """
        if not entities_to_ground or not self.known_entities_prompt_segment:
            return {}

        print(f"--- 🧠 Bắt đầu Semantic Grounding cho: {entities_to_ground} ---")
        
        prompt = (
            f"You are a helpful assistant. Your task is to map a list of input entities to the closest matching entities from a predefined dictionary. "
            f"For each input entity, find the single most appropriate term from the dictionary.\n\n"
            f"**Predefined Dictionary:**\n{self.known_entities_prompt_segment}\n\n"
            f"**Input Entities to Map:**\n{json.dumps(entities_to_ground)}\n\n"
            f"Provide your answer ONLY as a valid JSON object mapping each input entity to its corresponding dictionary term. "
            f"Example format: {{\"input_entity_1\": \"dictionary_term_1\", \"input_entity_2\": \"dictionary_term_2\"}}"
        )
        
        try:
            response = self.model.generate_content(prompt)
            # Giả định response.text là một chuỗi JSON hợp lệ
            grounding_map = json.loads(response.text)
            print(f"    -> Kết quả Grounding: {grounding_map}")
            return grounding_map
        except Exception as e:
            print(f"--- ⚠️ Lỗi trong quá trình Semantic Grounding: {e} ---")
            # Fallback: Trả về mapping rỗng nếu có lỗi
            return {}
    
    def _deduplicate_temporally(self, results: List[Dict[str, Any]], time_threshold: int = 2) -> List[Dict[str, Any]]:
        """
        Lọc các kết quả bị trùng lặp về mặt thời gian trong cùng một video.

        Args:
            results (List[Dict[str, Any]]): Danh sách kết quả đã được sắp xếp theo điểm.
            time_threshold (int): Ngưỡng thời gian (giây). Các frame trong cùng video
                                  cách nhau dưới ngưỡng này sẽ bị coi là trùng lặp.

        Returns:
            List[Dict[str, Any]]: Danh sách kết quả đã được lọc.
        """
        if not results:
            return []

        print(f"--- 🛡️ Bắt đầu Lọc Trùng lặp Thời gian (Ngưỡng: {time_threshold}s)... ---")
        
        last_timestamp_per_video = {}
        
        deduplicated_results = []

        for result in results:
            video_id = result.get('video_id')
            timestamp = result.get('timestamp')

            if not video_id or timestamp is None:
                continue # Bỏ qua nếu thiếu thông tin

            last_seen_timestamp = last_timestamp_per_video.get(video_id)

            if last_seen_timestamp is None or abs(timestamp - last_seen_timestamp) > time_threshold:
                deduplicated_results.append(result)
                last_timestamp_per_video[video_id] = timestamp
        
        print(f"--- ✅ Lọc hoàn tất. Từ {len(results)} -> còn {len(deduplicated_results)} kết quả. ---")
        return deduplicated_results

# /search_core/master_searcher.py - PHIÊN BẢN CUỐI CÙNG (COPY VÀ THAY THẾ)

    def search(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi tìm kiếm theo chiến lược "PHOENIX REBORN" đã được tinh gọn.
        Luồng xử lý duy nhất, không phân nhánh.
        """
        print("\n" + "="*20 + " 🚀 Kích hoạt Chiến dịch PHOENIX REBORN " + "="*20)

        # === BƯỚC 1: PHÂN RÃ TRUY VẤN ===
        if not self.query_decomposer or not self.ai_enabled:
             print("--- ⚠️ Decomposer chưa sẵn sàng. Chạy ở chế độ KIS đơn giản. ---")
             sub_queries = [query]
        else:
             sub_queries = self.query_decomposer.decompose(query)
        
        print(f"   -> Truy vấn được phân rã thành {len(sub_queries)} truy vấn con: {sub_queries}")

        if not sub_queries:
            return {"task_type": TaskType.KIS, "results": [], "query_analysis": {}}

        # === BƯỚC 2: NÉM LƯỚI SONG SONG ===
        kis_retrieval_count = int(config.get('initial_retrieval_slider', 500))
        weights = {
            'w_clip': config.get('w_clip_slider', 0.4),
            'w_obj': config.get('w_obj_slider', 0.3),
            'w_semantic': config.get('w_semantic_slider', 0.3),
            'w_spatial': config.get('w_spatial_slider', 0.25),
            'w_fine_grained': config.get('w_fine_grained_slider', 0.25)
        }
        all_results_raw = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(
                    self.semantic_searcher.search, query_text=sq,
                    top_k_final=kis_retrieval_count, top_k_retrieval=kis_retrieval_count,
                    precomputed_analysis={}, weights=weights
                ): sq for sq in sub_queries
            }
            for future in tqdm(as_completed(future_to_query), total=len(sub_queries), desc="   -> Đang Ném Lưới"):
                sub_query = future_to_query[future]
                try:
                    results = future.result()
                    for res in results: res['matched_query'] = sub_query
                    all_results_raw.extend(results)
                except Exception as exc:
                    print(f"   -> ❌ Lỗi khi tìm kiếm cho sub-query '{sub_query}': {exc}")

        print(f"   -> Thu về tổng cộng {len(all_results_raw)} ứng viên thô.")
        if not all_results_raw:
             return {"task_type": TaskType.KIS, "results": [], "query_analysis": {"sub_queries": sub_queries}}

        # === BƯỚC 3: HỢP NHẤT VÀ TÍNH ĐIỂM ĐỒNG XUẤT HIỆN ===
        print("   -> Đang hợp nhất kết quả và tính điểm theo bằng chứng...")
        merged_candidates = defaultdict(lambda: {'sum_score': 0.0, 'matched_queries': set(), 'data': None})
        for res in all_results_raw:
            key = res['keyframe_id']
            merged_candidates[key]['sum_score'] += res.get('final_score', 0)
            merged_candidates[key]['matched_queries'].add(res['matched_query'])
            if merged_candidates[key]['data'] is None: merged_candidates[key]['data'] = res
        
        final_results = []
        for key, value in merged_candidates.items():
            final_data = value['data']
            num_matches = len(value['matched_queries'])
            sum_score = value['sum_score']
            final_score = (num_matches ** 1.5) * sum_score # Công thức điểm thưởng cho sự đồng xuất hiện
            final_data['final_score'] = final_score
            final_data['matched_queries'] = list(value['matched_queries'])
            final_results.append(final_data)
        
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        print(f"   -> Hợp nhất còn {len(final_results)} kết quả độc nhất.")

        # === BƯỚC 4: HOÀN THIỆN VÀ TRẢ VỀ ===
        top_n_before_dedup = final_results[:kis_retrieval_count]
        deduplicated_results = self._deduplicate_temporally(top_n_before_dedup)
        
        lambda_mmr = config.get('lambda_mmr_slider', 0.7)
        diverse_results = deduplicated_results
        if self.mmr_builder and diverse_results:
             print(f"   -> Áp dụng MMR (λ={lambda_mmr}) để đa dạng hóa kết quả...")
             diverse_results = self.mmr_builder.build_diverse_list(
                 candidates=diverse_results, target_size=len(diverse_results), lambda_val=lambda_mmr
             )
        
        top_k_final = int(config.get('num_results', 100))
        final_results_for_submission = diverse_results[:top_k_final]
        
        if self.video_path_map:
            for result in final_results_for_submission:
                result['video_path'] = self.video_path_map.get(result.get('video_id'))

        print(f"--- ✅ Chiến dịch PHOENIX REBORN hoàn tất. Trả về {len(final_results_for_submission)} kết quả. ---")
        
        return {
            "task_type": TaskType.KIS, # Luôn là KIS
            "results": final_results_for_submission,
            "query_analysis": {"sub_queries": sub_queries}
        }