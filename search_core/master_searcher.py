# search_core/master_searcher.py
from typing import Dict, Any, Optional, List
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from search_core.basic_searcher import BasicSearcher
from search_core.semantic_searcher import SemanticSearcher
from search_core.trake_solver import TRAKESolver
from search_core.gemini_text_handler import GeminiTextHandler
from search_core.openai_handler import OpenAIHandler
from search_core.task_analyzer import TaskType
from search_core.mmr_builder import MMRResultBuilder 


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
        if gemini_api_key:
            try:
                self.gemini_handler = GeminiTextHandler(api_key=gemini_api_key)
                if self.known_entities and self.gemini_handler:
                    self.gemini_handler.load_known_entities(self.known_entities)
                self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo Gemini Handler: {e}. Các tính năng text AI sẽ bị hạn chế. ---")
        if openai_api_key:
            try:
                self.openai_handler = OpenAIHandler(api_key=openai_api_key)
                if not self.openai_handler.check_api_health():
                    self.openai_handler = None
                else:
                    self.ai_enabled = True
            except Exception as e:
                print(f"--- ⚠️ Lỗi khi khởi tạo OpenAI Handler: {e}. Các tính năng vision AI sẽ bị hạn chế. ---")
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
            grounding_map = json.loads(response.text)
            print(f"    -> Kết quả Grounding: {grounding_map}")
            return grounding_map
        except Exception as e:
            print(f"--- ⚠️ Lỗi trong quá trình Semantic Grounding: {e} ---")
            return {}
    
    def _deduplicate_temporally(self, results: List[Dict[str, Any]], time_threshold: int = 5) -> List[Dict[str, Any]]:
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
                continue 
            last_seen_timestamp = last_timestamp_per_video.get(video_id)
            if last_seen_timestamp is None or abs(timestamp - last_seen_timestamp) > time_threshold:
                deduplicated_results.append(result)
                last_timestamp_per_video[video_id] = timestamp
        
        print(f"--- ✅ Lọc hoàn tất. Từ {len(results)} -> còn {len(deduplicated_results)} kết quả. ---")
        return deduplicated_results

    def search(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hàm tìm kiếm chính, nhận một dictionary config để tùy chỉnh hành vi.
        """
        top_k_final = int(config.get('top_k_final', 100))
        kis_retrieval = int(config.get('kis_retrieval', 200))
        vqa_candidates_to_rank = int(config.get('vqa_candidates', 20))
        vqa_retrieval = int(config.get('vqa_retrieval', 200))
        trake_candidates_per_step = int(config.get('trake_candidates_per_step', 20))
        trake_max_sequences = int(config.get('trake_max_sequences', 50))
        w_clip = config.get('w_clip', 0.4)
        w_obj = config.get('w_obj', 0.3)
        w_semantic = config.get('w_semantic', 0.3)
        lambda_mmr = config.get('lambda_mmr', 0.7)

        query_analysis = {}
        task_type = TaskType.KIS
        if self.ai_enabled and self.gemini_handler:
            print("--- ✨ Bắt đầu phân tích truy vấn bằng Gemini Text Handler... ---")
            query_analysis = self.gemini_handler.analyze_query_fully(query)
            
            entities_to_ground = query_analysis.get('entities_to_ground', [])
            if entities_to_ground:
                grounding_map = self.gemini_handler.perform_semantic_grounding(entities_to_ground)
                query_analysis['grounding_map'] = grounding_map
            else:
                query_analysis['grounding_map'] = {}
            
            original_objects = query_analysis.get('objects_en', [])
            if original_objects:
                grounded_objects = self.gemini_handler.perform_semantic_grounding(original_objects)
                if original_objects != grounded_objects:
                     print(f"--- 🧠 Semantic Grounding: {original_objects} -> {grounded_objects} ---")
                query_analysis['objects_en'] = grounded_objects
                
            task_type_str = query_analysis.get('task_type', 'KIS').upper()
            try:
                task_type = TaskType[task_type_str]
            except KeyError:
                task_type = TaskType.KIS
        
        print(f"--- Đã phân loại truy vấn là: {task_type.value} ---")

        final_results = []
        query_analysis.update({'w_clip': w_clip, 'w_obj': w_obj, 'w_semantic': w_semantic})
        search_context = query_analysis.get('search_context', query)

        if task_type == TaskType.TRAKE:
            if self.trake_solver:
                sub_queries = self.trake_solver.decompose_query(query)
                final_results = self.trake_solver.find_sequences(
                    sub_queries, 
                    self.semantic_searcher,
                    original_query_analysis=query_analysis,
                    top_k_per_step=trake_candidates_per_step,
                    max_sequences=trake_max_sequences
                )
            else:
                task_type = TaskType.KIS

        elif task_type == TaskType.QNA:
            if self.openai_handler:
                candidates = self.semantic_searcher.search(
                    query_text=search_context,
                    precomputed_analysis=query_analysis,
                    top_k_final=vqa_retrieval,
                    top_k_retrieval=vqa_retrieval
                )
                
                if not candidates:
                    final_results = []
                else:
                    candidates_for_vqa = candidates[:vqa_candidates_to_rank]
                    specific_question = query_analysis.get('specific_question', query)
                    vqa_enhanced_candidates = []
                    
                    print(f"--- 💬 Bắt đầu Quét VQA song song trên {len(candidates_for_vqa)} ứng viên... ---")
                    
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        future_to_candidate = {
                            executor.submit(
                                self.openai_handler.perform_vqa, 
                                image_path=cand['keyframe_path'], 
                                question=specific_question, 
                                context_text=cand.get('transcript_text', '')
                            ): cand 
                            for cand in candidates_for_vqa
                        }
                        
                        for future in tqdm(as_completed(future_to_candidate), total=len(candidates_for_vqa), desc="   -> VQA Progress"):
                            cand = future_to_candidate[future]
                            try:
                                vqa_result = future.result()
                                new_cand = cand.copy()
                                new_cand['answer'] = vqa_result['answer']
                                search_score = new_cand.get('final_score', 0)
                                vqa_confidence = vqa_result.get('confidence', 0)
                                new_cand['final_score'] = search_score * vqa_confidence
                                new_cand['scores']['vqa_confidence'] = vqa_confidence
                                vqa_enhanced_candidates.append(new_cand)
                            except Exception as exc:
                                print(f"--- ❌ Lỗi khi xử lý VQA cho keyframe {cand.get('keyframe_id')}: {exc} ---")
                    
                    if vqa_enhanced_candidates:
                        final_results = sorted(vqa_enhanced_candidates, key=lambda x: x['final_score'], reverse=True)
                    else:
                        final_results = []
            else:
                print("--- ⚠️ OpenAI (VQA) handler chưa được kích hoạt. Fallback về KIS. ---")
                task_type = TaskType.KIS

        if not final_results or task_type == TaskType.KIS:
            final_results = self.semantic_searcher.search(
                query_text=search_context,
                precomputed_analysis=query_analysis,
                top_k_final=kis_retrieval, 
                top_k_retrieval=kis_retrieval
            )
        if task_type in [TaskType.KIS, TaskType.QNA]:
            final_results = self._deduplicate_temporally(final_results, time_threshold=2)
        if self.video_path_map and task_type in [TaskType.KIS, TaskType.QNA]:
            for result in final_results:
                result['video_path'] = self.video_path_map.get(result.get('video_id'))
        diverse_results = final_results
        # if self.mmr_builder and final_results:
        #     if task_type in [TaskType.KIS, TaskType.QNA]:
        #         diverse_results = self.mmr_builder.build_diverse_list(
        #             candidates=final_results, 
        #             target_size=len(final_results),
        #             lambda_val=lambda_mmr
        #         )
        final_results_for_submission = diverse_results[:top_k_final]
        print("\n" + "="*20 + " DEBUG LOG: MASTER SEARCHER OUTPUT " + "="*20)
        print(f"-> Task Type cuối cùng: {task_type.value}")
        print(f"-> Số lượng kết quả cuối cùng: {len(final_results)}")
        if final_results:
            print("-> Ví dụ kết quả đầu tiên:")
            first_result = final_results[0]
            if task_type == TaskType.TRAKE:
                print(f"  - video_id: {first_result.get('video_id')}")
                print(f"  - final_score: {first_result.get('final_score')}")
                print(f"  - Số bước trong chuỗi: {len(first_result.get('sequence', []))}")
            else:
                print(f"  - keyframe_id: {first_result.get('keyframe_id')}")
                print(f"  - final_score: {first_result.get('final_score')}")
                if 'answer' in first_result:
                    print(f"  - answer: {first_result.get('answer')}")
        else:
            print("-> Không có kết quả nào được tạo ra.")
        print("="*68 + "\n")
        
        return {
            "task_type": task_type,
            "results": final_results_for_submission,
            "query_analysis": query_analysis
        }