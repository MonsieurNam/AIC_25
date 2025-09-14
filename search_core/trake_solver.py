
from typing import List, Dict, Any

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from search_core.openai_handler import OpenAIHandler
    from search_core.semantic_searcher import SemanticSearcher

class TRAKESolver:
    """
    Class xử lý Nhiệm vụ 3: TRAKE (TRacking Action KEyframes).

    Nó bao gồm hai chức năng chính:
    1.  Phân rã một truy vấn hành động phức tạp thành các bước con (sử dụng AI handler).
    2.  Tìm kiếm các chuỗi keyframe hợp lệ bằng cách:
        a. Phân tích và tìm kiếm ứng viên cho từng bước con.
        b. Lắp ráp các ứng viên thành các chuỗi hợp lệ (cùng video, đúng thứ tự).
    """

    def __init__(self, ai_handler: 'OpenAIHandler'):
        """
        Khởi tạo TRAKESolver.

        Args:
            ai_handler (OpenAIHandler): Một instance của AI Handler (ví dụ: OpenAIHandler)
                                        để thực hiện việc phân rã và phân tích truy vấn.
        """
        self.ai_handler = ai_handler

    def decompose_query(self, query: str) -> List[str]:
        """
        Sử dụng AI Handler được cung cấp để tách truy vấn TRAKE thành các bước hành động con.
        """
        print(f"--- 🤖 Phân rã truy vấn TRAKE bằng AI Handler... ---")
        return self.ai_handler.decompose_trake_query(query)

    def find_sequences(self, 
                       sub_queries: List[str], 
                       searcher: 'SemanticSearcher',
                       original_query_analysis: Dict[str, Any],
                       top_k_per_step: int, 
                       max_sequences: int,
                       beam_width: int = 5 
                      ) -> List[Dict[str, Any]]:
        """
        Tìm các chuỗi keyframe hợp lệ bằng thuật toán Beam Search.
        """
        if not sub_queries:
            return []

        print(f"--- Bắt đầu tìm kiếm ứng viên cho {len(sub_queries)} bước TRAKE ---")
        
        step_candidates = []
        for i, sub_query in enumerate(sub_queries):
            print(f"   -> Bước {i+1}: Đang tìm kiếm cho '{sub_query}'")
            
            sub_query_analysis = self.ai_handler.analyze_query_fully(sub_query)
            sub_query_analysis['w_clip'] = original_query_analysis.get('w_clip')
            sub_query_analysis['w_obj'] = original_query_analysis.get('w_obj')
            sub_query_analysis['w_semantic'] = original_query_analysis.get('w_semantic')
            search_context = sub_query_analysis.get('search_context', sub_query)
            
            results = searcher.search(
                query_text=search_context,
                precomputed_analysis=sub_query_analysis,
                top_k_final=top_k_per_step,
                top_k_retrieval=200
            )
            step_candidates.append(results)
        
        print("\n--- Đang nhóm các ứng viên theo video ---")
        candidates_by_video: Dict[str, List[List[Dict]]] = {}
        for i, candidates in enumerate(step_candidates):
            for cand in candidates:
                video_id = cand['video_id']
                if video_id not in candidates_by_video:
                    candidates_by_video[video_id] = [[] for _ in sub_queries]
                candidates_by_video[video_id][i].append(cand)
        
        print(f"\n--- Bắt đầu lắp ráp chuỗi bằng Beam Search (beam_width={beam_width}) ---")
        all_valid_sequences = []
        for video_id, video_step_candidates in candidates_by_video.items():
            if not all(video_step_candidates):
                continue
            
            beam = [([cand], cand['final_score']) for cand in video_step_candidates[0]]
            
            for step_idx in range(1, len(sub_queries)):
                next_beam = []
                for current_sequence, current_score in beam:
                    last_frame_timestamp = current_sequence[-1]['timestamp']
                    
                    for next_candidate in video_step_candidates[step_idx]:
                        if next_candidate['timestamp'] > last_frame_timestamp:
                            new_sequence = current_sequence + [next_candidate]
                            new_score = current_score + next_candidate['final_score']
                            next_beam.append((new_sequence, new_score))
                
                next_beam.sort(key=lambda x: x[1], reverse=True)
                beam = next_beam[:beam_width]

            for final_sequence, total_score in beam:
                avg_score = total_score / len(final_sequence)
                all_valid_sequences.append({
                    "video_id": video_id,
                    "sequence": final_sequence,
                    "final_score": avg_score
                })

        print(f"--- Tìm thấy tổng cộng {len(all_valid_sequences)} chuỗi hợp lệ. Đang sắp xếp... ---")
        sorted_sequences = sorted(all_valid_sequences, key=lambda x: x['final_score'], reverse=True)
        
        return sorted_sequences[:max_sequences]