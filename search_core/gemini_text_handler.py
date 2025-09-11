# ==============================================================================
# GEMINI TEXT HANDLER - PHIÊN BẢN CUỐI CÙNG (ENTITY-AWARE)
# File: gemini_text_handler.py
#
# THAY ĐỔI CỐT LÕI:
#   - Sửa đổi SYSTEM_PROMPT để tích hợp "Từ điển Đối tượng Toàn cục",
#     hướng dẫn Gemini ưu tiên sử dụng các nhãn thực thể đã biết.
#   - Tối ưu hóa việc tạo prompt để tăng độ chính xác của Semantic Grounding
#     và phân tích không gian.
# ==============================================================================

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, List, Set
import json
import re

from utils import api_retrier

class GeminiTextHandler:
    """
    Một class chuyên dụng để xử lý TẤT CẢ các tác vụ liên quan đến văn bản
    bằng API của Google Gemini. PHIÊN BẢN NÂNG CẤP (ENTITY-AWARE).
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Khởi tạo và xác thực Gemini Text Handler.
        """
        print(f"--- ✨ Khởi tạo Gemini Text Handler với model: {model_name} ---")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.known_entities_prompt_segment: str = "[]" # Mặc định là list rỗng dạng chuỗi
            
            # --- Cấu hình API call ---
            self.generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
            self.safety_settings = {
                'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE',
                'SEXUAL': 'BLOCK_NONE', 'DANGEROUS': 'BLOCK_NONE'
            }
            
            # --- Xác thực API Key bằng một lệnh gọi nhỏ ---
            print("--- 🩺 Đang thực hiện kiểm tra trạng thái API Gemini... ---")
            self.model.count_tokens("test") 
            print("--- ✅ Trạng thái API Gemini: OK ---")

        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi khởi tạo Gemini Handler: {e} ---")
            print("    -> Vui lòng kiểm tra lại API Key và kết nối mạng.")
            raise e

    def _get_system_prompt(self) -> str:
        """
        Hàm tạo ra SYSTEM_PROMPT động, nhúng danh sách 600 thực thể của OpenImagesV4.
        Đây là phiên bản "chuyên gia", được tối ưu hóa cho dữ liệu của cuộc thi.
        """
        return f"""
    You are a highly precise multimedia scene analyst. Your task is to deconstruct a user's query into a structured JSON object. Your analysis is for a video search engine whose object detection system is based on the **OpenImagesV4 dataset**.

    **CRITICAL INSTRUCTION: The detection system ONLY knows the following 600 object classes. When creating `entity` and `targets` for `spatial_rules`, you MUST use the EXACT labels from this list. This is the most important rule.**

    **List of 600 Known Object Classes (OpenImagesV4 Vocabulary):**
    {self.known_entities_prompt_segment}

    **Analysis Rules:**
    1.  **If the user describes an object that is in the list**, use the label from the list.
        *   Example: User says "biểu tượng con cua khổng lồ". The closest known entity is "Sculpture" or "Building". You MUST choose one of them. DO NOT create a new label like "giant_crab_symbol".
    2.  **If the user describes an object that is NOT in the list**, find the most logical **parent class or component** from the list.
        *   Example: User says "bản đồ Việt Nam làm bằng trái cây". The known list has "Fruit", "Apple", "Orange", but not "map". You should use "Fruit" as the best available approximation.
    3.  **Your JSON output MUST contain three components:**
        *   `search_context`: A general summary of the scene.
        *   `spatial_rules`: A list of spatial relationships using ONLY the labels from the provided list.
        *   `fine_grained_verification`: A list for detailed visual attributes not covered by the object labels.

    ---
    **EXAMPLE:**
    User Query: "Nhiều người mặc áo cờ đỏ sao vàng đứng trước biểu tượng một con cua khổng lồ."

    **Your CORRECT JSON output (because 'Sculpture' and 'Person' are in the list):**
    {{
    "search_context": "a crowd of people celebrating in front of a large sculpture",
    "spatial_rules": [
        {{
        "entity": "Sculpture",
        "relation": "is_behind",
        "targets": ["Person"]
        }}
    ],
    "fine_grained_verification": [
        {{
        "target_entity": "Flag",
        "detailed_description": "a Vietnamese flag (red flag with a yellow star)"
        }},
        {{
        "target_entity": "Sculpture",
        "detailed_description": "a giant sculpture shaped like a crab"
        }}
    ]
    }}
    ---
    Now, analyze the user's query and provide ONLY the JSON output, strictly following these rules.
    """

    @api_retrier(max_retries=3, initial_delay=1)
    def _gemini_api_call(self, content_list: list) -> genai.GenerativeModel.generate_content:
        """Hàm con được 'trang trí', chuyên thực hiện lệnh gọi API của Gemini."""
        return self.model.generate_content(
            content_list,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )

    def load_known_entities(self, known_entities: Set[str]):
        """
        Chuẩn bị và cache lại phần prompt chứa từ điển đối tượng.
        Chỉ cần gọi một lần khi MasterSearcher khởi tạo.
        """
        if not known_entities:
            print("--- ⚠️ Từ điển đối tượng rỗng. Semantic Grounding sẽ không hoạt động tối ưu. ---")
            return
        
        sorted_entities = sorted(list(known_entities))
        # Định dạng thành chuỗi JSON để nhúng vào prompt
        self.known_entities_prompt_segment = json.dumps(sorted_entities)
        print(f"--- ✅ GeminiTextHandler: Đã nạp {len(sorted_entities)} thực thể vào bộ nhớ prompt. ---")

    def analyze_query_fully(self, query: str) -> Dict[str, Any]:
        """
        Phân tích sâu một truy vấn, trích xuất ngữ cảnh, đối tượng, và các quy tắc.
        """
        print("--- ✨ Bắt đầu phân tích truy vấn có cấu trúc bằng Gemini (Entity-Aware)... ---")
        
        system_prompt = self._get_system_prompt()
        user_prompt = f"User Query: \"{query}\""
        
        try:
            response = self._gemini_api_call([system_prompt, user_prompt])
            raw_response_text = response.text.strip()
            
            try:
                if raw_response_text.startswith("```json"):
                    raw_response_text = raw_response_text[7:-3].strip()
                analysis_json = json.loads(raw_response_text)
                
                # Trích xuất các thực thể cần được "grounding" sau này
                entities_to_ground = set()
                if 'spatial_rules' in analysis_json and isinstance(analysis_json['spatial_rules'], list):
                    for rule in analysis_json['spatial_rules']:
                        if 'entity' in rule and isinstance(rule['entity'], str):
                            entities_to_ground.add(rule['entity'].replace('_', ' '))
                        if 'targets' in rule and isinstance(rule['targets'], list):
                            for target in rule['targets']:
                                if isinstance(target, str):
                                    entities_to_ground.add(target.replace('_', ' '))
                analysis_json['entities_to_ground'] = list(entities_to_ground)
                return analysis_json

            except json.JSONDecodeError:
                print(f"--- ⚠️ Lỗi: Gemini không trả về JSON hợp lệ. Sử dụng fallback. Raw response: {raw_response_text}")
                return {"search_context": query, "spatial_rules": [], "fine_grained_verification": [], "entities_to_ground": []}

        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi gọi API Gemini: {e} ---")
            return {"search_context": query, "spatial_rules": [], "fine_grained_verification": [], "entities_to_ground": []}

    def perform_semantic_grounding(self, entities_to_ground: List[str]) -> Dict[str, str]:
        """
        Dịch các nhãn entity tự do về các nhãn chuẩn có trong từ điển.
        """
        if not entities_to_ground or self.known_entities_prompt_segment == "[]":
            return {}

        print(f"--- 🧠 Bắt đầu Semantic Grounding cho: {entities_to_ground} ---")
        
        prompt = (
            f"You are a helpful assistant. Your task is to map a list of input entities to the closest matching entities from a predefined dictionary.\n\n"
            f"**Predefined Dictionary:**\n{self.known_entities_prompt_segment}\n\n"
            f"**Input Entities to Map:**\n{json.dumps(entities_to_ground)}\n\n"
            f"Provide your answer ONLY as a valid JSON object mapping each input entity to its corresponding dictionary term. The keys of the JSON must be the original input entities."
        )
        
        try:
            response = self._gemini_api_call([prompt])
            raw_response_text = response.text.strip()
            if raw_response_text.startswith("```json"):
                raw_response_text = raw_response_text[7:-3].strip()
            
            grounding_map = json.loads(raw_response_text)
            print(f"    -> Kết quả Grounding Map: {grounding_map}")
            
            if not isinstance(grounding_map, dict):
                print(f"--- ⚠️ Lỗi Grounding: Gemini không trả về dictionary. Fallback. ---")
                return {}
            return grounding_map

        except Exception as e:
            print(f"--- ⚠️ Lỗi trong quá trình Semantic Grounding: {e} ---")
            return {}
            
    # --- CÁC HÀM CŨ KHÔNG THAY ĐỔI ---
    def decompose_trake_query(self, query: str) -> List[str]:
        """Phân rã truy vấn TRAKE bằng Gemini."""
        prompt = f"""
        Decompose the Vietnamese query describing a sequence of actions into a JSON array of short, self-contained phrases. Return ONLY the JSON array.

        Example:
        Query: "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: (1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
        JSON: ["vận động viên giậm nhảy", "vận động viên bay qua xà", "vận động viên tiếp đất", "vận động viên đứng dậy"]

        Query: "{query}"
        JSON:
        """
        try:
            response = self._gemini_api_call([prompt])
            match = re.search(r"\[.*?\]", response.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return [query]
        except Exception:
            return [query]
    def get_text_response(self, prompt: str, system_prompt: str) -> str:
        """
        Gửi một prompt tới Gemini và trả về kết quả dạng text.
        Đây là giao diện chung, sạch sẽ cho các module khác sử dụng.
        """
        try:
            # Gửi cả system prompt và user prompt
            response = self._gemini_api_call([system_prompt, prompt])
            return response.text.strip()
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi gọi API Gemini qua get_text_response: {e} ---")
            # Trả về chuỗi rỗng để bên gọi có thể xử lý fallback một cách an toàn
            return ""