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

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
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
        Hàm tạo ra SYSTEM_PROMPT động, nhúng danh sách thực thể đã biết vào.
        Đây là phần nâng cấp cốt lõi để Gemini "nhận thức" được từ điển của hệ thống.
        """
        return f"""
You are a highly precise multimedia scene analyst for a Vietnamese video search engine. Your task is to deconstruct a user's query into a structured, machine-readable JSON object. Your SOLE output must be a single, valid JSON object and nothing else. Adhere strictly to the specified formats and keywords.

**IMPORTANT RULE:** When creating `entity` and `targets` for `spatial_rules`, you MUST prioritize using a label from this list of KNOWN ENTITIES if it is relevant: {self.known_entities_prompt_segment}. This is crucial for the system to understand. For example, if the user mentions a "giant crab symbol" and "building" is in the KNOWN ENTITIES list, you should prefer using "building". If no known entity is a good fit, you may create a new descriptive label.

Analyze the user query to extract three components:
1.  **`search_context` (string):**
    *   This is a conceptual summary. IGNORE specific, verifiable details like exact counts ("three people"), colors ("red shirt"), or text on signs.
    *   FOCUS on the **core activity, environment, and overall theme**.
    *   Good Example: For "a girl in a red dress with a yellow balloon", the context is "a child enjoying an outdoor festival or celebration".

2.  **`spatial_rules` (list of objects):**
    *   Identify ALL explicit spatial relationships.
    *   For each, create an object with `entity`, `relation`, and `targets`.
    *   **`relation` MUST be one of these exact keywords:** `is_between`, `is_behind`, `is_next_to`, `is_above`, `is_below`, `is_on`, `is_inside`.
    *   **`entity` and `targets`**: Use descriptive, snake_cased English labels (e.g., "person_white_shirt", "black_car"). Remember to use labels from the KNOWN ENTITIES list when possible.

3.  **`fine_grained_verification` (list of objects):**
    *   Identify objects with highly specific visual descriptions.
    *   For each, create an object with `target_entity` (the general, common, single-word English class name, e.g., "Bird", "Car") and `detailed_description` (the full descriptive English sentence).

---
**COMPREHENSIVE EXAMPLE:**
User Query: "Find a clip of three people (a woman in a white shirt sitting between two men in black shirts) playing instruments, with a bookshelf behind them."

Your JSON output:
{{
  "search_context": "a group of people playing musical instruments in a cozy, indoor setting like a library or studio",
  "spatial_rules": [
    {{
      "entity": "woman",
      "relation": "is_between",
      "targets": ["man", "man"]
    }},
    {{
      "entity": "bookshelf",
      "relation": "is_behind",
      "targets": ["woman"]
    }}
  ],
  "fine_grained_verification": []
}}
---
Now, analyze the user's query and provide ONLY the JSON output.
"""

    @api_retrier(max_retries=3, initial_delay=1)
    def _gemini_api_call(self, content_list: list) -> genai.GenerativeModel.generate_content:
        """Hàm con được "trang trí", chuyên thực hiện lệnh gọi API của Gemini."""
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