import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, List, Set
import json
import re

from utils import api_retrier

SYSTEM_PROMPT = """
You are an expert multimedia scene analyst for a Vietnamese video search engine. Your task is to dissect a user's query into structured, machine-readable components. Do NOT answer the user's query. Your SOLE output must be a single, valid JSON object and nothing else, without any markdown formatting like ```json.

The user query will describe a visual scene. You must analyze it and extract three key components:

1.  **`search_context` (string):**
    *   An abstract, conceptual summary of the scene. IGNORE fine-grained details.
    *   FOCUS on the **essence, atmosphere, and overall action**.
    *   Example: For "a girl in a red dress holding a yellow balloon", the context is "a child enjoying an outdoor festival".

2.  **`spatial_rules` (list of objects):**
    *   Identify ALL spatial relationships (e.g., "between", "behind").
    *   Create a JSON object for each with `entity`, `relation` (`is_between`, `is_behind`, etc.), and `targets`.
    *   Use descriptive English labels (e.g., "person_white_shirt").
    *   If none, this MUST be an empty list `[]`.

3.  **`fine_grained_verification` (list of objects):**
    *   Identify specific objects whose appearance is described in great detail (colors, textures, specific parts). These are details the main search might miss.
    *   For each, create a JSON object with two fields:
        *   `target_entity` (string): The general class name of the object (e.g., "Bird", "Flower", "Dessert"). This MUST be a common, single-word noun.
        *   `detailed_description` (string): A full, descriptive English sentence detailing the object's specific visual characteristics as mentioned in the query.
    *   Example: For "a bird with bright red eyes", the object would be:
        `{"target_entity": "Bird", "detailed_description": "a bird with bright red eyes and blue-black feathers"}`
    *   If no such detailed descriptions exist, this MUST be an empty list `[]`.

**OUTPUT FORMAT EXAMPLE:**
User Query: "On a white plate, there is a panna cotta dessert decorated with red grape slices, a green mint leaf, and two small edible flowers (one red, one yellow)."

Your JSON output:
{
  "search_context": "a close-up shot of a gourmet dessert, panna cotta, being plated or displayed",
  "spatial_rules": [],
  "fine_grained_verification": [
    {
      "target_entity": "Grape",
      "detailed_description": "slices of red grapes used as a garnish"
    },
    {
      "target_entity": "Mint",
      "detailed_description": "a fresh green mint leaf on a dessert"
    },
    {
      "target_entity": "Flower",
      "detailed_description": "a small, edible red and yellow flower for decoration"
    }
  ]
}

Now, analyze the user's query and provide ONLY the JSON output.
"""

class GeminiTextHandler:
    """
    Một class chuyên dụng để xử lý TẤT CẢ các tác vụ liên quan đến văn bản
    bằng API của Google Gemini (cụ thể là model Flash).
    
    Bao gồm: phân loại tác vụ, phân tích chi tiết truy vấn, và phân rã truy vấn TRAKE.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Khởi tạo và xác thực Gemini Text Handler.
        PHIÊN BẢN ĐÃ SỬA LỖI: Lưu trữ generation_config và safety_settings.
        """
        print(f"--- ✨ Khởi tạo Gemini Text Handler với model: {model_name} ---")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.known_entities_prompt_segment: str = "" # Sẽ được nạp sau
            
            # --- ✅ LƯU TRỮ CÁC CẤU HÌNH THÀNH THUỘC TÍNH CỦA CLASS ---
            self.generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
            
            # Cấu hình an toàn để tránh bị block do các nội dung nhạy cảm
            self.safety_settings = {
                'HATE': 'BLOCK_NONE',
                'HARASSMENT': 'BLOCK_NONE',
                'SEXUAL': 'BLOCK_NONE',
                'DANGEROUS': 'BLOCK_NONE'
            }
            
            # --- Xác thực API Key bằng một lệnh gọi nhỏ ---
            print("--- 🩺 Đang thực hiện kiểm tra trạng thái API Gemini... ---")
            # Lệnh gọi đơn giản để kiểm tra xem API key có hoạt động không
            self.model.count_tokens("test") 
            print("--- ✅ Trạng thái API Gemini: OK ---")
            print("--- ✅ Gemini Text Handler đã được khởi tạo và xác thực thành công! ---")

        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi khởi tạo Gemini Handler: {e} ---")
            print("    -> Vui lòng kiểm tra lại API Key và kết nối mạng.")
            # Ném lại lỗi để quá trình khởi tạo backend có thể dừng lại nếu cần
            raise e

    @api_retrier(max_retries=3, initial_delay=1)
    def _gemini_text_call(self, prompt: str):
        """Hàm con được "trang trí", chỉ để thực hiện lệnh gọi API text của Gemini."""
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        response = self.model.generate_content(prompt, safety_settings=safety_settings, generation_config=generation_config)
        return response

    def health_check(self):
        """Thực hiện một lệnh gọi API đơn giản để kiểm tra key và kết nối."""
        print("--- 🩺 Đang thực hiện kiểm tra trạng thái API Gemini... ---")
        try:
            self.model.count_tokens("kiểm tra")
            print("--- ✅ Trạng thái API Gemini: OK ---")
            return True
        except Exception as e:
            print(f"--- ❌ Lỗi API Gemini: {e} ---")
            raise e

    def analyze_task_type(self, query: str) -> str:
        """Phân loại truy vấn bằng Gemini, sử dụng prompt có Quy tắc Ưu tiên."""
        prompt = f"""
        You are a highly precise query classifier. Your task is to classify a Vietnamese query into one of four categories: TRAKE, QNA, or KIS. You MUST follow a strict priority order.

        **Priority Order for Classification (Check from top to bottom):**
        
        1.  **First, check for TRAKE:** Does the query ask for a SEQUENCE of DIFFERENT, ordered actions? Look for patterns like "(1)...(2)...", "bước 1... bước 2", "sau đó". If it matches, classify as **TRAKE** and stop.
            - Example: "người đàn ông đứng lên rồi bước đi"

        2.  **Then, check for QNA:** If not TRAKE, does the query ask a **direct question** that expects a factual answer about something in the scene? This is more than just describing a scene. Look for:
            - **Interrogative words:** "ai", "cái gì", "ở đâu", "khi nào", "tại sao", "như thế nào", "màu gì", "hãng nào", etc.
            - **Question structures:** "có phải là...", "đang làm gì", "là ai", "trông như thế nào".
            - A question mark "?".
            If it matches, classify as **QNA** and stop.
            - Example: "người phụ nữ mặc áo màu gì?" -> QNA
            - Example: "ai là người đàn ông đang phát biểu?" -> QNA
            - Example: "có bao nhiêu chiếc xe trên đường?" -> This asks for a count of a single scene, so it is **QNA**. 

        3.  **Default to KIS:** If the query is a statement or a descriptive phrase looking for a moment, classify as **KIS**. It describes "what to find", not "what to answer".
            - Example: "cảnh người đàn ông đang phát biểu" -> KIS
            - Example: "tìm người phụ nữ mặc áo đỏ" -> KIS
            - Example: "một chiếc xe đang chạy" -> KIS

        **Your Task:**
        Follow the priority order strictly. Analyze the query below and return ONLY the final category as a single word.

        **Query:** "{query}"
        **Category:**
        """
        try:
            response = self._gemini_text_call(prompt)
            task_type = response.text.strip().upper()
            if task_type in ["KIS", "QNA", "TRAKE"]:
                return task_type
            return "KIS"
        except Exception:
            return "KIS" # Fallback an toàn
        
    def analyze_query_fully(self, query: str) -> Dict[str, Any]:
        """
        Phân tích sâu một truy vấn, trích xuất ngữ cảnh, đối tượng, và các quy tắc.
        PHIÊN BẢN NÂNG CẤP: Xử lý output JSON có cấu trúc.
        """
        print("--- ✨ Bắt đầu phân tích truy vấn có cấu trúc bằng Gemini... ---")
        
        user_prompt = f"Entities Dictionary for Grounding: {self.known_entities_prompt_segment}\n\nUser Query: \"{query}\""
        
        try:
            response = self.model.generate_content(
                [SYSTEM_PROMPT, user_prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            raw_response_text = response.text.strip()
            
            try:
                if raw_response_text.startswith("```json"):
                    raw_response_text = raw_response_text[7:]
                if raw_response_text.endswith("```"):
                    raw_response_text = raw_response_text[:-3]

                analysis_json = json.loads(raw_response_text)
                
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
                print(f"--- ⚠️ Lỗi: Gemini không trả về JSON hợp lệ. Sử dụng fallback. ---")
                print(f"    Raw response: {raw_response_text}")
                return {
                    "search_context": query, # Dùng query gốc làm context
                    "spatial_rules": [],
                    "fine_grained_verification": [],
                    "grounded_entities": []
                }

        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng khi gọi API Gemini: {e} ---")
            import traceback
            traceback.print_exc()
            # Fallback trong trường hợp API lỗi
            return {
                "search_context": query,
                "spatial_rules": [],
                "fine_grained_verification": [],
                "grounded_entities": []
            }

    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Phân tích và trích xuất thông tin truy vấn bằng Gemini."""
        fallback_result = {
            'search_context': query, 'specific_question': "", 'aggregation_instruction': "",
            'objects_vi': [], 'objects_en': []
        }
        prompt = f"""
        Analyze a Vietnamese user query for a video search system. **Return ONLY a valid JSON object** with five keys: "search_context", "specific_question", "aggregation_instruction", "objects_vi", and "objects_en".

        **Rules:**
        1.  `search_context`: A Vietnamese phrase for finding the general scene. This is used for vector search.
        2.  `specific_question`: The specific question to ask the Vision model for EACH individual frame.
        3.  `aggregation_instruction`: The final instruction for the AI to synthesize all individual answers. This should reflect the user's ultimate goal (e.g., counting, listing, summarizing).
        4.  `objects_vi`: A list of Vietnamese nouns/entities.
        5.  `objects_en`: The English translation for EACH item in `objects_vi`.

        **Example (VQA):**
        Query: "Trong video quay cảnh bữa tiệc, người phụ nữ mặc váy đỏ đang cầm ly màu gì?"
        JSON: {{"search_context": "cảnh bữa tiệc có người phụ nữ mặc váy đỏ", "specific_question": "cô ấy đang cầm ly màu gì?", "aggregation_instruction": "trả lời câu hỏi người phụ nữ cầm ly màu gì", "objects_vi": ["bữa tiệc", "người phụ nữ", "váy đỏ"], "objects_en": ["party", "woman", "red dress"]}}

        **Example (Track-VQA):**
        Query: "đếm xem có bao nhiêu con lân trong buổi biểu diễn"
        JSON: {{"search_context": "buổi biểu diễn múa lân", "specific_question": "có con lân nào trong ảnh này không và màu gì?", "aggregation_instruction": "từ các quan sát, đếm tổng số lân và liệt kê màu sắc của chúng", "objects_vi": ["con lân", "buổi biểu diễn"], "objects_en": ["lion dance", "performance"]}}

        **Your Task:**
        Analyze the query below and generate the JSON.

        **Query:** "{query}"
        **JSON:**
        """
        try:
            response = self._gemini_text_call(prompt)
            # Trích xuất JSON từ markdown block (Gemini thường trả về như vậy)
            match = re.search(r"```json\s*(\{.*?\})\s*```", response.text, re.DOTALL)
            if not match:
                match = re.search(r"(\{.*?\})", response.text, re.DOTALL) # Thử tìm JSON không có markdown
            
            if match:
                result = json.loads(match.group(1))
                # Validate ...
                return result
            return fallback_result
        except Exception as e:
            print(f"Lỗi Gemini enhance_query: {e}")
            return fallback_result
            
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
            response = self._gemini_text_call(prompt)
            match = re.search(r"\[.*?\]", response.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return [query]
        except Exception:
            return [query]
        
    def load_known_entities(self, known_entities: Set[str]):
        """
        Chuẩn bị và cache lại phần prompt chứa từ điển đối tượng.
        Chỉ cần gọi một lần khi MasterSearcher khởi tạo.
        """
        if not known_entities:
            print("--- ⚠️ Từ điển đối tượng rỗng. Semantic Grounding sẽ không hoạt động. ---")
            return
        
        # Sắp xếp để đảm bảo prompt nhất quán giữa các lần chạy
        sorted_entities = sorted(list(known_entities))
        # Định dạng thành chuỗi JSON để nhúng vào prompt
        self.known_entities_prompt_segment = json.dumps(sorted_entities)
        print(f"--- ✅ GeminiTextHandler: Đã nạp {len(sorted_entities)} thực thể vào bộ nhớ prompt. ---")

    def perform_semantic_grounding(self, entities_to_ground: List[str]) -> Dict[str, str]:
        """
        Dịch các nhãn entity tự do về các nhãn chuẩn có trong từ điển.
        PHIÊN BẢN NÂNG CẤP: Trả về một dictionary mapping: {entity_gốc: entity_đã_dịch}.
        """
        if not entities_to_ground or not self.known_entities_prompt_segment:
            return {}

        print(f"--- 🧠 Bắt đầu Semantic Grounding cho: {entities_to_ground} ---")
        
        # Prompt được thiết kế để yêu cầu Gemini trả về JSON object
        prompt = (
            f"You are a helpful assistant. Your task is to map a list of input entities to the closest matching entities from a predefined dictionary. "
            f"For each input entity, find the single most appropriate term from the dictionary.\n\n"
            f"**Predefined Dictionary:**\n{self.known_entities_prompt_segment}\n\n"
            f"**Input Entities to Map:**\n{json.dumps(entities_to_ground)}\n\n"
            f"Provide your answer ONLY as a valid JSON object mapping each input entity to its corresponding dictionary term. "
            f"The keys of the JSON must be the original input entities. "
            f"Example format: {{\"input entity 1\": \"dictionary term 1\", \"input entity 2\": \"dictionary term 2\"}}"
        )
        
        try:
            # Sử dụng generation_config đã được định nghĩa trong __init__
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Xử lý response text để đảm bảo nó là JSON hợp lệ
            raw_response_text = response.text.strip()
            if raw_response_text.startswith("```json"):
                raw_response_text = raw_response_text[7:]
            if raw_response_text.endswith("```"):
                raw_response_text = raw_response_text[:-3]

            # Parse chuỗi JSON thành dictionary
            grounding_map = json.loads(raw_response_text)
            print(f"    -> Kết quả Grounding Map: {grounding_map}")
            
            # Kiểm tra xem output có phải là dictionary không
            if not isinstance(grounding_map, dict):
                print(f"--- ⚠️ Lỗi Grounding: Gemini không trả về dictionary. Fallback. ---")
                return {}

            return grounding_map

        except (json.JSONDecodeError, Exception) as e:
            print(f"--- ⚠️ Lỗi trong quá trình Semantic Grounding: {e} ---")
            # Fallback: Trả về mapping rỗng nếu có lỗi
            return {}