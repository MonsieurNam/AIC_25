import re
from enum import Enum
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional

class TaskType(Enum):
    """
    Enum để đại diện cho các loại nhiệm vụ tìm kiếm khác nhau.
    """
    KIS = "Textual KIS"
    QNA = "Question Answering"
    TRAKE = "Action Keyframe Tracking"
    
def analyze_query_heuristic(query: str) -> TaskType:
    """
    Phân loại truy vấn bằng phương pháp heuristic dựa trên regex và từ khóa.

    Đây là phương pháp nhanh, không cần API call, và hoạt động tốt cho các trường hợp rõ ràng.
    Nó đóng vai trò là một cơ chế dự phòng tin cậy.

    Args:
        query (str): Câu truy vấn của người dùng.

    Returns:
        TaskType: Loại nhiệm vụ được phân loại (KIS, QNA, hoặc TRAKE).
    """
    if not isinstance(query, str) or not query:
        return TaskType.KIS 

    query_lower = query.lower().strip()

    qna_start_keywords = [
        'màu gì', 'ai là', 'ai đang', 'ở đâu', 'khi nào', 'tại sao',
        'cái gì', 'bao nhiêu', 'có bao nhiêu', 'hành động gì', 'đang làm gì'
    ]
    if '?' in query or any(query_lower.startswith(k) for k in qna_start_keywords):
        return TaskType.QNA

    trake_keywords = ['tìm các khoảnh khắc', 'tìm những khoảnh khắc', 'chuỗi hành động', 'các bước']
    trake_pattern = r'\(\d+\)|bước \d+|\d\.'
    if any(k in query_lower for k in trake_keywords) or re.search(trake_pattern, query_lower):
        return TaskType.TRAKE

    return TaskType.KIS

def analyze_query_gemini(query: str, model: Optional[genai.GenerativeModel] = None) -> TaskType:
    """
    Phân loại truy vấn bằng mô hình Gemini để có độ chính xác cao hơn với các câu phức tạp.

    Args:
        query (str): Câu truy vấn của người dùng.
        model (genai.GenerativeModel, optional): Instance của Gemini model đã được khởi tạo.
                                                 Nếu là None, sẽ fallback về heuristic.

    Returns:
        TaskType: Loại nhiệm vụ được phân loại.
    """
    if not model:
        return analyze_query_heuristic(query)
    
    prompt = f"""
    Analyze the following Vietnamese user query for a video search system. Classify it into one of three types:
    1.  "KIS": The user is looking for a single, specific moment or scene. (e.g., "a man opening a laptop", "a red car on the street")
    2.  "QNA": The user is asking a direct question about a scene that needs an answer. (e.g., "What color is the woman's dress?", "Who is speaking on the stage?")
    3.  "TRAKE": The user is looking for a sequence of multiple distinct moments in order. (e.g., "Find the moments of: (1) jumping, (2) landing", "a person stands up, walks, and then sits down")

    Return ONLY the type as a single word: KIS, QNA, or TRAKE.

    Query: "{query}"
    Type:
    """
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        result = response.text.strip().upper()
        
        if "QNA" in result:
            return TaskType.QNA
        if "TRAKE" in result:
            return TaskType.TRAKE
        return TaskType.KIS
        
    except Exception as e:
        print(f"--- ⚠️ Lỗi khi gọi API phân loại của Gemini: {e}. Sử dụng fallback heuristic. ---")
        return analyze_query_heuristic(query)
