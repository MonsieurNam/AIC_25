# /search_core/query_decomposer.py

import json
import re
from typing import List, TYPE_CHECKING

# Sử dụng TYPE_CHECKING để tránh lỗi circular import, một kỹ thuật rất hữu ích trong các dự án lớn.
if TYPE_CHECKING:
    from .gemini_text_handler import GeminiTextHandler

# ==============================================================================
# === PROMPT VÀNG (THE GOLDEN PROMPT) - TRÁI TIM CỦA BỘ NÃO ===
# ==============================================================================
# Prompt này được thiết kế cực kỳ cẩn thận để "ép" Gemini suy nghĩ theo đúng
# định hướng "Co-pilot" của chúng ta: phân rã thành các yếu tố visual đơn giản.
SYSTEM_PROMPT = """
Bạn là một chuyên gia phân tích truy vấn cho hệ thống tìm kiếm video. Nhiệm vụ của bạn là PHÂN RÃ một truy vấn phức tạp của người dùng thành một danh sách các MÔ TẢ HÌNH ẢNH (visual description) đơn giản, độc lập và có thể tìm kiếm được.

**QUY TẮC BẮT BUỘC:**
1.  **KHÔNG Diễn giải:** Loại bỏ hoàn toàn các khái niệm trừu tượng, cảm xúc, ý định (ví dụ: 'đẹp', 'lung linh', 'quyết tâm', 'thống kê cho tác dụng phụ').
2.  **KHÔNG Đọc Chữ (OCR):** Nếu truy vấn chứa văn bản hoặc số (ví dụ: '9.00€', 'Happy New Year', '5,8%'), hãy mô tả vật thể chứa văn bản đó (ví dụ: 'bảng giá', 'chiếc kính', 'biểu đồ').
3.  **KHÔNG Đếm Chính xác:** Chuyển các yêu cầu đếm (ví dụ: 'hàng trăm ngọn nến', '12 cái bánh') thành các mô tả định tính (ví dụ: 'nhiều ngọn nến', 'bánh ngọt xếp trên khay').
4.  **TẬP TRUNG vào Danh từ & Tính từ Cốt lõi:** Chỉ giữ lại các đối tượng, màu sắc, hình dạng, và hành động đơn giản có thể nhìn thấy được.
5.  **ĐỘC LẬP:** Mỗi mô tả trong danh sách trả về phải có thể được tìm kiếm một cách độc lập.

**ĐỊNH DẠNG TRẢ VỀ:** Chỉ trả về một mảng JSON chứa các chuỗi mô tả. KHÔNG thêm bất kỳ giải thích nào khác.

---
**VÍ DỤ 1:**
Query: "Đoạn clip mở đầu với cảnh đài phun nước được bao quanh bởi hàng trăm ngọn nến xếp ngay ngắn, tạo nên khung cảnh lung linh giữa đám đông."
JSON:
[
    "đài phun nước vào ban đêm",
    "nhiều ngọn nến được thắp sáng",
    "đám đông người tụ tập"
]

**VÍ DỤ 2:**
Query: "Hình ảnh những chiếc bánh có 2 mức giá được viết mẫu giấy đen nhỏ lần lượt là 9.00€ và 4€.80."
JSON:
[
    "bánh ngọt trưng bày trong tủ kính",
    "bảng giá nhỏ màu đen"
]

**VÍ DỤ 3:**
Query: "Cảnh phỏng vấn tổng thống Donald Trump, đằng sau ông là các bức hình và những lá cờ. Sau đó là cảnh một đoàn người đang di chuyển trên đường dọc theo một bờ biển."
JSON:
[
    "tổng thống Donald Trump đang được phỏng vấn",
    "lá cờ Mỹ phía sau",
    "đoàn người diễu hành gần bờ biển"
]
---

Bây giờ, hãy phân rã truy vấn sau:
"""

class QueryDecomposer:
    """
    "Bộ não" mới của hệ thống, chịu trách nhiệm phân rã một truy vấn phức tạp
    thành nhiều truy vấn con đơn giản, tập trung vào các yếu tố hình ảnh.
    """
    def __init__(self, gemini_handler: 'GeminiTextHandler'):
        """
        Khởi tạo bộ phân rã với một Gemini handler đã được cấu hình.
        """
        print("--- 🧠 Khởi tạo Query Decomposer (Bộ não Chiến dịch PHOENIX REBORN) ---")
        self.gemini_handler = gemini_handler

    def decompose(self, query: str) -> List[str]:
        """
        Phân rã truy vấn chính.

        Args:
            query (str): Truy vấn gốc từ người dùng.

        Returns:
            List[str]: Một danh sách các truy vấn con đã được đơn giản hóa.
                       Nếu có lỗi xảy ra, sẽ trả về một danh sách chỉ chứa truy vấn gốc.
        """
        # --- Bước 1: Kiểm tra đầu vào ---
        if not self.gemini_handler or not query or not query.strip():
            # Nếu không có handler hoặc query rỗng, trả về chính nó để hệ thống không bị lỗi
            return [query] if query and query.strip() else []

        print(f"   -> [Decomposer] Bắt đầu phân rã truy vấn: '{query}'")
        
        # --- Bước 2: Gọi API Gemini với cơ chế retry đã có ---
        try:
            user_prompt = f"Query: \"{query}\""
            # Tận dụng lại hàm gọi API đã có sẵn retry từ GeminiTextHandler
            response = self.gemini_handler._gemini_api_call([SYSTEM_PROMPT, user_prompt])
            raw_text = response.text.strip()
            
            # --- Bước 3: Xử lý response một cách an toàn ---
            # Gemini đôi khi trả về trong khối markdown, cần trích xuất JSON từ đó
            match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if not match:
                print(f"   -> ⚠️ [Decomposer] Gemini không trả về mảng JSON hợp lệ. Fallback. Raw response: {raw_text}")
                return [query]
            
            try:
                sub_queries = json.loads(match.group(0))
                if isinstance(sub_queries, list) and all(isinstance(i, str) for i in sub_queries):
                    print(f"   -> ✅ [Decomposer] Phân rã thành công: {sub_queries}")
                    return sub_queries
                else:
                    print(f"   -> ⚠️ [Decomposer] JSON trả về không phải là một danh sách chuỗi. Fallback.")
                    return [query]
            except json.JSONDecodeError:
                print(f"   -> ⚠️ [Decomposer] Lỗi giải mã JSON. Fallback. Raw match: {match.group(0)}")
                return [query]

        except Exception as e:
            print(f"--- ❌ [Decomposer] Lỗi nghiêm trọng khi gọi API Gemini: {e}. Sử dụng fallback. ---")
            # Fallback tối quan trọng: nếu có bất kỳ lỗi gì, hệ thống vẫn chạy được
            # bằng cách sử dụng chính truy vấn gốc.
            return [query]