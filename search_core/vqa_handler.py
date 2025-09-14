import time
import re
import json
import hashlib
from collections import deque
from typing import Dict, Optional, Any

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image


class RateLimiter:
    """Token-bucket đơn giản cho giới hạn requests-per-minute."""
    def __init__(self, rpm_limit: int = 12):
        self.capacity = max(1, rpm_limit)
        self.window = 60.0
        self.events = deque()  

    def acquire(self):
        now = time.time()
        while self.events and now - self.events[0] > self.window:
            self.events.popleft()
        if len(self.events) >= self.capacity:
            sleep_s = self.window - (now - self.events[0]) + 0.01
            if sleep_s > 0:
                time.sleep(sleep_s)
            return self.acquire()
        self.events.append(time.time())


class VQAHandler:
    """
    Xử lý Visual Question Answering (VQA) dùng Gemini.
    - Có giới hạn tốc độ (RPM)
    - Tự retry khi gặp 429 và tôn trọng retry_delay
    - Cache kết quả theo (image_sha, question)
    """

    def __init__(
        self,
        model: Optional[genai.GenerativeModel] = None,
        model_name: str = "gemini-2.5-flash",
        rpm_limit: int = 12,
        max_retries: int = 3,
    ):
        """
        Args:
            model: Có thể truyền vào một instance Gemini dùng chung toàn hệ thống.
            model_name: tên model nếu không truyền sẵn 'model'.
            rpm_limit: giới hạn request/phút để tránh 429 (free tier thường ~15).
            max_retries: số lần retry tối đa khi gặp 429/resource exhausted.
        """
        if model is not None:
            self.model = model
        else:
            print(f"--- VQA Handler: Khởi tạo model mới '{model_name}' ---")
            self.model = genai.GenerativeModel(model_name)

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.limiter = RateLimiter(rpm_limit=rpm_limit)
        self.max_retries = max_retries
        self._cache: Dict[tuple, Dict[str, Any]] = {}


    def _file_sha1(self, path: str) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def _estimate_conf(self, resp_text: str) -> float:
        """
        Heuristic: nếu có JSON với 'confidence', sẽ dùng giá trị đó.
        Nếu fallback text: gán 0.5 mặc định (đã xử lý ở dưới).
        """
        m = re.search(r"\{.*\}", resp_text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "confidence" in obj:
                    c = float(obj["confidence"])
                    return max(0.0, min(1.0, c))
            except Exception:
                pass
        return 0.5

    def _parse_json_answer(self, response_text: str) -> Dict[str, Any]:
        """
        Cố gắng trích JSON {"answer": str, "confidence": float} từ response.
        Nếu thất bại, fallback sang text thuần + confidence 0.5.
        """
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return {"answer": response_text.strip(), "confidence": 0.5}
        try:
            payload = json.loads(match.group(0))
            answer = str(payload.get("answer", "Không có câu trả lời"))
            conf = float(payload.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
            return {"answer": answer, "confidence": conf}
        except Exception:
            return {"answer": response_text.strip(), "confidence": 0.5}


    def get_answer(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Args:
            image_path: đường dẫn ảnh keyframe
            question: câu hỏi VQA (tiếng Việt)
        Returns:
            {"answer": str, "confidence": float}
        """
        try:
            key = (self._file_sha1(image_path), question.strip())
        except FileNotFoundError:
            print(f"--- ⚠️ Lỗi VQA: Không tìm thấy file ảnh: '{image_path}' ---")
            return {"answer": "Lỗi: Không tìm thấy ảnh", "confidence": 0.0}
        except Exception as e:
            print(f"--- ⚠️ Lỗi VQA: Không thể đọc ảnh '{image_path}'. Lỗi: {e} ---")
            return {"answer": "Lỗi: Ảnh bị hỏng", "confidence": 0.0}

        if key in self._cache:
            return self._cache[key]

        try:
            img = Image.open(image_path)
        except FileNotFoundError:
            print(f"--- ⚠️ Lỗi VQA: Không tìm thấy file ảnh tại '{image_path}' ---")
            return {"answer": "Lỗi: Không tìm thấy ảnh", "confidence": 0.0}
        except Exception as e:
            print(f"--- ⚠️ Lỗi VQA: Không thể mở ảnh '{image_path}'. Lỗi: {e} ---")
            return {"answer": "Lỗi: Ảnh bị hỏng", "confidence": 0.0}

        prompt = f"""
Bạn là trợ lý VQA. Hãy xem ảnh và trả lời NGẮN GỌN bằng **tiếng Việt** cho câu hỏi của người dùng.
- Trả về DUY NHẤT một JSON hợp lệ với 2 trường: "answer" (chuỗi) và "confidence" (0.0 → 1.0).
- Không thêm giải thích nào khác ngoài JSON.

**Câu hỏi:** "{question}"

**JSON trả lời:**
""".strip()

        attempt = 0
        while True:
            self.limiter.acquire()
            try:
                resp = self.model.generate_content(
                    [prompt, img],
                    safety_settings=self.safety_settings,
                )
                text = (resp.text or "").strip()
                out = self._parse_json_answer(text)
                self._cache[key] = out
                return out

            except Exception as e:
                msg = str(e)
                if ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg):
                    attempt += 1
                    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
                    delay = int(m.group(1)) if m else 15
                    if attempt > self.max_retries:
                        print(f"--- ⚠️ Lỗi VQA: Quá số lần retry ({self.max_retries}). Trả về low-confidence. ---")
                        out = {"answer": "", "confidence": 0.0}
                        self._cache[key] = out
                        return out
                    print(f"--- ⚠️ 429/Resource exhausted. Đợi {delay}s rồi thử lại (attempt {attempt}/{self.max_retries}) ---")
                    time.sleep(delay + 0.5)
                    continue

                print(f"--- ⚠️ Lỗi VQA: Lỗi khi gọi Gemini/parse JSON. Lỗi: {e} ---")
                return {"answer": "Lỗi xử lý VQA", "confidence": 0.0}
