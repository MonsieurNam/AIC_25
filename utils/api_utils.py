import time
import random
from functools import wraps

def api_retrier(max_retries=5, initial_delay=1, backoff_factor=2, jitter=0.1):
    """
    Một decorator để tự động thử lại các lệnh gọi API Gemini khi gặp lỗi 429.

    Sử dụng thuật toán Exponential Backoff with Jitter để tránh làm quá tải API.

    Args:
        max_retries (int): Số lần thử lại tối đa.
        initial_delay (int): Thời gian chờ ban đầu (giây).
        backoff_factor (float): Hệ số nhân cho thời gian chờ sau mỗi lần thất bại.
        jitter (float): Hệ số ngẫu nhiên để tránh các client thử lại cùng lúc.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if 'resource has been exhausted' in error_str or 'rate limit' in error_str or '429' in error_str:
                        if i == max_retries - 1:
                            print(f"--- ❌ API Call Thất bại sau {max_retries} lần thử. Bỏ qua. Lỗi: {e} ---")
                            raise e

                        jitter_value = delay * jitter * random.uniform(-1, 1)
                        wait_time = delay + jitter_value
                        
                        print(f"--- ⚠️ API Rate Limit. Thử lại lần {i+1}/{max_retries} sau {wait_time:.2f} giây... ---")
                        time.sleep(wait_time)
                        delay *= backoff_factor
                    else:
                        raise e
            return None
        return wrapper
    return decorator