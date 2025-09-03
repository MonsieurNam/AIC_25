# /AIC25_Video_Search_Engine/backend_loader.py

import json
import os
import glob
from search_core.basic_searcher import BasicSearcher
from search_core.master_searcher import MasterSearcher
from sentence_transformers import SentenceTransformer
from config import (
    VIDEO_BASE_PATH, FAISS_INDEX_PATH, RERANK_METADATA_PATH, 
    CLIP_FEATURES_PATH, ALL_ENTITIES_PATH, OPENAI_API_KEY, GEMINI_API_KEY
)
from search_core.transcript_searcher import TranscriptSearcher

def initialize_backend():
    """
    Khởi tạo và trả về một dictionary chứa TẤT CẢ các instance backend cần thiết.
    """
    print("--- 🚀 Giai đoạn 2/4: Đang cấu hình và khởi tạo TOÀN BỘ Backend... ---")
    
    # --- 1. Master Searcher (Visual Scout Engine) ---
    print("--- 1/3: Quét và lập bản đồ đường dẫn video... ---")
    all_video_files = glob.glob(os.path.join(VIDEO_BASE_PATH, "**", "*.mp4"), recursive=True)
    video_path_map = {os.path.basename(f).replace('.mp4', ''): f for f in all_video_files}
    print(f"--- ✅ Lập bản đồ thành công cho {len(video_path_map)} video. ---")
    print("   -> Đang tải mô hình Bi-Encoder tiếng Việt cho Reranking...")
    try:
        # Thay thế bằng tên model rerank của bạn nếu khác
        rerank_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device='cuda')
        print("--- ✅ Tải model Bi-Encoder thành công! ---")
    except Exception as e:
        print(f"--- ❌ Lỗi nghiêm trọng khi tải model Rerank: {e}. Hệ thống có thể không hoạt động đúng. ---")
        rerank_model = None
    
    basic_searcher = BasicSearcher(
        faiss_index_path=FAISS_INDEX_PATH, 
        metadata_path=RERANK_METADATA_PATH
    )
    master_searcher = MasterSearcher(basic_searcher=basic_searcher, rerank_model=rerank_model, openai_api_key=OPENAI_API_KEY, gemini_api_key=GEMINI_API_KEY, entities_path=ALL_ENTITIES_PATH, clip_features_path=CLIP_FEATURES_PATH, video_path_map=video_path_map)    
    print("--- ✅ MasterSearcher đã sẵn sàng. ---")

    # --- 2. Transcript Searcher (Transcript Intel Engine) ---
    print("--- 2/3: Khởi tạo TranscriptSearcher (Tai Thính)... ---")
    # Sử dụng file metadata v6 đã được làm giàu
    # METADATA_V6_PATH = RERANK_METADATA_PATH.replace("_v5.parquet", "_v6.parquet")
    METADATA_V6_PATH = "/kaggle/input/stage1/rerank_metadata_v6.parquet"
    if not os.path.exists(METADATA_V6_PATH):
         print(f"--- ⚠️ CẢNH BÁO: Không tìm thấy file metadata v6 tại {METADATA_V6_PATH}. TranscriptSearcher sẽ không hoạt động. ---")
         transcript_searcher = None
    else:
        transcript_searcher = TranscriptSearcher(metadata_path=METADATA_V6_PATH)
    print("--- ✅ TranscriptSearcher đã sẵn sàng. ---")
    
    # --- 3. Tải Bản đồ FPS (Submission Engine Prerequisite) ---
    print("--- 3/3: Tải Bản đồ FPS... ---")
    fps_map = {}
    fps_map_path = "/kaggle/input/stage1/video_fps_map.json"
    if os.path.exists(fps_map_path):
        try:
            with open(fps_map_path, 'r') as f:
                fps_map = json.load(f)
            print(f"--- ✅ Tải thành công bản đồ FPS cho {len(fps_map)} video. ---")
        except Exception as e:
            print(f"--- ❌ Lỗi khi tải FPS map: {e}. Sẽ sử dụng FPS mặc định. ---")
    else:
        print(f"--- ⚠️ Không tìm thấy file FPS map tại {fps_map_path}. Sẽ sử dụng FPS mặc định. ---")

    print("--- ✅ Backend đã khởi tạo thành công! Hạm đội sẵn sàng. ---")
    
    # Trả về một dictionary để dễ dàng truy cập
    return {
        "master_searcher": master_searcher,
        "transcript_searcher": transcript_searcher,
        "fps_map": fps_map,
        "video_path_map": video_path_map 
    }