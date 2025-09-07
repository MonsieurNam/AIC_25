# ==============================================================================
# BACKEND LOADER - PHIÊN BẢN HỖ TRỢ ĐA BATCH
# File: /AIC_25/backend_loader.py
#
# MỤC TIÊU:
#   - Khởi tạo tất cả các thành phần backend cần thiết cho ứng dụng.
#   - Đọc cấu hình từ file config.py đã được nâng cấp.
#   - Quét nhiều thư mục video để tạo bản đồ đường dẫn video hoàn chỉnh.
#   - Tải các tài sản dữ liệu đã hợp nhất (metadata, FPS map, ...).
# ==============================================================================

import json
import os
import glob
from search_core.basic_searcher import BasicSearcher
from search_core.master_searcher import MasterSearcher
from sentence_transformers import SentenceTransformer
from search_core.transcript_searcher import TranscriptSearcher

# ✅ Import các danh sách đường dẫn và biến cấu hình đã được nâng cấp từ config.py
from config import (
    VIDEO_BASE_PATHS, 
    FAISS_INDEX_PATH, 
    RERANK_METADATA_PATH, 
    CLIP_FEATURES_PATH, 
    ALL_ENTITIES_PATH, 
    OPENAI_API_KEY, 
    GEMINI_API_KEY
)


def initialize_backend():
    """
    Khởi tạo và trả về một dictionary chứa TẤT CẢ các instance backend cần thiết.
    PHIÊN BẢN HỖ TRỢ ĐA BATCH.
    """
    print("--- 🚀 Giai đoạn 2/4: Đang cấu hình và khởi tạo Backend (Chế độ Đa Batch)... ---")
    
    # --- 1. Master Searcher (Visual Scout Engine) ---
    
    # ✅ NÂNG CẤP GIAI ĐOẠN 3: Quét và lập bản đồ đường dẫn video từ nhiều thư mục
    print("--- 1/3: Quét và lập bản đồ đường dẫn video (Hỗ trợ đa Batch)... ---")
    all_video_files = []
    # Lặp qua danh sách các thư mục video đã định nghĩa trong config.py
    for path in VIDEO_BASE_PATHS:
        if os.path.isdir(path):
            print(f"   -> Đang quét thư mục video: {path}")
            # Quét đệ quy để tìm tất cả các file .mp4
            all_video_files.extend(glob.glob(os.path.join(path, "**", "*.mp4"), recursive=True))
        else:
            print(f"   -> ⚠️ Cảnh báo: Thư mục video '{path}' trong config không tồn tại. Bỏ qua.")
            
    video_path_map = {os.path.basename(f).replace('.mp4', ''): f for f in all_video_files}
    print(f"--- ✅ Lập bản đồ thành công cho {len(video_path_map)} video từ cả hai batch. ---")
    
    print("   -> Đang tải mô hình Bi-Encoder tiếng Việt cho Reranking...")
    try:
        rerank_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device='cuda')
        print("--- ✅ Tải model Bi-Encoder thành công! ---")
    except Exception as e:
        print(f"--- ❌ Lỗi nghiêm trọng khi tải model Rerank: {e}. Hệ thống có thể không hoạt động đúng. ---")
        rerank_model = None
    
    # Các hàm khởi tạo này sẽ tự động sử dụng các đường dẫn _combined đã được import từ config
    basic_searcher = BasicSearcher(
        faiss_index_path=FAISS_INDEX_PATH, 
        metadata_path=RERANK_METADATA_PATH
    )
    master_searcher = MasterSearcher(
        basic_searcher=basic_searcher, 
        rerank_model=rerank_model, 
        openai_api_key=OPENAI_API_KEY, 
        gemini_api_key=GEMINI_API_KEY, 
        entities_path=ALL_ENTITIES_PATH, 
        clip_features_path=CLIP_FEATURES_PATH, 
        video_path_map=video_path_map
    )    
    print("--- ✅ MasterSearcher đã sẵn sàng. ---")

    # --- 2. Transcript Searcher (Transcript Intel Engine) ---
    print("--- 2/3: Khởi tạo TranscriptSearcher (Tai Thính)... ---")
    # METADATA_V6_PATH bây giờ chính là RERANK_METADATA_PATH đã hợp nhất
    METADATA_V6_COMBINED_PATH = RERANK_METADATA_PATH 
    if not os.path.exists(METADATA_V6_COMBINED_PATH):
         print(f"--- ⚠️ CẢNH BÁO: Không tìm thấy file metadata hợp nhất tại {METADATA_V6_COMBINED_PATH}. TranscriptSearcher sẽ không hoạt động. ---")
         transcript_searcher = None
    else:
        # TranscriptSearcher giờ sẽ làm việc trên dữ liệu của cả 2 batch
        transcript_searcher = TranscriptSearcher(metadata_path=METADATA_V6_COMBINED_PATH)
    print("--- ✅ TranscriptSearcher đã sẵn sàng. ---")
    
    # --- 3. Tải Bản đồ FPS (Submission Engine Prerequisite) ---
    print("--- 3/3: Tải Bản đồ FPS đã Hợp nhất... ---")
    fps_map = {}
    
    # ✅ NÂNG CẤP GIAI ĐOẠN 3: Tải file FPS map đã hợp nhất từ /kaggle/working/
    fps_map_path = "/kaggle/input/stage1/video_fps_map_combined.json" 
    
    if os.path.exists(fps_map_path):
        try:
            with open(fps_map_path, 'r') as f:
                fps_map = json.load(f)
            print(f"--- ✅ Tải thành công bản đồ FPS hợp nhất cho {len(fps_map)} video. ---")
        except Exception as e:
            print(f"--- ❌ Lỗi khi tải FPS map hợp nhất: {e}. Sẽ sử dụng FPS mặc định. ---")
    else:
        print(f"--- ⚠️ Không tìm thấy file FPS map hợp nhất tại {fps_map_path}. Sẽ sử dụng FPS mặc định. ---")

    print("\n--- ✅ Backend đã khởi tạo thành công! Hạm đội sẵn sàng chiến đấu trên mọi mặt trận. ---")
    
    # Trả về một dictionary chứa các đối tượng đã được khởi tạo
    return {
        "master_searcher": master_searcher,
        "transcript_searcher": transcript_searcher,
        "fps_map": fps_map,
        "video_path_map": video_path_map 
    }