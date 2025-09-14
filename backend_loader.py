import json
import os
import glob
from search_core.basic_searcher import BasicSearcher
from search_core.master_searcher import MasterSearcher
from sentence_transformers import SentenceTransformer
from search_core.transcript_searcher import TranscriptSearcher

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
    
    
    print("--- 1/3: Quét và lập bản đồ đường dẫn video (Hỗ trợ đa Batch)... ---")
    all_video_files = []
    for path in VIDEO_BASE_PATHS:
        if os.path.isdir(path):
            print(f"   -> Đang quét thư mục video: {path}")
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

    print("--- 2/3: Khởi tạo TranscriptSearcher (Tai Thính)... ---")
    METADATA_V6_COMBINED_PATH = RERANK_METADATA_PATH 
    if not os.path.exists(METADATA_V6_COMBINED_PATH):
         print(f"--- ⚠️ CẢNH BÁO: Không tìm thấy file metadata hợp nhất tại {METADATA_V6_COMBINED_PATH}. TranscriptSearcher sẽ không hoạt động. ---")
         transcript_searcher = None
    else:
        transcript_searcher = TranscriptSearcher(metadata_path=METADATA_V6_COMBINED_PATH)
    print("--- ✅ TranscriptSearcher đã sẵn sàng. ---")
    
    print("--- 3/3: Tải Bản đồ FPS đã Hợp nhất... ---")
    fps_map = {}
    
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
    
    return {
        "master_searcher": master_searcher,
        "transcript_searcher": transcript_searcher,
        "fps_map": fps_map,
        "video_path_map": video_path_map 
    }