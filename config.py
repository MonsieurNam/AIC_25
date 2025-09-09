# /AIC25_Video_Search_Engine/config.py

import os
from kaggle_secrets import UserSecretsClient

# --- Hằng số Giao diện & Tìm kiếm ---
ITEMS_PER_PAGE = 20
MAX_SUBMISSION_RESULTS = 100

# --- Đường dẫn tới các file dữ liệu ---
KAGGLE_INPUT_DIR = '/kaggle/input'
KAGGLE_WORKING_DIR = '/kaggle/working'
# UNIFIED_DATA_DIR = os.path.join(KAGGLE_WORKING_DIR, 'unified_data')

CLIP_FEATURES_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/features_combined.npy')
FAISS_INDEX_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/faiss_combined.index')
RERANK_METADATA_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/rerank_metadata_v6_combined.parquet')
ALL_ENTITIES_PATH = os.path.join(KAGGLE_INPUT_DIR, 'stage1/all_entities_combined.json') 

# VIDEO_BASE_PATH = os.path.join(KAGGLE_INPUT_DIR, 'aic2025-batch-1-video/')
TRANSCRIPTS_JSON_DIR = os.path.join(KAGGLE_INPUT_DIR, 'aic25-transcripts/transcripts') 
# KEYFRAME_BASE_PATH = os.path.join(KAGGLE_INPUT_DIR, 'aic25-keyframes-and-metadata/keyframes/')
# KEYFRAME_BASE_PATH = os.path.join(UNIFIED_DATA_DIR, 'keyframes')

VIDEO_BASE_PATHS = [
    # Batch 1
    '/kaggle/input/aic2025-batch-1-video/',
    # Batch 2 (Part 1: K01-K10)
    '/kaggle/input/video-part1-batch2-aic25/Videos_K01/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K02/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K03/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K04/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K05/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K06/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K07/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K08/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K09/video/',
    '/kaggle/input/video-part1-batch2-aic25/Videos_K10/video/',
    # Batch 2 (Part 2: K11-K20)
    '/kaggle/input/batch2-k11-k20/Videos_K11/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K12/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K13/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K14/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K15/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K16/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K17/video/',  
    '/kaggle/input/batch2-k11-k20/Videos_K18/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K19/video/',
    '/kaggle/input/batch2-k11-k20/Videos_K20/video/',
]

# --- 3. DANH SÁCH CÁC THƯ MỤC KEYFRAME TỪ CẢ HAI BATCH ---
KEYFRAME_BASE_PATHS = [
    # Batch 1
    '/kaggle/input/aic25-keyframes-and-metadata/keyframes/',
    # Batch 2
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K01/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K02/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K03/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K04/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K05/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K06/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K07/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K08/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K09/keyframes/',
    '/kaggle/input/aic25-keyframes-k01-k20/Keyframes_K10/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K11/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K12/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K13/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K14/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K15/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K16/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K17/keyframes/',  
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K18/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K19/keyframes/',
    '/kaggle/input/aic25-keyframes-k11-k20/Keyframes_K20/keyframes/',
]

def load_api_keys():
    """
    Tải API keys từ Kaggle Secrets một cách an toàn.
    Trả về một tuple chứa (openai_api_key, gemini_api_key).
    """
    openai_key, gemini_key = None, None
    try:
        user_secrets = UserSecretsClient()
        openai_key = user_secrets.get_secret("OPENAI_API_KEY")
        print("--- ✅ Cấu hình OpenAI API Key thành công! ---")
    except Exception:
        print("--- ⚠️ Không tìm thấy OpenAI API Key. ---")
        
    try:
        user_secrets = UserSecretsClient()
        gemini_key = user_secrets.get_secret("GOOGLE_API_KEY")
        print("--- ✅ Cấu hình GEMINI API Key thành công! ---")
    except Exception:
        print("--- ⚠️ Không tìm thấy GEMINI API Key. ---")
        
    return openai_key, gemini_key

# Tải keys khi module được import
OPENAI_API_KEY, GEMINI_API_KEY = load_api_keys()