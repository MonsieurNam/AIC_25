import pandas as pd
import os
from typing import Optional

class TranscriptSearcher:
    """
    Một công cụ tìm kiếm chuyên dụng, hiệu năng cao trên dữ liệu transcript.
    Nó tải trước toàn bộ dữ liệu vào bộ nhớ để thực hiện các thao tác
    lọc và tìm kiếm lồng nhau một cách gần như tức thời.
    """
    def __init__(self, metadata_path: str):
        """
        Khởi tạo TranscriptSearcher bằng cách tải và chuẩn bị dữ liệu.
        PHIÊN BẢN NÂNG CẤP: Tự động làm sạch (strip) dữ liệu transcript.

        Args:
            metadata_path (str): Đường dẫn đến file rerank_metadata_v6.parquet.
        """
        print("--- 🧠 Khởi tạo Transcript Searcher (Động cơ 'Tai Thính')... ---")
        self.full_data: Optional[pd.DataFrame] = None
        
        try:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"File metadata không tồn tại tại: {metadata_path}")
            
            cols_to_load = [
                'video_id', 'timestamp', 'transcript_text', 'keyframe_path'
            ]
            
            print(f"-> Đang tải dữ liệu transcript từ {metadata_path}...")
            self.full_data = pd.read_parquet(metadata_path, columns=cols_to_load)
            print("-> Đang làm sạch (strip) và lọc dữ liệu transcript...")
            self.full_data['transcript_text'] = self.full_data['transcript_text'].str.strip()
            self.full_data.dropna(subset=['transcript_text'], inplace=True)
            self.full_data = self.full_data[self.full_data['transcript_text'] != ''].copy()
            self.full_data.reset_index(drop=True, inplace=True)
            
            print(f"--- ✅ Transcript Searcher đã nạp và chuẩn bị {len(self.full_data)} dòng transcript sạch. Sẵn sàng hoạt động! ---")

        except Exception as e:
            print(f"--- ❌ LỖI NGHIÊM TRỌNG khi khởi tạo TranscriptSearcher: {e} ---")
            
    def search(self, search_term: str, current_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Thực hiện tìm kiếm trên transcript. (Logic không thay đổi)
        """
        if self.full_data is None:
            print("--- ⚠️ TranscriptSearcher chưa được khởi tạo thành công. Bỏ qua tìm kiếm. ---")
            return pd.DataFrame()
        
        if not search_term or not search_term.strip():
            return current_results if current_results is not None else self.full_data

        source_df = self.full_data if current_results is None else current_results
        
        filtered_df = source_df[
            source_df['transcript_text'].str.contains(search_term, case=False, na=False)
        ].copy()
        
        return filtered_df