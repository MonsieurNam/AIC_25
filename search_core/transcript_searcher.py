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

        Args:
            metadata_path (str): Đường dẫn đến file rerank_metadata_v6.parquet.
        """
        print("--- 🧠 Khởi tạo Transcript Searcher (Động cơ 'Tai Thính')... ---")
        self.full_data: Optional[pd.DataFrame] = None
        
        try:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"File metadata không tồn tại tại: {metadata_path}")
            
            # Tải toàn bộ dữ liệu, nhưng chỉ giữ lại các cột cần thiết để tiết kiệm RAM
            all_columns = pd.read_parquet(metadata_path, columns=['keyframe_id']) # Đọc nhanh 1 cột để lấy list
            cols_to_load = [
                'video_id', 'timestamp', 'transcript_text', 'keyframe_path'
            ]
            
            print(f"-> Đang tải dữ liệu transcript từ {metadata_path}...")
            self.full_data = pd.read_parquet(metadata_path, columns=cols_to_load)
            
            # Loại bỏ các dòng không có transcript để tối ưu tìm kiếm
            self.full_data = self.full_data[self.full_data['transcript_text'] != ''].copy()
            
            # Reset index để thao tác dễ dàng hơn
            self.full_data.reset_index(drop=True, inplace=True)
            
            print(f"--- ✅ Transcript Searcher đã nạp và chuẩn bị {len(self.full_data)} dòng transcript. Sẵn sàng hoạt động! ---")

        except Exception as e:
            print(f"--- ❌ LỖI NGHIÊM TRỌNG khi khởi tạo TranscriptSearcher: {e} ---")
            # Nếu có lỗi, self.full_data sẽ vẫn là None, các hàm sau sẽ xử lý
            
    def search(self, search_term: str, current_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Thực hiện tìm kiếm trên transcript.

        Args:
            search_term (str): Từ khóa tìm kiếm.
            current_results (pd.DataFrame, optional): DataFrame kết quả từ lần
                tìm kiếm trước. Nếu là None, tìm kiếm trên toàn bộ dữ liệu.

        Returns:
            pd.DataFrame: Một DataFrame mới chứa các kết quả đã được lọc.
        """
        if self.full_data is None:
            print("--- ⚠️ TranscriptSearcher chưa được khởi tạo thành công. Bỏ qua tìm kiếm. ---")
            return pd.DataFrame() # Trả về DataFrame rỗng
        
        if not search_term or not search_term.strip():
            return current_results if current_results is not None else self.full_data

        # Xác định nguồn dữ liệu để tìm kiếm
        source_df = self.full_data if current_results is None else current_results
        
        # Thực hiện tìm kiếm không phân biệt hoa thường và bỏ qua các giá trị NaN
        # `str.contains` là phương thức cốt lõi, cực kỳ nhanh của Pandas
        filtered_df = source_df[
            source_df['transcript_text'].str.contains(search_term, case=False, na=False)
        ].copy()
        
        return filtered_df