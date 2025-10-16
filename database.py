# database.py
# Chứa tất cả các hàm để tương tác với Pinecone (thêm, sửa, xóa, tìm kiếm).
# PHIÊN BẢN NÂNG CẤP: Hỗ trợ Lọc Metadata và Tìm kiếm Lai (Simulated).

import uuid
import datetime
import time
import streamlit as st
from config import VECTOR_DIMENSION

def get_embedding(model, text):
    """Tạo vector embedding từ văn bản."""
    return model.encode(text).tolist()

def process_and_upsert(index, model, service_data, note_id=None):
    """Hàm trung tâm để xử lý và lưu dữ liệu, dùng cho cả Thêm và Sửa."""
    
    service_name = service_data.get("service_name", "").strip()
    category = service_data.get("category", "").strip()
    description = service_data.get("description", "").strip()
    price_info = service_data.get("price_info", "").strip()
    tech_reqs = service_data.get("tech_reqs", "").strip()

    if not service_name or not category or category == "--- Thêm danh mục mới ---":
        st.warning("Tên dịch vụ và Danh mục là bắt buộc.")
        return

    combined_text = (
        f"Tên dịch vụ: {service_name}. "
        f"Danh mục: {category}. "
        f"Mô tả và quy cách: {description}. "
        f"Thông tin giá: {price_info}. "
        f"Yêu cầu kỹ thuật: {tech_reqs}."
    )

    with st.spinner("Đang xử lý và lưu lên cloud..."):
        embedding = get_embedding(model, combined_text)
        now = datetime.datetime.now().isoformat()
        
        metadata = {
            "service_name": service_name,
            "category": category,
            "description": description,
            "price_info": price_info,
            "tech_reqs": tech_reqs,
            "combined_text_for_search": combined_text,
            "updated_at": now
        }

        if note_id:
            old_meta_res = index.fetch(ids=[note_id])
            old_metadata = old_meta_res.vectors[note_id].metadata
            metadata["created_at"] = old_metadata.get("created_at", now)
        else:
            note_id = str(uuid.uuid4())
            metadata["created_at"] = now

        try:
            index.upsert(vectors=[{"id": note_id, "values": embedding, "metadata": metadata}])
            st.success("Lưu thông tin thành công!")
            st.session_state.editing_note_id = None
            time.sleep(1)
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Lỗi khi lưu thông tin: {e}")

@st.cache_data(ttl=60)
def get_all_notes(_index):
    """Lấy tất cả dịch vụ từ Pinecone, cache trong 60s."""
    try:
        all_ids_query = _index.query(vector=[0]*VECTOR_DIMENSION, top_k=10000)
        all_ids = [v.id for v in all_ids_query.matches]
        if not all_ids: return []
        
        results = _index.fetch(ids=all_ids)
        notes_data = []
        for note_id, vector_data in results.vectors.items():
            notes_data.append({
                "id": note_id,
                "metadata": vector_data.metadata
            })
        return notes_data
    except Exception as e:
        st.error(f"Lỗi khi lấy danh sách dịch vụ: {e}")
        return []

def delete_note_from_db(index, note_id):
    """Xóa một dịch vụ khỏi Pinecone."""
    try:
        index.delete(ids=[note_id])
        st.success("Đã xóa dịch vụ thành công!")
        time.sleep(1)
        st.cache_data.clear()
    except Exception as e: 
        st.error(f"Lỗi khi xóa dịch vụ: {e}")

def search_notes(index, model, query_text, category_filters=None, n_results=5):
    """
    Tìm kiếm nâng cao trên Pinecone.
    - Hỗ trợ Lọc Metadata theo danh mục.
    - Mô phỏng Tìm kiếm Lai.
    """
    if not query_text.strip(): return []

    # Tạo vector cho câu hỏi
    query_embedding = get_embedding(model, query_text)

    # Xây dựng bộ lọc metadata nếu có
    filter_dict = None
    if category_filters:
        filter_dict = {
            "category": {"$in": category_filters}
        }
    
    try:
        # Gửi query tới Pinecone với bộ lọc
        results = index.query(
            vector=query_embedding, 
            top_k=n_results, 
            include_metadata=True,
            filter=filter_dict
        )
        
        # --- Mô phỏng Tìm kiếm Lai (Simulated Hybrid Search) ---
        # Tăng điểm cho các kết quả có chứa từ khóa trong câu hỏi
        
        query_keywords = set(query_text.lower().split())
        final_results = []
        
        for match in results.matches:
            # Lấy văn bản từ metadata để kiểm tra từ khóa
            text_to_check = match.metadata.get("combined_text_for_search", "").lower()
            
            # Đếm số từ khóa khớp
            keyword_match_count = sum(1 for keyword in query_keywords if keyword in text_to_check)
            
            # Tính toán điểm mới: điểm ngữ nghĩa + điểm thưởng cho từ khóa
            # Trọng số 0.1 cho mỗi từ khóa khớp
            boost_score = keyword_match_count * 0.1
            new_score = match.score + boost_score
            
            final_results.append((new_score, match.metadata))

        # Sắp xếp lại kết quả dựa trên điểm mới (cao nhất lên đầu)
        final_results.sort(key=lambda x: x[0], reverse=True)
        
        # Chỉ trả về phần metadata
        return [metadata for score, metadata in final_results]

    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm: {e}")
        return []

