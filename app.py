# app.py
# File chính, chỉ chứa logic để xây dựng giao diện người dùng với Streamlit.

import streamlit as st
import os
import time
import datetime
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Import các hàm từ các file đã được tách ra
import database as db
from config import (
    MODEL_NAME, 
    VECTOR_DIMENSION, 
    PINECONE_INDEX_NAME, 
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION
)

# --- 1. CẤU HÌNH BAN ĐẦU ---

load_dotenv()
st.set_page_config(page_title="Quản Lý Dịch Vụ In Ấn", page_icon="🖨️", layout="wide")

st.title("🖨️ Hệ Thống Quản Lý Dịch Vụ In Ấn")
st.caption("Nơi cập nhật kiến thức cho chatbot bán hàng của bạn. (Kiến trúc đã tái cấu trúc)")

# Khởi tạo session_state để quản lý trạng thái
if 'editing_note_id' not in st.session_state:
    st.session_state.editing_note_id = None
if 'category_filter' not in st.session_state:
    st.session_state.category_filter = "Tất cả"

# --- 2. TẢI MÔ HÌNH VÀ KẾT NỐI PINECONE (Sử dụng cache của Streamlit) ---

@st.cache_resource
def load_model():
    """Tải mô hình AI vào bộ nhớ (chỉ một lần)."""
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def init_pinecone():
    """Khởi tạo kết nối tới Pinecone (chỉ một lần)."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("Vui lòng cung cấp PINECONE_API_KEY trong file .env")
        st.stop()
    
    pc = Pinecone(api_key=api_key)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        st.warning(f"Index '{PINECONE_INDEX_NAME}' không tồn tại. Đang tạo mới...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION) 
        )
        st.success("Tạo index mới thành công! Vui lòng chờ giây lát để index khởi tạo.")
        time.sleep(10)

    return pc.Index(PINECONE_INDEX_NAME)

try:
    with st.spinner("Đang tải tài nguyên và kết nối tới cloud..."):
        model = load_model()
        index = init_pinecone()
except Exception as e:
    st.error(f"Đã xảy ra lỗi khởi tạo: {e}")
    st.stop()

# --- 3. GIAO DIỆN DASHBOARD ---

all_notes_data = db.get_all_notes(index)

# Chuẩn bị danh sách danh mục cho các form và bộ lọc
all_categories_for_filter = ["Tất cả"] + sorted(list(set(note['metadata'].get('category', 'Chưa phân loại') for note in all_notes_data)))
unique_categories_for_forms = sorted(list(set(note['metadata'].get('category') for note in all_notes_data if note['metadata'].get('category'))))
category_options_for_forms = unique_categories_for_forms + ["--- Thêm danh mục mới ---"]

# --- THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR) ---
with st.sidebar:
    st.header("Thêm Dịch Vụ Mới")
    with st.form("new_service_form"):
        service_name = st.text_input("Tên Dịch Vụ (*)")
        selected_category = st.selectbox("Danh Mục (*)", options=category_options_for_forms, key="new_cat_select")
        
        new_category = ""
        if selected_category == "--- Thêm danh mục mới ---":
            new_category = st.text_input("Nhập tên danh mục mới:")
            
        description = st.text_area("Mô Tả / Quy Cách")
        price_info = st.text_area("Bảng Giá")
        tech_reqs = st.text_area("Yêu Cầu Kỹ Thuật")

        submitted = st.form_submit_button("Lưu Dịch Vụ Mới")
        if submitted:
            final_category = new_category.strip() if new_category else selected_category
            new_service_data = {
                "service_name": service_name,
                "category": final_category,
                "description": description,
                "price_info": price_info,
                "tech_reqs": tech_reqs
            }
            db.process_and_upsert(index, model, new_service_data)
            st.rerun()

    st.divider()
    st.header("Bộ lọc")
    st.session_state.category_filter = st.selectbox("Lọc theo danh mục:", options=all_categories_for_filter)

# --- KHU VỰC HIỂN THỊ CHÍNH ---
st.header("🔍 Tìm kiếm thông minh (Dành cho Chatbot)")
search_query = st.text_input("Nhập câu hỏi của khách hàng để thử nghiệm:", placeholder="VD: In 500 card visit giá bao nhiêu?")
if search_query:
    with st.spinner("Đang tìm kiếm trên cloud..."):
        search_results_meta = db.search_notes(index, model, search_query)
    if search_results_meta:
        st.subheader("Kết quả phù hợp nhất:")
        for meta in search_results_meta:
            with st.expander(f"**{meta.get('service_name')}** - Danh mục: {meta.get('category')}"):
                st.markdown(f"**Mô tả:** {meta.get('description', 'N/A')}")
                st.markdown(f"**Giá:** {meta.get('price_info', 'N/A')}")
                st.markdown(f"**Yêu cầu:** {meta.get('tech_reqs', 'N/A')}")
    else:
        st.write("Không tìm thấy thông tin phù hợp.")

st.divider()

st.header(f"Danh sách Dịch vụ ({st.session_state.category_filter})")

if all_notes_data:
    if st.session_state.category_filter == "Tất cả":
        filtered_notes = all_notes_data
    else:
        filtered_notes = [
            note for note in all_notes_data 
            if note['metadata'].get('category') == st.session_state.category_filter
        ]
    
    sorted_notes = sorted(filtered_notes, key=lambda x: x['metadata'].get('updated_at', '1970-01-01'), reverse=True)
    
    if not sorted_notes:
        st.info(f"Không có dịch vụ nào trong danh mục '{st.session_state.category_filter}'.")
    
    for note in sorted_notes:
        note_id, meta = note['id'], note['metadata']
        
        with st.container(border=True):
            if st.session_state.editing_note_id == note_id:
                st.subheader(f"Chỉnh sửa: {meta.get('service_name')}")
                with st.form(f"edit_form_{note_id}"):
                    edited_service_name = st.text_input("Tên Dịch Vụ (*)", value=meta.get('service_name'))
                    
                    try:
                        current_category_index = category_options_for_forms.index(meta.get('category'))
                    except ValueError:
                        current_category_index = 0
                    
                    selected_category_edit = st.selectbox("Danh Mục (*)", options=category_options_for_forms, index=current_category_index, key=f"cat_edit_{note_id}")
                    
                    new_category_edit = ""
                    if selected_category_edit == "--- Thêm danh mục mới ---":
                        new_category_edit = st.text_input("Nhập tên danh mục mới:", key=f"new_cat_edit_{note_id}")

                    edited_description = st.text_area("Mô Tả / Quy Cách", value=meta.get('description'))
                    edited_price_info = st.text_area("Bảng Giá", value=meta.get('price_info'))
                    edited_tech_reqs = st.text_area("Yêu Cầu Kỹ Thuật", value=meta.get('tech_reqs'))
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.form_submit_button("✅ Lưu thay đổi", type="primary"):
                            final_category_edit = new_category_edit.strip() if new_category_edit else selected_category_edit
                            edited_data = {
                                "service_name": edited_service_name,
                                "category": final_category_edit,
                                "description": edited_description,
                                "price_info": edited_price_info,
                                "tech_reqs": edited_tech_reqs
                            }
                            db.process_and_upsert(index, model, edited_data, note_id)
                            st.rerun()
                    with c2:
                        if st.form_submit_button("❌ Hủy"):
                            st.session_state.editing_note_id = None
                            st.rerun()

            else:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.subheader(meta.get('service_name', 'Chưa có tên'))
                    st.markdown(f"**Danh mục:** `{meta.get('category', 'N/A')}`")
                    if meta.get('description'):
                        st.markdown(f"**Mô tả:**\n{meta.get('description')}")
                    if meta.get('price_info'):
                        st.markdown(f"**Giá:**\n{meta.get('price_info')}")
                    if meta.get('tech_reqs'):
                        st.markdown(f"**Yêu cầu:**\n{meta.get('tech_reqs')}")
                    
                    if meta.get('updated_at'):
                        updated_time = datetime.datetime.fromisoformat(meta.get('updated_at')).strftime('%H:%M, %d-%m-%Y')
                        st.caption(f"Cập nhật lần cuối: {updated_time}")
                
                with col2:
                    if st.button("Sửa", key=f"edit_btn_{note_id}"):
                        st.session_state.editing_note_id = note_id
                        st.rerun()
                    if st.button("Xóa", key=f"del_btn_{note_id}"):
                        db.delete_note_from_db(index, note_id)
                        st.rerun()

else:
    st.write("Chưa có dịch vụ nào được thêm vào. Vui lòng sử dụng form bên trái để bắt đầu.")

