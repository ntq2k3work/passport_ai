import streamlit as st
import requests
from PIL import Image
import io
import json

st.title("Chuyển đổi ảnh visa - API demo")

# Đọc file data.json
@st.cache_data
def load_photo_types():
    try:
        with open('data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Lỗi đọc file data.json: {e}")
        return []

photo_types = load_photo_types()

uploaded_file = st.file_uploader("Chọn ảnh để xử lý", type=["jpg", "jpeg", "png"])

# Tạo selectbox từ data.json
if photo_types:
    # Tạo danh sách options cho selectbox
    options = []
    for photo_type in photo_types:
        display_name = f"{photo_type['type']} ({photo_type['size_mm']} mm)"
        options.append((display_name, photo_type))
    
    selected_display = st.selectbox(
        "Chọn loại ảnh visa", 
        [opt[0] for opt in options],
        format_func=lambda x: x
    )
    
    # Lấy data tương ứng với lựa chọn
    selected_data = None
    for display_name, data in options:
        if display_name == selected_display:
            selected_data = data
            break

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)
    
    if selected_data:
        # Hiển thị thông tin chi tiết về loại ảnh đã chọn
        st.info(f"""
        **Thông tin ảnh đã chọn:**
        - Loại: {selected_data['type']}
        - Kích thước: {selected_data['size_mm']} mm
        - Độ phân giải: {selected_data['size_px']} px
        - DPI: {selected_data['dpi']}
        - Nền: {selected_data['background']}
        """)
    
    if st.button("Xử lý ảnh") and selected_data:
        with st.spinner("Đang xử lý..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            
            # Gửi đầy đủ data theo format yêu cầu
            data = {
                "size_mm": selected_data['size_mm'],
                "size_px": selected_data['size_px'],
                "dpi": selected_data['dpi'],
                "background": selected_data['background'],
                "top_margin_mm": selected_data['top_margin_mm'],
                "bottom_margin_mm": selected_data['bottom_margin_mm'],
                "left_margin_mm": selected_data['left_margin_mm'],
                "right_margin_mm": selected_data['right_margin_mm']
            }
            
            response = requests.post("http://localhost:8000/convert", files=files, data=data)
            if response.status_code == 200:
                # Xử lý ảnh trong bộ nhớ mà không lưu file
                img_bytes = response.content
                img = Image.open(io.BytesIO(img_bytes))
                
                # Hiển thị ảnh từ bytes trực tiếp
                st.image(img_bytes, caption="Ảnh sau xử lý", use_column_width=True)
                
                # Nút tải xuống sử dụng bytes gốc từ API
                st.download_button(
                    label="📥 Tải xuống ảnh đã xử lý",
                    data=img_bytes,
                    file_name=f"visa_photo_{selected_data['type']}.jpg",
                    mime="image/jpeg"
                )
            else:
                st.error(f"Lỗi xử lý ảnh! Status code: {response.status_code}")
                try:
                    error_detail = response.json()
                    st.error(f"Chi tiết lỗi: {error_detail}")
                except:
                    st.error(f"Response: {response.text}")
else:
    if not photo_types:
        st.error("Không thể tải dữ liệu loại ảnh từ file data.json")