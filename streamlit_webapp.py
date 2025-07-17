import streamlit as st
import requests
from PIL import Image
import io

st.title("Chuyển đổi ảnh visa - API demo")

uploaded_file = st.file_uploader("Chọn ảnh để xử lý", type=["jpg", "jpeg", "png"])

# Thêm selectbox cho phép chọn loại ảnh visa
visa_types = {
    "2x3 cm (20x30 mm)": "2x3",
    "3x4 cm (30x40 mm)": "3x4"
}
selected_type = st.selectbox("Chọn loại ảnh visa", list(visa_types.keys()))

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)
    if st.button("Xử lý ảnh"):
        with st.spinner("Đang xử lý..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            # Gửi thêm loại ảnh visa lên API
            data = {"type": visa_types[selected_type]}
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
                    file_name=f"visa_photo_{visa_types[selected_type]}.jpg",
                    mime="image/jpeg"
                )
            else:
                st.error("Lỗi xử lý ảnh!")