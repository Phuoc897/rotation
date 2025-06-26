import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io

# Cấu hình trang
st.set_page_config(
    page_title="🎨 Chỉnh sửa ảnh đơn giản",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4A90E2;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎨 Chỉnh sửa ảnh đơn giản</h1>', unsafe_allow_html=True)

# Hàm chỉnh sửa ảnh
def edit_image(image, brightness, contrast, saturation, blur_radius, rotation):
    """
    Chỉnh sửa ảnh với các tham số được cung cấp
    """
    if image is None:
        return None
    
    # Chuyển đổi sang RGB nếu cần
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Áp dụng độ sáng
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Áp dụng độ tương phản
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Áp dụng độ bão hòa
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation)
    
    # Áp dụng làm mờ
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Áp dụng xoay
    if rotation != 0:
        image = image.rotate(rotation, expand=True, fillcolor='white')
    
    return image

# Sidebar - Tham số chỉnh sửa
st.sidebar.markdown("## 🎛️ Tham số chỉnh sửa")

# Upload ảnh
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh để chỉnh sửa", 
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
    help="Hỗ trợ: PNG, JPG, JPEG, GIF, BMP, TIFF"
)

if uploaded_file is not None:
    # Đọc ảnh gốc
    original_image = Image.open(uploaded_file)
    
    # Hiển thị ảnh gốc
    st.subheader("📷 Ảnh gốc")
    st.image(original_image, use_column_width=True)
    
    # Sidebar controls
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    brightness = st.sidebar.slider(
        "🔆 Độ sáng", 
        min_value=0.0, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="0.0 = Tối nhất, 1.0 = Bình thường, 2.0 = Sáng nhất"
    )
    
    contrast = st.sidebar.slider(
        "🌈 Độ tương phản", 
        min_value=0.0, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="0.0 = Mờ nhất, 1.0 = Bình thường, 2.0 = Rõ nét nhất"
    )
    
    saturation = st.sidebar.slider(
        "🎨 Độ bão hòa", 
        min_value=0.0, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="0.0 = Đen trắng, 1.0 = Bình thường, 2.0 = Màu sắc đậm"
    )
    
    blur_radius = st.sidebar.slider(
        "🔲 Làm mờ", 
        min_value=0, 
        max_value=10, 
        value=0, 
        step=1,
        help="0 = Không mờ, 10 = Rất mờ"
    )
    
    rotation = st.sidebar.slider(
        "🌀 Xoay (độ)", 
        min_value=0, 
        max_value=360, 
        value=0, 
        step=1,
        help="0° đến 360°"
    )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Nút reset
    if st.sidebar.button("🔄 Đặt lại tất cả"):
        st.rerun()
    
    # Thông tin ảnh
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Thông tin ảnh")
    st.sidebar.write(f"📏 **Kích thước:** {original_image.size[0]} x {original_image.size[1]} pixels")
    st.sidebar.write(f"🎨 **Định dạng:** {original_image.format}")
    st.sidebar.write(f"🔢 **Mode:** {original_image.mode}")
    
    # Áp dụng chỉnh sửa
    edited_image = edit_image(
        original_image.copy(), 
        brightness, 
        contrast, 
        saturation, 
        blur_radius, 
        rotation
    )
    
    # Hiển thị kết quả
    st.subheader("✨ Ảnh đã chỉnh sửa")
    st.image(edited_image, use_column_width=True)
    
    # Buttons dưới ảnh
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        # Tạo link download
        buf = io.BytesIO()
        edited_image.save(buf, format='PNG')
        byte_data = buf.getvalue()
        
        st.download_button(
            label="💾 Tải xuống ảnh",
            data=byte_data,
            file_name=f"edited_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png",
            use_container_width=True
        )
    
    # So sánh trước/sau
    if st.checkbox("👁️ So sánh trước/sau"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trước")
            st.image(original_image, use_column_width=True)
        with col2:
            st.subheader("Sau")
            st.image(edited_image, use_column_width=True)

else:
    st.info("👆 Vui lòng upload ảnh để bắt đầu chỉnh sửa!")
    
    # Hướng dẫn sử dụng
    st.markdown("---")
    st.subheader("📖 Hướng dẫn sử dụng")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Bắt đầu:
        1. **Upload ảnh:** Click "Browse files" và chọn ảnh
        2. **Chờ tải lên:** Ảnh sẽ hiển thị sau khi upload thành công
        
        ### 🎛️ Chỉnh sửa:
        3. **Sử dụng sidebar:** Các tham số điều chỉnh ở bên trái
        4. **Xem real-time:** Ảnh thay đổi ngay khi bạn kéo slider
        5. **So sánh:** Tick vào "So sánh trước/sau" để đối chiếu
        """)
    
    with col2:
        st.markdown("""
        ### 📥 Lưu kết quả:
        6. **Tải xuống:** Click "Tải xuống ảnh" để lưu file
        7. **Đặt lại:** Click "Đặt lại tất cả" để về ban đầu
        
        ### 🎨 Tham số:
        - **Độ sáng:** Tăng/giảm ánh sáng
        - **Tương phản:** Độ rõ nét của ảnh  
        - **Bão hòa:** Cường độ màu sắc
        - **Làm mờ:** Hiệu ứng blur
        - **Xoay:** Quay ảnh theo góc độ
        """)
    
    # Demo
    st.markdown("---")
    st.subheader("🖼️ Định dạng được hỗ trợ")
    st.markdown("PNG • JPG • JPEG • GIF • BMP • TIFF")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🎨 Được tạo bằng Streamlit</div>", 
    unsafe_allow_html=True
)
