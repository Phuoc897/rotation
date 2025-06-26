import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

# Cấu hình trang
st.set_page_config(
    page_title="🎨 Chỉnh sửa ảnh 2D & 3D",
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
    
    .tab-content {
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎨 Chỉnh sửa ảnh 2D & 3D</h1>', unsafe_allow_html=True)

# Hàm chỉnh sửa ảnh 2D
def edit_image_2d(image, brightness, contrast, saturation, blur_radius, rotation):
    """Chỉnh sửa ảnh 2D với các tham số được cung cấp"""
    if image is None:
        return None
    
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

# Hàm tạo depth map từ ảnh
def create_depth_map(image, method='sobel'):
    """Tạo depth map từ ảnh để tạo hiệu ứng 3D"""
    # Chuyển đổi sang grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    if method == 'sobel':
        # Sử dụng Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        depth = np.sqrt(sobelx**2 + sobely**2)
    elif method == 'laplacian':
        # Sử dụng Laplacian
        depth = cv2.Laplacian(gray, cv2.CV_64F)
        depth = np.abs(depth)
    elif method == 'brightness':
        # Sử dụng độ sáng làm depth
        depth = gray.astype(np.float64)
    else:
        # Gaussian blur difference
        blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
        blur2 = cv2.GaussianBlur(gray, (15, 15), 0)
        depth = np.abs(blur2.astype(np.float64) - blur1.astype(np.float64))
    
    # Normalize depth map
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return depth

# Hàm tạo anaglyphs 3D (Red-Cyan)
def create_anaglyph(image, depth_intensity=0.1, shift_amount=5):
    """Tạo ảnh anaglyph 3D (Red-Cyan)"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Tạo depth map
    depth = create_depth_map(image, 'brightness')
    
    # Tạo displacement map
    displacement = (depth * shift_amount * depth_intensity).astype(int)
    
    # Tạo left và right eye views
    left_eye = img_array.copy()
    right_eye = img_array.copy()
    
    # Shift pixels based on depth
    for y in range(height):
        for x in range(width):
            shift = displacement[y, x]
            
            # Left eye (red channel) - shift left
            if x - shift >= 0:
                left_eye[y, x] = img_array[y, x - shift]
            
            # Right eye (cyan channels) - shift right  
            if x + shift < width:
                right_eye[y, x] = img_array[y, x + shift]
    
    # Combine channels for anaglyph
    anaglyph = np.zeros_like(img_array)
    anaglyph[:, :, 0] = left_eye[:, :, 0]  # Red from left eye
    anaglyph[:, :, 1] = right_eye[:, :, 1]  # Green from right eye
    anaglyph[:, :, 2] = right_eye[:, :, 2]  # Blue from right eye
    
    return Image.fromarray(anaglyph)

# Hàm tạo stereogram
def create_stereogram(image, pattern_width=100):
    """Tạo stereogram từ ảnh"""
    depth = create_depth_map(image, 'brightness')
    height, width = depth.shape
    
    # Tạo random pattern
    pattern = np.random.randint(0, 256, (height, pattern_width, 3), dtype=np.uint8)
    
    # Tạo stereogram
    stereogram = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Lấy pattern position
            pattern_x = x % pattern_width
            
            # Áp dụng depth displacement
            displacement = int(depth[y, x] * 20)  # Scale depth
            source_x = (pattern_x - displacement) % pattern_width
            
            stereogram[y, x] = pattern[y, source_x]
    
    return Image.fromarray(stereogram)

# Hàm tạo heightmap 3D
def create_3d_heightmap(image, height_scale=50):
    """Tạo 3D heightmap visualization"""
    # Resize image for performance
    img_resized = image.resize((100, 100))
    gray = img_resized.convert('L')
    height_data = np.array(gray)
    
    # Tạo coordinate grids
    x = np.arange(0, height_data.shape[1])
    y = np.arange(0, height_data.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = height_data * height_scale / 255.0
    
    # Tạo màu từ ảnh gốc
    img_array = np.array(img_resized)
    colors = img_array.reshape(-1, 3) / 255.0
    
    return X, Y, Z, colors

# Upload ảnh
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh để chỉnh sửa", 
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
    help="Hỗ trợ: PNG, JPG, JPEG, GIF, BMP, TIFF"
)

if uploaded_file is not None:
    # Đọc ảnh gốc
    original_image = Image.open(uploaded_file)
    
    # Tạo tabs cho 2D và 3D
    tab1, tab2 = st.tabs(["🖼️ Chỉnh sửa 2D", "🌐 Hiệu ứng 3D"])
    
    with tab1:
        # Hiển thị ảnh gốc
        st.subheader("📷 Ảnh gốc")
        st.image(original_image, use_column_width=True)
        
        # Sidebar controls cho 2D
        st.sidebar.markdown("## 🎛️ Tham số 2D")
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        brightness = st.sidebar.slider("🔆 Độ sáng", 0.0, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("🌈 Độ tương phản", 0.0, 2.0, 1.0, 0.1)
        saturation = st.sidebar.slider("🎨 Độ bão hòa", 0.0, 2.0, 1.0, 0.1)
        blur_radius = st.sidebar.slider("🔲 Làm mờ", 0, 10, 0, 1)
        rotation = st.sidebar.slider("🌀 Xoay (độ)", 0, 360, 0, 1)
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Áp dụng chỉnh sửa 2D
        edited_image = edit_image_2d(
            original_image.copy(), 
            brightness, contrast, saturation, blur_radius, rotation
        )
        
        # Hiển thị kết quả 2D
        st.subheader("✨ Ảnh đã chỉnh sửa")
        st.image(edited_image, use_column_width=True)
        
        # Download button cho 2D
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            buf = io.BytesIO()
            edited_image.save(buf, format='PNG')
            byte_data = buf.getvalue()
            
            st.download_button(
                label="💾 Tải xuống ảnh 2D",
                data=byte_data,
                file_name=f"edited_2d_{uploaded_file.name.split('.')[0]}.png",
                mime="image/png",
                use_container_width=True
            )
    
    with tab2:
        st.subheader("🌐 Các hiệu ứng 3D")
        
        # Sidebar controls cho 3D
        st.sidebar.markdown("## 🌐 Tham số 3D")
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        effect_type = st.sidebar.selectbox(
            "🎭 Loại hiệu ứng",
            ["Anaglyph 3D", "Stereogram", "Depth Map", "3D Heightmap"]
        )
        
        if effect_type == "Anaglyph 3D":
            depth_intensity = st.sidebar.slider("🔍 Cường độ độ sâu", 0.0, 1.0, 0.1, 0.05)
            shift_amount = st.sidebar.slider("↔️ Độ dịch chuyển", 1, 20, 5, 1)
        elif effect_type == "3D Heightmap":
            height_scale = st.sidebar.slider("⬆️ Tỷ lệ độ cao", 10, 100, 50, 5)
        
        depth_method = st.sidebar.selectbox(
            "🎯 Phương pháp tạo độ sâu",
            ["sobel", "laplacian", "brightness", "gaussian"]
        )
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Tạo hiệu ứng 3D
        if effect_type == "Anaglyph 3D":
            st.markdown("### 🔴🔵 Anaglyph 3D (Cần kính 3D đỏ-xanh)")
            anaglyph_image = create_anaglyph(original_image, depth_intensity, shift_amount)
            st.image(anaglyph_image, use_column_width=True)
            
            # Download anaglyph
            buf = io.BytesIO()
            anaglyph_image.save(buf, format='PNG')
            st.download_button(
                "💾 Tải xuống Anaglyph 3D",
                buf.getvalue(),
                f"anaglyph_3d_{uploaded_file.name.split('.')[0]}.png",
                "image/png"
            )
            
        elif effect_type == "Stereogram":
            st.markdown("### 👀 Stereogram (Magic Eye)")
            st.info("💡 Mẹo: Nhìn xuyên qua màn hình và thả lỏng mắt để thấy hình 3D")
            stereogram_image = create_stereogram(original_image)
            st.image(stereogram_image, use_column_width=True)
            
            # Download stereogram
            buf = io.BytesIO()
            stereogram_image.save(buf, format='PNG')
            st.download_button(
                "💾 Tải xuống Stereogram",
                buf.getvalue(),
                f"stereogram_{uploaded_file.name.split('.')[0]}.png",
                "image/png"
            )
            
        elif effect_type == "Depth Map":
            st.markdown("### 🗺️ Bản đồ độ sâu")
            depth_map = create_depth_map(original_image, depth_method)
            
            # Hiển thị depth map
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.imshow(original_image)
            ax1.set_title("Ảnh gốc")
            ax1.axis('off')
            
            im = ax2.imshow(depth_map, cmap='viridis')
            ax2.set_title(f"Depth Map ({depth_method})")
            ax2.axis('off')
            plt.colorbar(im, ax=ax2)
            
            st.pyplot(fig)
            
        elif effect_type == "3D Heightmap":
            st.markdown("### 🏔️ 3D Heightmap")
            
            # Tạo 3D heightmap
            X, Y, Z, colors = create_3d_heightmap(original_image, height_scale)
            
            # Sử dụng Plotly để tạo 3D surface
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                showscale=True
            )])
            
            fig.update_layout(
                title="3D Heightmap của ảnh",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y", 
                    zaxis_title="Độ cao",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị thông tin về hiệu ứng
        st.markdown("---")
        st.subheader("ℹ️ Thông tin về hiệu ứng 3D")
        
        if effect_type == "Anaglyph 3D":
            st.markdown("""
            **Anaglyph 3D** tạo hiệu ứng chiều sâu bằng cách:
            - Tạo hai hình ảnh từ góc nhìn khác nhau
            - Kết hợp kênh đỏ (mắt trái) và kênh xanh lam/lục (mắt phải)
            - Cần kính 3D đỏ-xanh để xem hiệu ứng tốt nhất
            """)
        elif effect_type == "Stereogram":
            st.markdown("""
            **Stereogram** (Magic Eye) tạo ảo giác 3D bằng cách:
            - Sử dụng pattern lặp lại với độ lệch nhỏ
            - Mắt phải và trái nhìn thấy pattern khác nhau
            - Não bộ kết hợp tạo cảm giác chiều sâu
            """)
        elif effect_type == "Depth Map":
            st.markdown("""
            **Depth Map** thể hiện độ sâu bằng:
            - Phân tích cường độ sáng/tối của pixel
            - Vùng sáng = gần, vùng tối = xa (hoặc ngược lại)
            - Sử dụng để tạo các hiệu ứng 3D khác
            """)
        elif effect_type == "3D Heightmap":
            st.markdown("""
            **3D Heightmap** biến đổi ảnh thành bề mặt 3D:
            - Cường độ sáng = độ cao
            - Tạo địa hình 3D từ ảnh 2D
            - Có thể xoay và zoom để xem từ mọi góc độ
            """)

    # Thông tin ảnh (sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Thông tin ảnh")
    st.sidebar.write(f"📏 **Kích thước:** {original_image.size[0]} x {original_image.size[1]} pixels")
    st.sidebar.write(f"🎨 **Định dạng:** {original_image.format}")
    st.sidebar.write(f"🔢 **Mode:** {original_image.mode}")
    
    # Nút reset
    if st.sidebar.button("🔄 Đặt lại tất cả"):
        st.rerun()

else:
    st.info("👆 Vui lòng upload ảnh để bắt đầu chỉnh sửa!")
    
    # Hướng dẫn sử dụng
    st.markdown("---")
    st.subheader("📖 Hướng dẫn sử dụng")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🖼️ Chỉnh sửa 2D:
        - **Độ sáng:** Tăng/giảm ánh sáng tổng thể
        - **Tương phản:** Độ rõ nét giữa vùng sáng/tối  
        - **Bão hòa:** Cường độ màu sắc
        - **Làm mờ:** Hiệu ứng blur Gaussian
        - **Xoay:** Quay ảnh theo góc độ
        
        ### 🌐 Hiệu ứng 3D:
        - **Anaglyph:** Cần kính 3D đỏ-xanh
        - **Stereogram:** Nhìn xuyên qua để thấy 3D
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Depth Map Methods:
        - **Sobel:** Edge detection để tìm độ sâu
        - **Laplacian:** Phát hiện biến đổi cường độ
        - **Brightness:** Độ sáng làm độ sâu
        - **Gaussian:** So sánh các mức blur
        
        ### 💡 Mẹo sử dụng:
        - Ảnh có contrast cao cho hiệu ứng 3D tốt hơn
        - Thử các depth method khác nhau
        - Điều chỉnh tham số để có kết quả tối ưu
        """)
    
    # Demo formats
    st.markdown("---")
    st.subheader("🖼️ Định dạng được hỗ trợ")
    st.markdown("PNG • JPG • JPEG • GIF • BMP • TIFF")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>🎨 Được tạo bằng Streamlit với khả năng 3D</div>", 
    unsafe_allow_html=True
)
