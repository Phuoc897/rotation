import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import io
import math

# Cấu hình trang
st.set_page_config(
    page_title="🎨 3D Image Editor",
    page_icon="🎨",
    layout="wide"
)

# CSS styling đơn giản
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .control-panel {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 3D Image Editor</h1>
    <p>Điều chỉnh độ sáng và xoay ảnh 3D với chất lượng cao</p>
</div>
""", unsafe_allow_html=True)

# =================== CORE FUNCTIONS ===================

def adjust_brightness(image, brightness_factor):
    """Điều chỉnh độ sáng của ảnh"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

def create_3d_rotation_effect(image, rx, ry, rz, brightness=1.0):
    """Tạo hiệu ứng xoay 3D với độ sáng điều chỉnh được"""
    
    # Điều chỉnh độ sáng trước
    if brightness != 1.0:
        image = adjust_brightness(image, brightness)
    
    width, height = image.size
    
    # Tạo depth map từ brightness
    gray_img = image.convert('L')
    depth_array = np.array(gray_img) / 255.0
    
    # Tạo vertices với mật độ cao
    vertices = []
    colors = []
    
    step = 2  # Bước nhảy nhỏ để giữ nhiều pixel
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Tọa độ 3D
            norm_x = (x / width - 0.5) * 2.0
            norm_y = (y / height - 0.5) * 2.0  
            norm_z = depth_array[y, x] * 0.3  # Depth từ brightness
            
            vertices.append([norm_x, norm_y, norm_z])
            
            # Lấy màu pixel
            if image.mode == 'RGB':
                color = list(image.getpixel((x, y)))
            else:
                gray_val = image.getpixel((x, y))
                color = [gray_val, gray_val, gray_val]
            colors.append(color)
    
    vertices = np.array(vertices)
    
    # Áp dụng rotation 3D
    if rx != 0:
        R_x = rotation_matrix_x(np.radians(rx))
        vertices = np.dot(vertices, R_x.T)
    
    if ry != 0:
        R_y = rotation_matrix_y(np.radians(ry))
        vertices = np.dot(vertices, R_y.T)
    
    if rz != 0:
        R_z = rotation_matrix_z(np.radians(rz))
        vertices = np.dot(vertices, R_z.T)
    
    # Render kết quả
    return render_3d_projection(vertices, colors, width, height)

def rotation_matrix_x(theta):
    """Ma trận xoay quanh trục X"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_y(theta):
    """Ma trận xoay quanh trục Y"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotation_matrix_z(theta):
    """Ma trận xoay quanh trục Z"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def render_3d_projection(vertices, colors, width, height):
    """Render projection 3D với chất lượng cao"""
    
    # Perspective projection
    distance = 3.0
    projected_points = []
    
    for i, vertex in enumerate(vertices):
        x, y, z = vertex
        z_cam = z + distance
        
        if z_cam > 0.1:
            px = x / z_cam
            py = y / z_cam
        else:
            px, py = x, y
        
        # Chuyển về tọa độ màn hình
        screen_x = int((px + 1) * width * 0.4 + width * 0.1)
        screen_y = int((1 - py) * height * 0.4 + height * 0.1)
        
        projected_points.append((screen_x, screen_y, colors[i], z_cam))
    
    # Sắp xếp theo depth
    projected_points.sort(key=lambda p: p[3], reverse=True)
    
    # Tạo ảnh kết quả
    result_img = Image.new('RGB', (width, height), (10, 15, 25))
    draw = ImageDraw.Draw(result_img)
    
    # Vẽ các points
    for screen_x, screen_y, color, depth in projected_points:
        if 0 <= screen_x < width and 0 <= screen_y < height:
            # Kích thước point dựa trên depth
            point_size = max(1, int(3 / (depth + 0.5)))
            
            # Vẽ point với anti-aliasing đơn giản
            for i in range(point_size, 0, -1):
                alpha = i / point_size * 0.8
                
                # Blend màu với background
                try:
                    bg_color = result_img.getpixel((screen_x, screen_y))
                    final_color = [
                        int(bg_color[j] * (1 - alpha) + color[j] * alpha)
                        for j in range(3)
                    ]
                except:
                    final_color = color
                
                draw.ellipse([
                    screen_x - i, screen_y - i,
                    screen_x + i, screen_y + i
                ], fill=tuple(final_color))
    
    return result_img

# =================== STREAMLIT UI ===================

# Upload ảnh
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh để chỉnh sửa", 
    type=['png', 'jpg', 'jpeg'],
    help="Hỗ trợ các định dạng: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    # Load ảnh
    image = Image.open(uploaded_file)
    
    # Resize nếu ảnh quá lớn
    max_size = 800
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("🎛️ Điều khiển")
        
        # Brightness control
        brightness = st.slider(
            "💡 Độ sáng", 
            min_value=0.1, 
            max_value=2.0, 
            value=1.0, 
            step=0.1,
            help="Điều chỉnh độ sáng của ảnh"
        )
        
        st.subheader("🔄 Xoay 3D")
        
        # Rotation controls
        rx = st.slider("🔄 Xoay X", -90, 90, 0, 5, help="Xoay quanh trục X")
        ry = st.slider("🔄 Xoay Y", -90, 90, 0, 5, help="Xoay quanh trục Y") 
        rz = st.slider("🔄 Xoay Z", -90, 90, 0, 5, help="Xoay quanh trục Z")
        
        # Reset button
        if st.button("🔄 Reset tất cả"):
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Hiển thị ảnh gốc và kết quả
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.subheader("📷 Ảnh gốc")
            st.image(image, use_column_width=True)
        
        with col2_2:
            st.subheader("✨ Kết quả 3D")
            
            # Xử lý ảnh
            if rx != 0 or ry != 0 or rz != 0 or brightness != 1.0:
                with st.spinner("🎨 Đang xử lý..."):
                    result_image = create_3d_rotation_effect(image, rx, ry, rz, brightness)
                    st.image(result_image, use_column_width=True)
                    
                    # Download button
                    img_buffer = io.BytesIO()
                    result_image.save(img_buffer, format='PNG')
                    
                    st.download_button(
                        label="💾 Tải xuống",
                        data=img_buffer.getvalue(),
                        file_name="3d_edited_image.png",
                        mime="image/png"
                    )
            else:
                st.image(image, use_column_width=True)
                st.info("🎯 Điều chỉnh các thanh trượt để xem hiệu ứng 3D")
        
        # Thông tin
        st.markdown("---")
        st.markdown("""
        **🎯 Hướng dẫn sử dụng:**
        - **Độ sáng**: Tăng/giảm độ sáng ảnh (ảnh hưởng đến depth 3D)
        - **Xoay X**: Xoay ảnh lên/xuống  
        - **Xoay Y**: Xoay ảnh trái/phải
        - **Xoay Z**: Xoay ảnh theo chiều kim đồng hồ
        
        *💡 Tip: Độ sáng cao sẽ tạo depth 3D rõ nét hơn*
        """)

else:
    st.info("👆 Vui lòng upload một ảnh để bắt đầu chỉnh sửa 3D!")
    
    # Demo image
    st.markdown("---")
    st.subheader("🖼️ Ảnh demo")
    
    # Tạo ảnh demo đơn giản
    demo_img = Image.new('RGB', (400, 300), (100, 150, 200))
    draw = ImageDraw.Draw(demo_img)
    
    # Vẽ gradient circle để demo
    center_x, center_y = 200, 150
    for r in range(100, 0, -5):
        intensity = int(255 * (100 - r) / 100)
        draw.ellipse([
            center_x - r, center_y - r,
            center_x + r, center_y + r
        ], fill=(intensity, intensity + 50, intensity + 100))
    
    st.image(demo_img, caption="Ảnh demo - Upload ảnh của bạn để trải nghiệm!", use_column_width=True)
