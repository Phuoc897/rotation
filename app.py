import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import io
import math

# Cấu hình trang
st.set_page_config(
    page_title="🎨 Givens Rotation Image Editor",
    page_icon="🎨",
    layout="wide"
)

# CSS styling
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
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .matrix-display {
        background: #1e1e1e;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 14px;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 Givens Rotation Image Editor</h1>
    <p>Chỉnh sửa ảnh với phép biến đổi toán học Givens Rotation</p>
</div>
""", unsafe_allow_html=True)

# =================== CORE FUNCTIONS ===================

def givens_rotation_matrix_2d(theta):
    """Tạo ma trận Givens rotation 2D"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

def givens_rotation_matrix_3d(theta, axis='z'):
    """Tạo ma trận Givens rotation 3D cho các trục x, y, z"""
    c = np.cos(theta)
    s = np.sin(theta)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    else:  # axis == 'z'
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

def apply_2d_rotation(image, angle_degrees):
    """Áp dụng xoay 2D cho ảnh"""
    # Chuyển đổi góc từ độ sang radian
    angle_rad = np.radians(angle_degrees)
    
    # Sử dụng PIL rotate với interpolation tốt
    rotated = image.rotate(
        -angle_degrees,  # PIL rotate ngược chiều kim đồng hồ
        resample=Image.BICUBIC,
        expand=True
    )
    
    return rotated

def create_3d_effect(image, rx, ry, rz, depth=30):
    """Tạo hiệu ứng 3D từ ảnh 2D"""
    width, height = image.size
    
    # Tạo depth map từ brightness
    gray_img = image.convert('L')
    depth_array = np.array(gray_img) / 255.0
    
    # Tạo vertices cho mesh 3D đơn giản
    vertices = []
    colors = []
    
    # Giảm resolution để tăng tốc độ
    step = max(1, min(width, height) // 50)
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Tọa độ 3D normalized
            norm_x = (x / width - 0.5) * 2
            norm_y = (y / height - 0.5) * 2
            norm_z = depth_array[y, x] * (depth / 100.0)
            
            vertices.append([norm_x, norm_y, norm_z])
            
            # Lấy màu pixel
            if image.mode == 'RGB':
                colors.append(list(image.getpixel((x, y))))
            else:
                gray_val = image.getpixel((x, y))
                colors.append([gray_val, gray_val, gray_val])
    
    vertices = np.array(vertices)
    
    # Áp dụng các rotation 3D
    if rx != 0:
        R_x = givens_rotation_matrix_3d(np.radians(rx), 'x')
        vertices = np.dot(vertices, R_x.T)
    
    if ry != 0:
        R_y = givens_rotation_matrix_3d(np.radians(ry), 'y')
        vertices = np.dot(vertices, R_y.T)
    
    if rz != 0:
        R_z = givens_rotation_matrix_3d(np.radians(rz), 'z')
        vertices = np.dot(vertices, R_z.T)
    
    return render_3d_projection(vertices, colors, width, height)

def render_3d_projection(vertices, colors, width, height):
    """Render 3D projection thành ảnh 2D"""
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
            
        # Chuyển về tọa độ screen
        screen_x = int((px + 1) * width / 2)
        screen_y = int((1 - py) * height / 2)
        
        projected_points.append((screen_x, screen_y, colors[i]))
    
    # Tạo ảnh kết quả
    result_img = Image.new('RGB', (width, height), (20, 30, 40))
    draw = ImageDraw.Draw(result_img)
    
    # Vẽ các điểm với màu gradient
    point_size = max(1, min(width, height) // 200)
    
    for x, y, color in projected_points:
        if 0 <= x < width and 0 <= y < height:
            # Vẽ điểm với màu
            x1, y1 = x - point_size, y - point_size
            x2, y2 = x + point_size, y + point_size
            draw.ellipse([x1, y1, x2, y2], fill=tuple(color))
    
    return result_img

def apply_image_filters(image, brightness=1.0, contrast=1.0, saturation=1.0):
    """Áp dụng các bộ lọc cơ bản cho ảnh"""
    # Brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    return image

# =================== STREAMLIT INTERFACE ===================

# Sidebar controls
with st.sidebar:
    st.markdown("### 🎛️ Điều khiển")
    
    # Upload ảnh
    uploaded_file = st.file_uploader(
        "📁 Chọn ảnh", 
        type=['png', 'jpg', 'jpeg'],
        help="Hỗ trợ PNG, JPG, JPEG"
    )
    
    st.markdown("---")
    
    # Mode selection
    mode = st.selectbox(
        "🔄 Chế độ xử lý",
        ["2D Rotation", "3D Effect", "Image Filters"],
        help="Chọn loại biến đổi muốn áp dụng"
    )
    
    st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    # Load image
    original_image = Image.open(uploaded_file)
    
    # Resize nếu ảnh quá lớn
    max_size = 800
    if max(original_image.size) > max_size:
        ratio = max_size / max(original_image.size)
        new_size = tuple(int(dim * ratio) for dim in original_image.size)
        original_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
    
    with col1:
        st.markdown("### 📷 Ảnh gốc")
        st.image(original_image, use_column_width=True)
    
    # Controls based on mode
    if mode == "2D Rotation":
        with st.sidebar:
            st.markdown("#### 🔄 2D Rotation Controls")
            angle_2d = st.slider(
                "Góc xoay (độ)", 
                -180, 180, 0, 1,
                help="Xoay ảnh theo góc chỉ định"
            )
            
            if st.button("🔄 Áp dụng 2D Rotation"):
                with st.spinner("Đang xử lý..."):
                    processed_image = apply_2d_rotation(original_image, angle_2d)
                    
                    # Display matrix
                    matrix_2d = givens_rotation_matrix_2d(np.radians(angle_2d))
                    st.markdown("#### 📐 Ma trận Givens 2D")
                    st.markdown(f"""
                    <div class="matrix-display">
                    [{matrix_2d[0,0]: .3f}  {matrix_2d[0,1]: .3f}]<br>
                    [{matrix_2d[1,0]: .3f}  {matrix_2d[1,1]: .3f}]
                    </div>
                    """, unsafe_allow_html=True)
        
        if 'processed_image' in locals():
            with col2:
                st.markdown("### 🎨 Ảnh đã xử lý")
                st.image(processed_image, use_column_width=True)
    
    elif mode == "3D Effect":
        with st.sidebar:
            st.markdown("#### 🎭 3D Effect Controls")
            rx = st.slider("Xoay X (độ)", -90, 90, 0, 5)
            ry = st.slider("Xoay Y (độ)", -90, 90, 0, 5)
            rz = st.slider("Xoay Z (độ)", -90, 90, 0, 5)
            depth = st.slider("Độ sâu", 10, 100, 30, 5)
            
            if st.button("🎭 Tạo hiệu ứng 3D"):
                with st.spinner("Đang tạo hiệu ứng 3D..."):
                    processed_image = create_3d_effect(original_image, rx, ry, rz, depth)
                    
                    # Display rotation matrices
                    if rx != 0:
                        R_x = givens_rotation_matrix_3d(np.radians(rx), 'x')
                        st.markdown("#### 📐 Ma trận Rotation X")
                        st.code(f"R_x =\n{R_x}")
                    
                    if ry != 0:
                        R_y = givens_rotation_matrix_3d(np.radians(ry), 'y')
                        st.markdown("#### 📐 Ma trận Rotation Y")
                        st.code(f"R_y =\n{R_y}")
                    
                    if rz != 0:
                        R_z = givens_rotation_matrix_3d(np.radians(rz), 'z')
                        st.markdown("#### 📐 Ma trận Rotation Z")
                        st.code(f"R_z =\n{R_z}")
        
        if 'processed_image' in locals():
            with col2:
                st.markdown("### 🎭 Hiệu ứng 3D")
                st.image(processed_image, use_column_width=True)
    
    elif mode == "Image Filters":
        with st.sidebar:
            st.markdown("#### 🎨 Filter Controls")
            brightness = st.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)
            contrast = st.slider("Độ tương phản", 0.1, 2.0, 1.0, 0.1)
            saturation = st.slider("Độ bão hòa", 0.0, 2.0, 1.0, 0.1)
            
            if st.button("🎨 Áp dụng bộ lọc"):
                with st.spinner("Đang áp dụng bộ lọc..."):
                    processed_image = apply_image_filters(
                        original_image, brightness, contrast, saturation
                    )
        
        if 'processed_image' in locals():
            with col2:
                st.markdown("### 🎨 Ảnh đã lọc")
                st.image(processed_image, use_column_width=True)
    
    # Download button
    if 'processed_image' in locals():
        buf = io.BytesIO()
        processed_image.save(buf, format='PNG')
        
        st.download_button(
            label="💾 Tải ảnh đã xử lý",
            data=buf.getvalue(),
            file_name=f"processed_{mode.lower().replace(' ', '_')}.png",
            mime="image/png"
        )

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h3>👋 Chào mừng đến với Givens Rotation Image Editor!</h3>
        <p>Ứng dụng này sử dụng phép biến đổi toán học <strong>Givens Rotation</strong> để xử lý ảnh.</p>
        
        <h4>🔥 Tính năng chính:</h4>
        <ul>
            <li><strong>2D Rotation:</strong> Xoay ảnh với ma trận Givens 2D</li>
            <li><strong>3D Effect:</strong> Tạo hiệu ứng 3D từ ảnh 2D</li>
            <li><strong>Image Filters:</strong> Điều chỉnh độ sáng, tương phản, bão hòa</li>
        </ul>
        
        <h4>📚 Về Givens Rotation:</h4>
        <p>Givens rotation là một phép biến đổi tuyến tính sử dụng ma trận orthogonal để xoay vector trong không gian. 
        Ma trận Givens 2D có dạng:</p>
        
        <div class="matrix-display">
        G(θ) = [cos(θ)  -sin(θ)]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[sin(θ)   cos(θ)]
        </div>
        
        <p><strong>👈 Hãy tải lên một ảnh ở sidebar để bắt đầu!</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎨 Givens Rotation Image Editor | Made with Streamlit & NumPy</p>
</div>
""", unsafe_allow_html=True)
