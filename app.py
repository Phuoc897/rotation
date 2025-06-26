import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import io
import math

# Cấu hình trang
st.set_page_config(
    page_title="🔄 Givens Rotation Editor",
    page_icon="🔄",
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
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .matrix-display {
        background: #1e1e1e;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 12px;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔄 Givens Rotation Image Editor</h1>
    <p>Áp dụng phép biến đổi Givens Rotation để tạo hiệu ứng 3D chất lượng cao</p>
</div>
""", unsafe_allow_html=True)

# =================== GIVENS ROTATION FUNCTIONS ===================

def givens_rotation_2d(angle_deg):
    """Tạo ma trận Givens rotation 2D chuẩn"""
    theta = np.radians(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def givens_rotation_3d(angle_deg, plane='xy'):
    """Tạo ma trận Givens rotation 3D cho các mặt phẳng khác nhau"""
    theta = np.radians(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    
    if plane == 'xy':  # Rotation trong mặt phẳng XY (quanh trục Z)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ], dtype=np.float64)
    elif plane == 'xz':  # Rotation trong mặt phẳng XZ (quanh trục Y)
        return np.array([
            [c,  0, s],
            [0,  1, 0],
            [-s, 0, c]
        ], dtype=np.float64)
    elif plane == 'yz':  # Rotation trong mặt phẳng YZ (quanh trục X)
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ], dtype=np.float64)

def apply_givens_2d_rotation(image, angle):
    """Áp dụng Givens rotation 2D trực tiếp lên ảnh"""
    if angle == 0:
        return image
    
    # Sử dụng PIL với interpolation cao
    rotated = image.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0) if image.mode == 'RGBA' else (255, 255, 255)
    )
    return rotated

def create_givens_3d_effect(image, xy_angle, xz_angle, yz_angle, brightness=1.0, quality='high'):
    """
    Tạo hiệu ứng 3D bằng cách áp dụng nhiều Givens rotations
    Sử dụng texture mapping thay vì point cloud để tránh vỡ ảnh
    """
    
    # Điều chỉnh độ sáng
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    width, height = image.size
    
    # Tạo depth map từ luminance
    gray_img = image.convert('L')
    depth_map = np.array(gray_img) / 255.0
    
    # Smooth depth map để tránh artifacts
    from scipy import ndimage
    try:
        depth_map = ndimage.gaussian_filter(depth_map, sigma=1.0)
    except:
        # Fallback nếu không có scipy
        pass
    
    # Tạo mesh grid với density phù hợp
    if quality == 'ultra':
        step = 1
    elif quality == 'high':
        step = 2
    else:
        step = 3
    
    # Tạo 3D mesh coordinates
    mesh_points = []
    mesh_colors = []
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Normalized coordinates [-1, 1]
            norm_x = (x / width - 0.5) * 2.0
            norm_y = (y / height - 0.5) * 2.0
            norm_z = depth_map[y, x] * 0.5  # Depth dựa trên brightness
            
            mesh_points.append([norm_x, norm_y, norm_z])
            
            # Lấy màu pixel
            try:
                if image.mode == 'RGB':
                    color = list(image.getpixel((x, y)))
                else:
                    gray_val = image.getpixel((x, y))
                    color = [gray_val, gray_val, gray_val]
                mesh_colors.append(color)
            except:
                mesh_colors.append([128, 128, 128])
    
    mesh_points = np.array(mesh_points)
    
    # Áp dụng sequential Givens rotations
    if xy_angle != 0:
        R_xy = givens_rotation_3d(xy_angle, 'xy')
        mesh_points = np.dot(mesh_points, R_xy.T)
    
    if xz_angle != 0:
        R_xz = givens_rotation_3d(xz_angle, 'xz')
        mesh_points = np.dot(mesh_points, R_xz.T)
    
    if yz_angle != 0:
        R_yz = givens_rotation_3d(yz_angle, 'yz')
        mesh_points = np.dot(mesh_points, R_yz.T)
    
    # Render với texture mapping
    return render_textured_mesh(mesh_points, mesh_colors, width, height, step)

def render_textured_mesh(vertices, colors, width, height, step):
    """Render 3D mesh với texture mapping để tránh vỡ ảnh"""
    
    # Perspective projection với FOV phù hợp
    fov = 50  # Field of view degrees
    distance = 3.0
    
    projected_points = []
    
    for i, vertex in enumerate(vertices):
        x, y, z = vertex
        z_cam = z + distance
        
        if z_cam > 0.1:
            # Perspective projection
            px = x / z_cam * math.tan(math.radians(fov/2))
            py = y / z_cam * math.tan(math.radians(fov/2))
        else:
            px, py = x * 0.5, y * 0.5
        
        # Convert to screen coordinates với margin
        margin = 0.1
        screen_x = int((px + 1) * width * (0.5 - margin) + width * margin)
        screen_y = int((1 - py) * height * (0.5 - margin) + height * margin)
        
        projected_points.append((screen_x, screen_y, colors[i], z_cam))
    
    # Sort by depth (back to front)
    projected_points.sort(key=lambda p: p[3], reverse=True)
    
    # Create result image với gradient background
    result_img = create_gradient_background(width, height)
    
    # Render points với adaptive sizing
    for screen_x, screen_y, color, depth in projected_points:
        if 0 <= screen_x < width and 0 <= screen_y < height:
            # Adaptive point size based on depth
            base_size = max(step, 2)
            depth_factor = max(0.3, min(1.5, distance / (depth + 0.1)))
            point_size = int(base_size * depth_factor)
            
            # Render với anti-aliasing
            render_smooth_point(result_img, screen_x, screen_y, point_size, color)
    
    return result_img

def create_gradient_background(width, height):
    """Tạo background gradient đẹp"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Radial gradient từ center
    center_x, center_y = width // 2, height // 2
    max_radius = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(height):
        for x in range(width):
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            factor = 1.0 - (distance / max_radius) * 0.5
            
            r = int(20 * factor)
            g = int(30 * factor)
            b = int(50 * factor)
            
            draw.point([x, y], fill=(r, g, b))
    
    return img

def render_smooth_point(img, x, y, size, color):
    """Render point với smooth anti-aliasing"""
    draw = ImageDraw.Draw(img)
    
    if size <= 1:
        draw.point([x, y], fill=tuple(color))
        return
    
    # Multi-layer rendering cho smooth effect
    layers = max(2, size // 2)
    
    for i in range(layers, 0, -1):
        layer_size = int(size * i / layers)
        alpha = (i / layers) * 0.8 + 0.2
        
        # Blend với background
        try:
            bg_color = img.getpixel((x, y))
            blended_color = [
                int(bg_color[j] * (1 - alpha) + color[j] * alpha)
                for j in range(3)
            ]
        except:
            blended_color = color
        
        draw.ellipse([
            x - layer_size, y - layer_size,
            x + layer_size, y + layer_size
        ], fill=tuple(blended_color))

# =================== STREAMLIT UI ===================

# Sidebar cho thông tin
with st.sidebar:
    st.markdown("### 📚 Givens Rotation")
    st.markdown("""
    **Givens rotation** là phép biến đổi orthogonal 
    dùng để xoay vector trong không gian 2D/3D.
    
    **Ma trận 2D:**
    ```
    [cos θ  -sin θ]
    [sin θ   cos θ]
    ```
    
    **Ưu điểm:**
    - Bảo toàn độ dài vector
    - Stable numerically  
    - Composition tốt
    """)

# Upload ảnh
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh để áp dụng Givens Rotation", 
    type=['png', 'jpg', 'jpeg'],
    help="Hỗ trợ PNG, JPG, JPEG"
)

if uploaded_file is not None:
    # Load và resize ảnh
    image = Image.open(uploaded_file)
    
    # Auto resize để tối ưu hiệu suất
    max_size = 600
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("🎛️ Givens Controls")
        
        # Quality setting
        quality = st.selectbox(
            "🎯 Chất lượng",
            ['normal', 'high', 'ultra'],
            index=1,
            help="Ultra: Chất lượng tốt nhất nhưng chậm hơn"
        )
        
        # Brightness
        brightness = st.slider(
            "💡 Độ sáng", 0.3, 2.0, 1.0, 0.1,
            help="Ảnh hưởng đến depth map 3D"
        )
        
        st.markdown("### 🔄 Givens Rotations")
        
        # 2D Rotation
        rotation_2d = st.slider(
            "🔄 2D Rotation", -180, 180, 0, 5,
            help="Givens rotation 2D cơ bản"
        )
        
        # 3D Rotations
        st.markdown("**3D Rotations:**")
        xy_rotation = st.slider(
            "🔄 XY Plane", -90, 90, 0, 5,
            help="Rotation trong mặt phẳng XY"
        )
        
        xz_rotation = st.slider(
            "🔄 XZ Plane", -90, 90, 0, 5,
            help="Rotation trong mặt phẳng XZ"
        )
        
        yz_rotation = st.slider(
            "🔄 YZ Plane", -90, 90, 0, 5,
            help="Rotation trong mặt phẳng YZ"
        )
        
        # Matrix display
        if xy_rotation != 0 or xz_rotation != 0 or yz_rotation != 0:
            st.markdown("### 📊 Ma trận hiện tại")
            
            if xy_rotation != 0:
                R_xy = givens_rotation_3d(xy_rotation, 'xy')
                st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                st.code(f"XY Rotation Matrix:\n{np.array2string(R_xy, precision=3)}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset button
        if st.button("🔄 Reset All"):
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.subheader("📷 Ảnh gốc")
            st.image(image, use_column_width=True)
        
        with col2_2:
            st.subheader("✨ Givens Transform")
            
            # Kiểm tra có transformation nào không
            has_2d = rotation_2d != 0
            has_3d = xy_rotation != 0 or xz_rotation != 0 or yz_rotation != 0
            has_brightness = brightness != 1.0
            
            if has_2d or has_3d or has_brightness:
                with st.spinner("🔄 Applying Givens Rotations..."):
                    
                    # Áp dụng 2D rotation trước (nếu có)
                    if has_2d:
                        result_image = apply_givens_2d_rotation(image, rotation_2d)
                    else:
                        result_image = image
                    
                    # Áp dụng 3D effect (nếu có)
                    if has_3d or has_brightness:
                        result_image = create_givens_3d_effect(
                            result_image, xy_rotation, xz_rotation, yz_rotation, 
                            brightness, quality
                        )
                    
                    st.image(result_image, use_column_width=True)
                    
                    # Download button
                    img_buffer = io.BytesIO()
                    result_image.save(img_buffer, format='PNG')
                    
                    st.download_button(
                        label="💾 Download Result",
                        data=img_buffer.getvalue(),
                        file_name="givens_rotation_result.png",
                        mime="image/png"
                    )
            else:
                st.image(image, use_column_width=True)
                st.info("👆 Điều chỉnh các slider để xem Givens transformations")
        
        # Thông tin chi tiết
        st.markdown("---")
        with st.expander("📖 Giải thích Givens Rotation"):
            st.markdown("""
            **Givens Rotation** là một phương pháp toán học để thực hiện phép xoay:
            
            🔄 **2D Rotation**: Xoay ảnh trong mặt phẳng 2D cơ bản
            
            🔄 **3D Rotations**:
            - **XY Plane**: Rotation quanh trục Z (xoay trong mặt phẳng XY)
            - **XZ Plane**: Rotation quanh trục Y (xoay trong mặt phẳng XZ)  
            - **YZ Plane**: Rotation quanh trục X (xoay trong mặt phẳng YZ)
            
            💡 **Độ sáng**: Ảnh hưởng đến depth map để tạo hiệu ứng 3D
            
            ⚡ **Chất lượng**: Ultra mode sẽ giữ được nhiều detail hơn nhưng xử lý lâu hơn
            """)

else:
    st.info("👆 Upload một ảnh để bắt đầu áp dụng Givens Rotations!")
    
    # Info về Givens Rotation
    st.markdown("---")
    st.markdown("""
    ### 🎓 Về Givens Rotation
    
    **Givens rotation** là một kỹ thuật toán học quan trọng trong:
    - Linear algebra và matrix decomposition
    - Computer graphics và 3D transformations  
    - Signal processing và image processing
    - Numerical methods và scientific computing
    
    App này demonstrate việc áp dụng Givens rotations để tạo hiệu ứng 3D cho ảnh!
    """)
