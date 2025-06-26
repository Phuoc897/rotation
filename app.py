import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import io
import math

# Cấu hình trang
st.set_page_config(
    page_title="🌐 Givens Rotation 3D Editor",
    page_icon="🌐",
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
        font-size: 11px;
        line-height: 1.3;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌐 Givens Rotation 3D Editor</h1>
    <p>Áp dụng phép biến đổi Givens Rotation 3D để tạo hiệu ứng không gian ba chiều</p>
</div>
""", unsafe_allow_html=True)

# =================== GIVENS ROTATION 3D FUNCTIONS ===================

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

def simple_gaussian_blur(arr, sigma=1.0):
    """Simple Gaussian blur implementation without scipy"""
    if sigma <= 0:
        return arr
    
    # Create simple Gaussian kernel
    kernel_size = int(2 * sigma * 3) + 1
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Fill kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            distance_sq = x*x + y*y
            kernel[i, j] = math.exp(-distance_sq / (2 * sigma * sigma))
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply convolution manually (simple version)
    height, width = arr.shape
    result = np.zeros_like(arr)
    pad = kernel_size // 2
    
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Apply kernel
            value = 0.0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    pi, pj = i - pad + ki, j - pad + kj
                    if 0 <= pi < height and 0 <= pj < width:
                        value += arr[pi, pj] * kernel[ki, kj]
            result[i, j] = value
    
    return result

def create_givens_3d_effect(image, xy_angle, xz_angle, yz_angle, depth_strength=0.3, brightness=1.0, quality='normal'):
    """
    Tạo hiệu ứng 3D bằng cách áp dụng nhiều Givens rotations
    Phiên bản đơn giản không cần scipy
    """
    
    # Điều chỉnh độ sáng
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    width, height = image.size
    
    # Tạo depth map từ luminance
    gray_img = image.convert('L')
    depth_map = np.array(gray_img) / 255.0
    
    # Simple smooth cho depth map (thay thế scipy)
    if quality in ['high', 'ultra']:
        depth_map = simple_gaussian_blur(depth_map, sigma=1.0)
    
    # Tạo mesh grid với density phù hợp
    if quality == 'ultra':
        step = 1
    elif quality == 'high':
        step = 2
    else:
        step = 4
    
    # Tạo 3D mesh coordinates
    mesh_points = []
    mesh_colors = []
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Normalized coordinates [-1, 1]
            norm_x = (x / width - 0.5) * 2.0
            norm_y = (y / height - 0.5) * 2.0
            norm_z = depth_map[y, x] * depth_strength  # Depth dựa trên brightness
            
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
    
    # Render với simple projection
    return render_simple_3d(mesh_points, mesh_colors, width, height, step)

def render_simple_3d(vertices, colors, width, height, step):
    """Render 3D mesh với simple projection"""
    
    # Simple perspective projection
    fov_factor = 0.8  # Field of view factor
    distance = 2.5
    
    projected_points = []
    
    for i, vertex in enumerate(vertices):
        x, y, z = vertex
        z_cam = z + distance
        
        if z_cam > 0.1:
            # Simple perspective projection
            px = x / z_cam * fov_factor
            py = y / z_cam * fov_factor
        else:
            px, py = x * 0.3, y * 0.3
        
        # Convert to screen coordinates
        screen_x = int((px + 1) * width * 0.4 + width * 0.1)
        screen_y = int((1 - py) * height * 0.4 + height * 0.1)
        
        # Clamp to screen bounds
        screen_x = max(0, min(width - 1, screen_x))
        screen_y = max(0, min(height - 1, screen_y))
        
        projected_points.append((screen_x, screen_y, colors[i], z_cam))
    
    # Sort by depth (back to front for proper rendering)
    projected_points.sort(key=lambda p: p[3], reverse=True)
    
    # Create result image with gradient background
    result_img = create_3d_background(width, height)
    
    # Render points
    for screen_x, screen_y, color, depth in projected_points:
        # Adaptive point size based on depth and step
        base_size = max(step, 1)
        depth_factor = max(0.4, min(1.2, distance / (depth + 0.1)))
        point_size = max(1, int(base_size * depth_factor))
        
        # Simple point rendering
        render_point(result_img, screen_x, screen_y, point_size, color, depth)
    
    return result_img

def create_3d_background(width, height):
    """Tạo background gradient cho 3D effect"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Simple radial gradient
    center_x, center_y = width // 2, height // 2
    max_radius = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(0, height, 2):  # Skip pixels for performance
        for x in range(0, width, 2):
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            factor = 1.0 - (distance / max_radius) * 0.6
            
            r = int(15 * factor)
            g = int(25 * factor)
            b = int(40 * factor)
            
            # Draw 2x2 block for efficiency
            draw.rectangle([x, y, x+1, y+1], fill=(r, g, b))
    
    return img

def render_point(img, x, y, size, color, depth):
    """Render single point với depth-based effects"""
    draw = ImageDraw.Draw(img)
    
    # Adjust color based on depth (closer = brighter)
    depth_factor = max(0.3, min(1.0, 2.0 / (depth + 0.5)))
    adjusted_color = [
        min(255, int(c * depth_factor)) for c in color
    ]
    
    if size <= 1:
        draw.point([x, y], fill=tuple(adjusted_color))
    else:
        # Draw filled circle
        draw.ellipse([
            x - size//2, y - size//2,
            x + size//2, y + size//2
        ], fill=tuple(adjusted_color))

def create_composite_rotation_matrix(xy_angle, xz_angle, yz_angle):
    """Tạo ma trận rotation tổng hợp từ các Givens rotations"""
    matrices = []
    
    if xy_angle != 0:
        matrices.append(('XY', givens_rotation_3d(xy_angle, 'xy')))
    if xz_angle != 0:
        matrices.append(('XZ', givens_rotation_3d(xz_angle, 'xz')))
    if yz_angle != 0:
        matrices.append(('YZ', givens_rotation_3d(yz_angle, 'yz')))
    
    # Composite matrix
    if matrices:
        composite = np.eye(3)
        for name, matrix in matrices:
            composite = np.dot(composite, matrix)
        return matrices, composite
    
    return [], np.eye(3)

# =================== STREAMLIT UI ===================

# Sidebar cho thông tin
with st.sidebar:
    st.markdown("### 🌐 Givens Rotation 3D")
    st.markdown("""
    **Givens rotation 3D** mở rộng phép xoay 
    lên không gian ba chiều với 3 mặt phẳng:
    
    **Ma trận cho mặt phẳng XY:**
    ```
    [cos θ -sin θ  0 ]
    [sin θ  cos θ  0 ]
    [ 0     0     1 ]
    ```
    
    **Tính chất:**
    - Bảo toàn độ dài và góc
    - Có thể compose nhiều rotations
    - Stable và invertible
    """)
    
    st.markdown("### 🎯 3D Planes")
    st.markdown("""
    - **XY**: Rotation quanh trục Z
    - **XZ**: Rotation quanh trục Y  
    - **YZ**: Rotation quanh trục X
    """)

# Upload ảnh
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh để áp dụng Givens Rotation 3D", 
    type=['png', 'jpg', 'jpeg'],
    help="Hỗ trợ file PNG, JPG, JPEG"
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Ảnh gốc", use_column_width=True)

        # Sidebar controls
        xy_angle = st.slider("Góc xoay XY (quanh trục Z)", -180, 180, 0, 1)
        xz_angle = st.slider("Góc xoay XZ (quanh trục Y)", -180, 180, 0, 1)
        yz_angle = st.slider("Góc xoay YZ (quanh trục X)", -180, 180, 0, 1)
        depth_strength = st.slider("Cồng sâu", 0.0, 1.0, 0.3, 0.01)
        brightness = st.slider("Độ sáng", 0.5, 2.0, 1.0, 0.05)
        quality = st.selectbox("Chất lượng render", ['normal', 'high', 'ultra'])

        # Render ảnh mới
        result_img = create_givens_3d_effect(
            image, xy_angle, xz_angle, yz_angle,
            depth_strength=depth_strength,
            brightness=brightness,
            quality=quality
        )

        st.image(result_img, caption="Ảnh sau khi xoay 3D", use_column_width=True)

    except:
        st.error("Ảnh không hợp lệ")
