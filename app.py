import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import io
import math

# Cấu hình trang
st.set_page_config(
    page_title="🎨 Givens Rotation Image Editor",
    page_icon="🎨",
    layout="wide"
)

# CSS đơn giản
st.markdown("""
<style>
    .matrix-box {
        background: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        margin: 10px 0;
    }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #4A90E2;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎨 Givens Rotation Image Editor")
st.markdown("Chỉnh sửa ảnh với phép biến đổi Givens Rotation")

# =================== GIVENS ROTATION FUNCTIONS ===================

def givens_2d(theta):
    """Tạo ma trận Givens 2D"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

def givens_3d(theta, axis='z'):
    """Tạo ma trận Givens 3D cho trục x, y, z"""
    c = np.cos(theta)
    s = np.sin(theta)
    
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotate_image_2d(image, angle_deg):
    """Xoay ảnh 2D bằng Givens rotation"""
    theta = np.radians(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Tâm xoay
    cx, cy = image.width // 2, image.height // 2
    
    # Ma trận affine cho PIL
    a, b = c, s
    c_val, d = -s, c
    e = cx * (1 - c) + cy * s
    f = cy * (1 - c) - cx * s
    
    return image.transform(
        image.size, Image.AFFINE,
        (a, b, c_val, d, e, f),
        resample=Image.BICUBIC,
        fillcolor='white'
    )

def create_3d_mesh(image, depth_scale=30, resolution=40):
    """Tạo mesh 3D đơn giản từ ảnh"""
    # Resize ảnh
    img_small = image.resize((resolution, resolution))
    img_array = np.array(img_small)
    
    # Tạo depth map từ brightness
    if len(img_array.shape) == 3:
        depth = np.mean(img_array, axis=2)
    else:
        depth = img_array
    
    # Normalize depth
    depth = depth / 255.0 * depth_scale / 100
    
    # Tạo vertices
    vertices = []
    colors = []
    
    h, w = depth.shape
    for y in range(h):
        for x in range(w):
            # Tọa độ 3D
            vertex_x = (x - w/2) / w * 2
            vertex_y = (y - h/2) / h * 2
            vertex_z = depth[y, x]
            
            vertices.append([vertex_x, vertex_y, vertex_z])
            
            # Màu từ ảnh
            if len(img_array.shape) == 3:
                colors.append(img_array[y, x] / 255.0)
            else:
                gray = img_array[y, x] / 255.0
                colors.append([gray, gray, gray])
    
    return np.array(vertices), np.array(colors), (h, w)

def apply_3d_rotation(vertices, rx, ry, rz):
    """Áp dụng rotation 3D"""
    # Chuyển sang radian
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)
    
    # Tạo ma trận rotation
    Rx = givens_3d(rx_rad, 'x')
    Ry = givens_3d(ry_rad, 'y')
    Rz = givens_3d(rz_rad, 'z')
    
    # Kết hợp rotation (Z * Y * X)
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # Áp dụng rotation
    rotated = np.dot(vertices, R.T)
    
    return rotated, R

def project_3d_to_2d(vertices_3d, distance=3):
    """Chiếu 3D lên 2D đơn giản"""
    projected = []
    for vertex in vertices_3d:
        x, y, z = vertex
        z_cam = z + distance
        if z_cam > 0.1:  # Tránh chia cho 0
            px = x / z_cam
            py = y / z_cam
        else:
            px, py = 0, 0
        projected.append([px, py])
    
    return np.array(projected)

def render_3d_mesh(vertices_3d, colors, projected_2d, mesh_size, image_size=600):
    """Render mesh 3D đơn giản"""
    img = Image.new('RGB', (image_size, image_size), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    
    # Scale tọa độ 2D
    proj_scaled = projected_2d.copy()
    proj_scaled[:, 0] = (proj_scaled[:, 0] + 1) * image_size / 2
    proj_scaled[:, 1] = (proj_scaled[:, 1] + 1) * image_size / 2
    
    h, w = mesh_size
    
    # Vẽ các điểm
    for i, (point, color) in enumerate(zip(proj_scaled, colors)):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_size and 0 <= y < image_size:
            # Tính màu
            color_int = tuple((color * 255).astype(int))
            # Vẽ điểm
            draw.ellipse([x-2, y-2, x+2, y+2], fill=color_int)
    
    return img

# =================== MAIN APP ===================

# Upload file
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh", 
    type=['png', 'jpg', 'jpeg'],
    help="Hỗ trợ PNG, JPG, JPEG"
)

if uploaded_file:
    # Đọc ảnh
    original_image = Image.open(uploaded_file)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Chỉnh sửa 2D", "🌐 Hiệu ứng 3D", "📊 Ma trận"])
    
    # TAB 1: 2D EDITING
    with tab1:
        st.subheader("🖼️ Chỉnh sửa 2D với Givens Rotation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ảnh gốc:**")
            st.image(original_image, use_column_width=True)
        
        # Controls
        st.sidebar.header("🎛️ Tham số 2D")
        
        # Basic adjustments
        brightness = st.sidebar.slider("🔆 Độ sáng", 0.5, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("🌈 Độ tương phản", 0.5, 2.0, 1.0, 0.1)
        blur = st.sidebar.slider("🔲 Làm mờ", 0, 5, 0)
        
        # Givens rotation
        rotation = st.sidebar.slider("🔄 Givens Rotation (độ)", -180, 180, 0, 15)
        
        # Áp dụng chỉnh sửa
        edited_image = original_image.copy()
        
        # Brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(edited_image)
            edited_image = enhancer.enhance(brightness)
        
        # Contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(edited_image)
            edited_image = enhancer.enhance(contrast)
        
        # Blur
        if blur > 0:
            edited_image = edited_image.filter(ImageFilter.GaussianBlur(blur))
        
        # Givens rotation
        if rotation != 0:
            edited_image = rotate_image_2d(edited_image, rotation)
        
        with col2:
            st.markdown("**Ảnh đã chỉnh sửa:**")
            st.image(edited_image, use_column_width=True)
        
        # Hiển thị ma trận 2D
        if rotation != 0:
            st.markdown("### 📊 Ma trận Givens 2D")
            theta = np.radians(rotation)
            matrix = givens_2d(theta)
            
            st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
            st.code(f"""
Givens Matrix G({rotation}°):
[{matrix[0,0]:7.4f}  {matrix[0,1]:7.4f}]
[{matrix[1,0]:7.4f}  {matrix[1,1]:7.4f}]

θ = {rotation}° = {theta:.4f} radians
cos(θ) = {np.cos(theta):7.4f}
sin(θ) = {np.sin(theta):7.4f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download
        if edited_image:
            buf = io.BytesIO()
            edited_image.save(buf, format='PNG')
            st.download_button(
                "💾 Tải xuống",
                buf.getvalue(),
                f"edited_{uploaded_file.name}",
                "image/png"
            )
    
    # TAB 2: 3D EFFECTS
    with tab2:
        st.subheader("🌐 Hiệu ứng 3D với Givens Rotation")
        
        # 3D Controls
        st.sidebar.header("🎛️ Tham số 3D")
        
        rx = st.sidebar.slider("🔄 Rotation X", -180, 180, 0, 15)
        ry = st.sidebar.slider("🔄 Rotation Y", -180, 180, 0, 15)
        rz = st.sidebar.slider("🔄 Rotation Z", -180, 180, 0, 15)
        
        depth_scale = st.sidebar.slider("🏔️ Độ sâu", 10, 100, 30, 10)
        resolution = st.sidebar.slider("🔍 Độ phân giải", 20, 60, 40, 10)
        
        if st.button("🚀 Tạo hiệu ứng 3D"):
            with st.spinner("Đang xử lý..."):
                # Tạo mesh 3D
                vertices, colors, mesh_size = create_3d_mesh(
                    original_image, depth_scale, resolution
                )
                
                # Áp dụng rotation
                rotated_vertices, rotation_matrix = apply_3d_rotation(vertices, rx, ry, rz)
                
                # Chiếu lên 2D
                projected = project_3d_to_2d(rotated_vertices)
                
                # Render
                result_3d = render_3d_mesh(rotated_vertices, colors, projected, mesh_size)
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Ảnh gốc:**")
                    st.image(original_image, use_column_width=True)
                
                with col2:
                    st.markdown("**Hiệu ứng 3D:**")
                    st.image(result_3d, use_column_width=True)
                
                # Ma trận 3D
                st.markdown("### 📊 Ma trận Rotation 3D")
                st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
                st.code(f"""
Combined Rotation Matrix:
[{rotation_matrix[0,0]:7.4f}  {rotation_matrix[0,1]:7.4f}  {rotation_matrix[0,2]:7.4f}]
[{rotation_matrix[1,0]:7.4f}  {rotation_matrix[1,1]:7.4f}  {rotation_matrix[1,2]:7.4f}]
[{rotation_matrix[2,0]:7.4f}  {rotation_matrix[2,1]:7.4f}  {rotation_matrix[2,2]:7.4f}]

Rotations: X={rx}°, Y={ry}°, Z={rz}°
Vertices: {len(vertices)}
Resolution: {resolution}x{resolution}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download 3D
                buf_3d = io.BytesIO()
                result_3d.save(buf_3d, format='PNG')
                st.download_button(
                    "💾 Tải xuống 3D",
                    buf_3d.getvalue(),
                    f"3d_{uploaded_file.name}",
                    "image/png"
                )
    
    # TAB 3: MATRICES
    with tab3:
        st.subheader("📊 Ma trận & Công thức Givens")
        
        st.markdown("### 🧮 Ma trận Givens 2D")
        st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
        st.code("""
G₂D(θ) = [cos(θ)  -sin(θ)]
         [sin(θ)   cos(θ)]

Tính chất:
• Trực giao: G^T × G = I
• Det(G) = 1
• G^(-1) = G^T = G(-θ)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🌐 Ma trận Givens 3D")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Trục X:**")
            st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
            st.code("""
Rx(θ) = [1    0       0   ]
        [0  cos(θ) -sin(θ)]
        [0  sin(θ)  cos(θ)]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Trục Y:**")
            st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
            st.code("""
Ry(θ) = [ cos(θ) 0  sin(θ)]
        [   0    1    0   ]
        [-sin(θ) 0  cos(θ)]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("**Trục Z:**")
            st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
            st.code("""
Rz(θ) = [cos(θ) -sin(θ) 0]
        [sin(θ)  cos(θ) 0]
        [  0       0    1]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo tương tác
        st.markdown("### 🎯 Demo tương tác")
        demo_angle = st.slider("Góc demo", 0, 360, 45, 15)
        demo_matrix = givens_2d(np.radians(demo_angle))
        
        st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
        st.code(f"""
G({demo_angle}°) = [{demo_matrix[0,0]:7.4f}  {demo_matrix[0,1]:7.4f}]
           [{demo_matrix[1,0]:7.4f}  {demo_matrix[1,1]:7.4f}]

Determinant: {np.linalg.det(demo_matrix):.6f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Landing page
    st.markdown("""
    ## 👋 Chào mừng đến với Givens Rotation Image Editor!
    
    ### 🎯 Tính năng:
    
    **🖼️ Chỉnh sửa 2D:**
    • Givens Rotation với góc tùy chỉnh
    • Điều chỉnh độ sáng, tương phản
    • Làm mờ Gaussian
    
    **🌐 Hiệu ứng 3D:**
    • Tạo mesh 3D từ ảnh
    • Rotation 3 trục với Givens
    • Projection và rendering
    
    **📊 Ma trận:**
    • Hiển thị ma trận transformation
    • Demo tương tác
    • Công thức toán học
    
    ### 🚀 Cách sử dụng:
    1. Upload ảnh (PNG, JPG, JPEG)
    2. Chọn tab để chỉnh sửa 2D hoặc tạo 3D
    3. Điều chỉnh tham số
    4. Download kết quả
    
    **📁 Hãy upload ảnh để bắt đầu!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🎨 <strong>Givens Rotation Image Editor</strong> - Simple & Effective
</div>
""", unsafe_allow_html=True)
