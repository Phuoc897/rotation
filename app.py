import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import io
import math

# Import optional dependencies với error handling
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    st.warning("⚠️ OpenCV không khả dụng - một số tính năng 3D sẽ bị hạn chế")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Cấu hình trang
st.set_page_config(
    page_title="🎨 Chỉnh sửa ảnh 2D & 3D với Givens Rotation",
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
    
    .matrix-display {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    
    .math-formula {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
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
    
    .transform-info {
        background: #e8f4fd;
        border-left: 4px solid #4A90E2;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎨 Chỉnh sửa ảnh 2D & 3D với Givens Rotation</h1>', unsafe_allow_html=True)

# =================== ENHANCED GIVENS ROTATION MATRICES ===================

def givens_rotation_matrix_2d(theta):
    """
    Tạo ma trận xoay Givens 2D
    G(θ) = [cos(θ) -sin(θ)]
           [sin(θ)  cos(θ)]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    
    return np.array([
        [c, -s],
        [s, c]
    ])

def givens_rotation_matrix_3d(theta, axis='z'):
    """
    Tạo ma trận xoay Givens 3D cho các trục x, y, z
    
    Rx(θ) = [1    0       0   ]    - Xoay quanh trục X
            [0  cos(θ) -sin(θ)]
            [0  sin(θ)  cos(θ)]
    
    Ry(θ) = [ cos(θ) 0  sin(θ)]    - Xoay quanh trục Y  
            [   0    1    0   ]
            [-sin(θ) 0  cos(θ)]
    
    Rz(θ) = [cos(θ) -sin(θ) 0]     - Xoay quanh trục Z
            [sin(θ)  cos(θ) 0]
            [  0       0    1]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    
    if axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis phải là 'x', 'y', hoặc 'z'")

def combined_givens_rotation_3d(theta_x, theta_y, theta_z, order='zyx'):
    """
    Kết hợp các ma trận Givens rotation theo thứ tự được chỉ định
    
    Default order 'zyx': R = Rz(θz) × Ry(θy) × Rx(θx)
    Các order khác: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy'
    """
    Rx = givens_rotation_matrix_3d(theta_x, 'x')
    Ry = givens_rotation_matrix_3d(theta_y, 'y') 
    Rz = givens_rotation_matrix_3d(theta_z, 'z')
    
    if order.lower() == 'xyz':
        return np.dot(Rz, np.dot(Ry, Rx))  # R = Rz × Ry × Rx
    elif order.lower() == 'xzy':
        return np.dot(Ry, np.dot(Rz, Rx))  # R = Ry × Rz × Rx
    elif order.lower() == 'yxz':
        return np.dot(Rz, np.dot(Rx, Ry))  # R = Rz × Rx × Ry
    elif order.lower() == 'yzx':
        return np.dot(Rx, np.dot(Rz, Ry))  # R = Rx × Rz × Ry
    elif order.lower() == 'zxy':
        return np.dot(Ry, np.dot(Rx, Rz))  # R = Ry × Rx × Rz
    elif order.lower() == 'zyx':
        return np.dot(Rx, np.dot(Ry, Rz))  # R = Rx × Ry × Rz
    else:
        raise ValueError("Order không hợp lệ. Sử dụng: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'")

def scaling_matrix_3d(sx, sy, sz):
    """
    Ma trận scale 3D
    S = [sx  0   0 ]
        [0  sy   0 ]
        [0   0  sz ]
    """
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, sz]
    ])

def translation_matrix_3d(tx, ty, tz):
    """
    Ma trận translation 3D (homogeneous coordinates)
    T = [1  0  0  tx]
        [0  1  0  ty]
        [0  0  1  tz]
        [0  0  0   1]
    """
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def shear_matrix_2d(shx, shy):
    """
    Ma trận shear 2D
    Sh = [1   shx]
         [shy  1 ]
    """
    return np.array([
        [1, shx],
        [shy, 1]
    ])

def reflection_matrix_2d(axis='x'):
    """
    Ma trận phản chiếu 2D
    Refl_x = [1   0]  - Phản chiếu qua trục X
             [0  -1]
             
    Refl_y = [-1  0]  - Phản chiếu qua trục Y  
             [0   1]
    """
    if axis.lower() == 'x':
        return np.array([
            [1, 0],
            [0, -1]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [-1, 0],  
            [0, 1]
        ])
    else:
        raise ValueError("Axis phải là 'x' hoặc 'y'")

# =================== ENHANCED 2D IMAGE TRANSFORMATIONS ===================

def apply_2d_givens_rotation(image, theta, center=None):
    """
    Áp dụng Givens rotation cho ảnh 2D bằng ma trận transformation
    """
    if center is None:
        center = (image.width // 2, image.height // 2)
    
    # Tạo ma trận Givens rotation
    R = givens_rotation_matrix_2d(theta)
    
    # Chuyển đổi để xoay quanh center
    # T = T(center) × R × T(-center)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    cx, cy = center
    
    # Ma trận affine transformation cho PIL
    # [a c e]   [cos -sin cx(1-cos)+cy*sin]
    # [b d f] = [sin  cos cy(1-cos)-cx*sin]
    # [0 0 1]   [0    0          1        ]
    
    a = cos_theta
    b = sin_theta  
    c = -sin_theta
    d = cos_theta
    e = cx * (1 - cos_theta) + cy * sin_theta
    f = cy * (1 - cos_theta) - cx * sin_theta
    
    # Áp dụng transformation
    return image.transform(
        image.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
        fillcolor='white'
    )

def apply_2d_shear(image, shear_x, shear_y):
    """
    Áp dụng shear transformation cho ảnh 2D
    """
    # Ma trận shear
    shear_matrix = shear_matrix_2d(shear_x, shear_y)
    
    # Chuyển đổi thành affine transform cho PIL
    a, c = shear_matrix[0, :]
    b, d = shear_matrix[1, :]
    e, f = 0, 0
    
    return image.transform(
        image.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
        fillcolor='white'
    )

def apply_2d_reflection(image, axis='x'):
    """
    Áp dụng reflection cho ảnh 2D
    """
    if axis.lower() == 'x':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    elif axis.lower() == 'y':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image

def edit_image_2d_advanced(image, brightness=1.0, contrast=1.0, saturation=1.0, 
                          blur_radius=0, rotation=0, shear_x=0, shear_y=0, 
                          scale_x=1.0, scale_y=1.0, reflection=None):
    """
    Chỉnh sửa ảnh 2D nâng cao với các phép biến đổi Givens
    """
    if image is None:
        return None, {}
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transforms_applied = {}
    
    # Áp dụng các filter cơ bản
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        transforms_applied['brightness'] = f"Brightness: {brightness:.2f}"
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        transforms_applied['contrast'] = f"Contrast: {contrast:.2f}"
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
        transforms_applied['saturation'] = f"Saturation: {saturation:.2f}"
    
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        transforms_applied['blur'] = f"Gaussian Blur: {blur_radius}px"
    
    # Áp dụng scaling
    if scale_x != 1.0 or scale_y != 1.0:
        new_width = int(image.width * scale_x)
        new_height = int(image.height * scale_y)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        transforms_applied['scale'] = f"Scale: {scale_x:.2f}x, {scale_y:.2f}y"
    
    # Áp dụng reflection
    if reflection:
        image = apply_2d_reflection(image, reflection)
        transforms_applied['reflection'] = f"Reflection: {reflection}-axis"
    
    # Áp dụng shear
    if shear_x != 0 or shear_y != 0:
        image = apply_2d_shear(image, shear_x, shear_y)
        transforms_applied['shear'] = f"Shear: X={shear_x:.2f}, Y={shear_y:.2f}"
    
    # Áp dụng Givens rotation
    if rotation != 0:
        theta = np.radians(rotation)
        image = apply_2d_givens_rotation(image, theta)
        transforms_applied['rotation'] = f"Givens Rotation: {rotation}° ({theta:.3f} rad)"
    
    return image, transforms_applied

# =================== ENHANCED 3D TRANSFORMATIONS ===================

def create_enhanced_3d_mesh(image, depth_scale=50, mesh_resolution=50, depth_method='enhanced'):
    """
    Tạo mesh 3D nâng cao từ ảnh với nhiều phương pháp depth
    """
    # Resize để tối ưu performance
    resized_img = image.resize((mesh_resolution, mesh_resolution))
    img_array = np.array(resized_img)
    
    # Tạo depth map dựa trên method
    if depth_method == 'enhanced' and HAS_CV2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Combine multiple methods
        # 1. Brightness depth
        brightness_depth = gray.astype(np.float64)
        
        # 2. Edge depth
        edges = cv2.Canny(gray, 50, 150)
        edges_blur = cv2.GaussianBlur(edges, (3, 3), 0)
        
        # 3. Gradient depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_depth = np.sqrt(grad_x**2 + grad_y**2)
        
        # Combine với weights
        depth = (0.5 * brightness_depth + 
                0.3 * edges_blur.astype(np.float64) + 
                0.2 * gradient_depth)
                
    elif depth_method == 'laplacian' and HAS_CV2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        depth = np.abs(laplacian)
        
    else:
        # Fallback: brightness method
        if len(img_array.shape) == 3:
            depth = np.mean(img_array, axis=2)
        else:
            depth = img_array.astype(np.float64)
    
    # Normalize depth
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    else:
        depth = np.zeros_like(depth)
    
    # Tạo mesh
    vertices = []
    colors = []
    faces = []
    normals = []
    
    h, w = depth.shape
    
    # Tạo vertices với enhanced depth
    for y in range(h):
        for x in range(w):
            # Normalize coordinates
            vertex_x = (x - w/2) / w * 2
            vertex_y = (y - h/2) / h * 2  
            vertex_z = depth[y, x] * depth_scale / 100
            
            vertices.append([vertex_x, vertex_y, vertex_z])
            
            # Color từ ảnh gốc
            if len(img_array.shape) == 3:
                colors.append(img_array[y, x] / 255.0)
            else:
                gray_val = img_array[y, x] / 255.0
                colors.append([gray_val, gray_val, gray_val])
            
            # Calculate normal (simplified)
            normal = [0, 0, 1]  # Default up
            normals.append(normal)
    
    # Tạo faces với better triangulation
    for y in range(h-1):
        for x in range(w-1):
            i1 = y * w + x
            i2 = y * w + (x + 1)
            i3 = (y + 1) * w + x
            i4 = (y + 1) * w + (x + 1)
            
            # Two triangles per quad
            faces.append([i1, i2, i3])
            faces.append([i2, i4, i3])
    
    return np.array(vertices), np.array(colors), np.array(faces), np.array(normals), depth

def apply_3d_givens_transformations(vertices, theta_x, theta_y, theta_z, 
                                  scale_x=1.0, scale_y=1.0, scale_z=1.0,
                                  translate_x=0.0, translate_y=0.0, translate_z=0.0,
                                  rotation_order='zyx'):
    """
    Áp dụng các phép biến đổi 3D với Givens rotation matrices
    """
    # 1. Scaling transformation
    if scale_x != 1.0 or scale_y != 1.0 or scale_z != 1.0:
        S = scaling_matrix_3d(scale_x, scale_y, scale_z)
        vertices = np.dot(vertices, S.T)
    
    # 2. Givens rotation transformation
    R = combined_givens_rotation_3d(theta_x, theta_y, theta_z, rotation_order)
    vertices = np.dot(vertices, R.T)
    
    # 3. Translation
    if translate_x != 0.0 or translate_y != 0.0 or translate_z != 0.0:
        vertices[:, 0] += translate_x
        vertices[:, 1] += translate_y  
        vertices[:, 2] += translate_z
    
    return vertices, R, scaling_matrix_3d(scale_x, scale_y, scale_z)

def perspective_projection_enhanced(vertices_3d, fov=np.pi/4, aspect=1.0, 
                                  near=0.1, far=100.0, camera_distance=3.0):
    """
    Perspective projection nâng cao với ma trận projection
    """
    # Translate vertices relative to camera
    camera_vertices = vertices_3d.copy()
    camera_vertices[:, 2] += camera_distance
    
    # Perspective projection matrix
    f = 1.0 / np.tan(fov / 2.0)
    
    projection_matrix = np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0], 
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ])
    
    # Convert to homogeneous coordinates
    vertices_homo = np.hstack([camera_vertices, np.ones((camera_vertices.shape[0], 1))])
    
    # Apply projection
    projected_homo = np.dot(vertices_homo, projection_matrix.T)
    
    # Perspective divide
    projected_2d = []
    for i, vertex in enumerate(projected_homo):
        if vertex[3] != 0:
            x = vertex[0] / vertex[3]
            y = vertex[1] / vertex[3]
        else:
            x, y = 0, 0
        projected_2d.append([x, y])
    
    return np.array(projected_2d)

def render_3d_mesh_enhanced(vertices, colors, faces, projected_vertices, 
                          image_size=(800, 800), lighting=True):
    """
    Render mesh 3D với enhanced lighting và shading
    """
    img = Image.new('RGB', image_size, (20, 20, 30))  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Scale projected vertices to image coordinates  
    proj_scaled = projected_vertices.copy()
    proj_scaled[:, 0] = (proj_scaled[:, 0] + 1) * image_size[0] / 2
    proj_scaled[:, 1] = (proj_scaled[:, 1] + 1) * image_size[1] / 2
    
    # Sort faces by depth (painter's algorithm)
    face_depths = []
    for face in faces:
        avg_z = np.mean([vertices[i][2] for i in face])
        face_depths.append(avg_z) 
    
    # Sort faces back to front
    sorted_indices = np.argsort(face_depths)
    
    # Light setup
    light_direction = np.array([0.0, 0.0, 1.0])  # Light from camera
    light_direction = light_direction / np.linalg.norm(light_direction)
    
    # Render faces
    for idx in sorted_indices:
        face = faces[idx]
        
        try:
            # Get triangle vertices in 2D
            triangle_2d = []
            for vertex_idx in face:
                if 0 <= vertex_idx < len(proj_scaled):
                    point = proj_scaled[vertex_idx]
                    triangle_2d.append((int(point[0]), int(point[1])))
            
            if len(triangle_2d) == 3:
                # Calculate face normal for lighting
                if lighting:
                    v1 = vertices[face[1]] - vertices[face[0]]
                    v2 = vertices[face[2]] - vertices[face[0]]
                    normal = np.cross(v1, v2)
                    if np.linalg.norm(normal) > 0:
                        normal = normal / np.linalg.norm(normal)
                        
                        # Calculate lighting
                        light_intensity = max(0.2, np.dot(normal, light_direction))
                    else:
                        light_intensity = 0.5
                else:
                    light_intensity = 1.0
                
                # Get face color
                face_colors = [colors[i] for i in face if i < len(colors)]
                if face_colors:
                    avg_color = np.mean(face_colors, axis=0)
                    # Apply lighting
                    lit_color = avg_color * light_intensity
                    color_int = tuple(np.clip(lit_color * 255, 0, 255).astype(int))
                    
                    # Draw triangle
                    draw.polygon(triangle_2d, fill=color_int, outline=color_int)
        except:
            continue
    
    return img

# =================== MAIN APPLICATION ===================

# Upload ảnh
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh để chỉnh sửa", 
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
    help="Hỗ trợ: PNG, JPG, JPEG, GIF, BMP, TIFF"
)

if uploaded_file is not None:
    # Đọc ảnh gốc
    original_image = Image.open(uploaded_file)
    
    # Tạo tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Chỉnh sửa 2D với Givens", "🌐 Hiệu ứng 3D với Givens", "📊 Ma trận & Công thức"])
    
    with tab1:
        st.subheader("🖼️ Chỉnh sửa 2D với Givens Rotation")
        
        # Hiển thị ảnh gốc
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📷 Ảnh gốc:**")
            st.image(original_image, use_column_width=True)
        
        # Controls cho 2D
        st.sidebar.markdown("## 🎛️ Tham số 2D Givens")
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # Basic adjustments
        st.sidebar.markdown("### 🎨 Chỉnh sửa cơ bản")
        brightness = st.sidebar.slider("🔆 Độ sáng", 0.0, 2.0, 1.0, 0.1)
        contrast = st.sidebar.slider("🌈 Độ tương phản", 0.0, 2.0, 1.0, 0.1)  
        saturation = st.sidebar.slider("🎨 Độ bão hòa", 0.0, 2.0, 1.0, 0.1)
        blur_radius = st.sidebar.slider("🔲 Làm mờ", 0, 10, 0, 1)
        
        # Givens transformations
        st.sidebar.markdown("### 🔄 Phép biến đổi Givens")
        rotation_2d = st.sidebar.slider("🌀 Givens Rotation (độ)", -180, 180, 0, 5)
        shear_x = st.sidebar.slider("📐 Shear X", -1.0, 1.0, 0.0, 0.1)
        shear_y = st.sidebar.slider("📐 Shear Y", -1.0, 1.0, 0.0, 0.1)
        scale_x = st.sidebar.slider("📏 Scale X", 0.1, 3.0, 1.0, 0.1)
        scale_y = st.sidebar.slider("📏 Scale Y", 0.1, 3.0, 1.0, 0.1)
        
        reflection_options = [None, 'x', 'y']
        reflection = st.sidebar.selectbox("🔄 Reflection", reflection_options)
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Áp dụng transformations
        edited_image, transforms_applied = edit_image_2d_advanced(
            original_image.copy(), brightness, contrast, saturation, blur_radius,
            rotation_2d, shear_x, shear_y, scale_x, scale_y, reflection
        )
        
        with col2:
            st.markdown("**✨ Ảnh đã chỉnh sửa:**") 
            st.image(edited_image, use_column_width=True)
        
        # Hiển thị ma trận 2D
        st.markdown("### 📊 Ma trận Givens 2D được áp dụng")
        
        if rotation_2d != 0:
            theta = np.radians(rotation_2d)
            rotation_matrix = givens_rotation_matrix_2d(theta)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="math-formula">', unsafe_allow_html=True)
                st.markdown(f"**Givens Rotation Matrix G({rotation_2d}°)**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                st.code(f"""
G(θ) = [  {rotation_matrix[0,0]:7.4f}   {rotation_matrix[0,1]:7.4f} ]
       [  {rotation_matrix[1,0]:7.4f}   {rotation_matrix[1,1]:7.4f} ]

θ = {rotation_2d}° = {theta:.4f} radians
cos(θ) = {np.cos(theta):7.4f}
sin(θ) = {np.sin(theta):7.4f}
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        if shear_x != 0 or shear_y != 0:
            shear_matrix = shear_matrix_2d(shear_x, shear_y)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="math-formula">', unsafe_allow_html=True)
                st.markdown(f"**Shear Matrix S({shear_x:.2f}, {shear_y:.2f})**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                st.code(f"""
S = [  {shear_matrix[0,0]:7.4f}   {shear_matrix[0,1]:7.4f} ]
    [  {shear_matrix[1,0]:7.4f}   {shear_matrix[1,1]:7.4f} ]
                """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Hiển thị thông tin transforms
        if transforms_applied:
            st.markdown("### 🔧 Phép biến đổi đã áp dụng")
            st.markdown('<div class="transform-info">', unsafe_allow_html=True)
            for key, value in transforms_applied.items():
                st.markdown(f"• **{value}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button
        if edited_image:
            buf = io.BytesIO()
            edited_image.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="💾 Tải xuống ảnh 2D",
                data=buf,
                file_name=f"edited_2d_{uploaded_file.name}",
                mime="image/png"
            )
    
    with tab2:
        st.subheader("🌐 Hiệu ứng 3D với Givens Rotation")
        
        # Sidebar controls cho 3D
        st.sidebar.markdown("## 🎛️ Tham số 3D Givens")
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # 3D Givens Rotations
        st.sidebar.markdown("### 🔄 Givens Rotations 3D")
        theta_x = st.sidebar.slider("🔄 Rotation X (độ)", -180, 180, 0, 5)
        theta_y = st.sidebar.slider("🔄 Rotation Y (độ)", -180, 180, 0, 5)
        theta_z = st.sidebar.slider("🔄 Rotation Z (độ)", -180, 180, 0, 5)
        
        rotation_order = st.sidebar.selectbox(
            "📐 Thứ tự rotation", 
            ['zyx', 'xyz', 'xzy', 'yxz', 'yzx', 'zxy'],
            help="Thứ tự áp dụng các rotation matrices"
        )
        
        # 3D Transformations
        st.sidebar.markdown("### 📏 Scale 3D")
        scale_3d_x = st.sidebar.slider("📏 Scale X", 0.1, 3.0, 1.0, 0.1)
        scale_3d_y = st.sidebar.slider("📏 Scale Y", 0.1, 3.0, 1.0, 0.1)
        scale_3d_z = st.sidebar.slider("📏 Scale Z", 0.1, 3.0, 1.0, 0.1)
        
        st.sidebar.markdown("### 📍 Translation 3D")
        translate_x = st.sidebar.slider("📍 Translate X", -2.0, 2.0, 0.0, 0.1)
        translate_y = st.sidebar.slider("📍 Translate Y", -2.0, 2.0, 0.0, 0.1)
        translate_z = st.sidebar.slider("📍 Translate Z", -2.0, 2.0, 0.0, 0.1)
        
        # Mesh parameters
        st.sidebar.markdown("### 🕸️ Tham số Mesh")
        depth_scale = st.sidebar.slider("🏔️ Độ sâu", 1, 100, 30, 5)
        mesh_resolution = st.sidebar.slider("🔍 Độ phân giải", 20, 100, 50, 10)
        
        depth_methods = ['enhanced', 'laplacian', 'brightness']
        depth_method = st.sidebar.selectbox("🎨 Phương pháp depth", depth_methods)
        
        # Camera parameters
        st.sidebar.markdown("### 📹 Tham số Camera")
        camera_distance = st.sidebar.slider("📏 Khoảng cách camera", 1.0, 10.0, 3.0, 0.5)
        fov = st.sidebar.slider("🔍 Field of View", 30, 120, 45, 5)
        
        lighting = st.sidebar.checkbox("💡 Lighting", True)
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Tạo và xử lý 3D mesh
        if st.button("🚀 Tạo hiệu ứng 3D"):
            with st.spinner("🔄 Đang tạo mesh 3D..."):
                # Tạo mesh 3D
                vertices, colors, faces, normals, depth_map = create_enhanced_3d_mesh(
                    original_image, depth_scale, mesh_resolution, depth_method
                )
                
                # Áp dụng transformations
                theta_x_rad = np.radians(theta_x)
                theta_y_rad = np.radians(theta_y)
                theta_z_rad = np.radians(theta_z)
                
                transformed_vertices, rotation_matrix, scale_matrix = apply_3d_givens_transformations(
                    vertices, theta_x_rad, theta_y_rad, theta_z_rad,
                    scale_3d_x, scale_3d_y, scale_3d_z,
                    translate_x, translate_y, translate_z,
                    rotation_order
                )
                
                # Perspective projection
                fov_rad = np.radians(fov)
                projected_vertices = perspective_projection_enhanced(
                    transformed_vertices, fov_rad, aspect=1.0, 
                    camera_distance=camera_distance
                )
                
                # Render mesh
                rendered_3d = render_3d_mesh_enhanced(
                    transformed_vertices, colors, faces, projected_vertices,
                    image_size=(800, 800), lighting=lighting
                )
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📷 Ảnh gốc:**")
                    st.image(original_image, use_column_width=True)
                    
                    st.markdown("**🗺️ Depth Map:**")
                    depth_img = Image.fromarray((depth_map * 255).astype(np.uint8))
                    st.image(depth_img, use_column_width=True)
                
                with col2:
                    st.markdown("**🌐 Ảnh 3D với Givens:**")
                    st.image(rendered_3d, use_column_width=True)
                
                # Hiển thị ma trận 3D
                st.markdown("### 📊 Ma trận Givens 3D được áp dụng")
                
                # Individual rotation matrices
                if theta_x != 0 or theta_y != 0 or theta_z != 0:
                    Rx = givens_rotation_matrix_3d(theta_x_rad, 'x')
                    Ry = givens_rotation_matrix_3d(theta_y_rad, 'y')
                    Rz = givens_rotation_matrix_3d(theta_z_rad, 'z')
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
                        st.markdown(f"**Rx({theta_x}°)**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                        st.code(f"""
Rx = [{Rx[0,0]:6.3f} {Rx[0,1]:6.3f} {Rx[0,2]:6.3f}]
     [{Rx[1,0]:6.3f} {Rx[1,1]:6.3f} {Rx[1,2]:6.3f}]
     [{Rx[2,0]:6.3f} {Rx[2,1]:6.3f} {Rx[2,2]:6.3f}]
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
                        st.markdown(f"**Ry({theta_y}°)**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                        st.code(f"""
Ry = [{Ry[0,0]:6.3f} {Ry[0,1]:6.3f} {Ry[0,2]:6.3f}]
     [{Ry[1,0]:6.3f} {Ry[1,1]:6.3f} {Ry[1,2]:6.3f}]
     [{Ry[2,0]:6.3f} {Ry[2,1]:6.3f} {Ry[2,2]:6.3f}]
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
                        st.markdown(f"**Rz({theta_z}°)**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                        st.code(f"""
Rz = [{Rz[0,0]:6.3f} {Rz[0,1]:6.3f} {Rz[0,2]:6.3f}]
     [{Rz[1,0]:6.3f} {Rz[1,1]:6.3f} {Rz[1,2]:6.3f}]
     [{Rz[2,0]:6.3f} {Rz[2,1]:6.3f} {Rz[2,2]:6.3f}]
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Combined rotation matrix
                st.markdown('<div class="math-formula">', unsafe_allow_html=True)
                st.markdown(f"**Ma trận rotation kết hợp (order: {rotation_order.upper()})**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                    st.code(f"""
R_combined = [{rotation_matrix[0,0]:7.4f} {rotation_matrix[0,1]:7.4f} {rotation_matrix[0,2]:7.4f}]
             [{rotation_matrix[1,0]:7.4f} {rotation_matrix[1,1]:7.4f} {rotation_matrix[1,2]:7.4f}]
             [{rotation_matrix[2,0]:7.4f} {rotation_matrix[2,1]:7.4f} {rotation_matrix[2,2]:7.4f}]
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Scale matrix
                if scale_3d_x != 1.0 or scale_3d_y != 1.0 or scale_3d_z != 1.0:
                    st.markdown('<div class="math-formula">', unsafe_allow_html=True)
                    st.markdown(f"**Ma trận Scale 3D**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
                        st.code(f"""
S = [{scale_matrix[0,0]:7.4f} {scale_matrix[0,1]:7.4f} {scale_matrix[0,2]:7.4f}]
    [{scale_matrix[1,0]:7.4f} {scale_matrix[1,1]:7.4f} {scale_matrix[1,2]:7.4f}]
    [{scale_matrix[2,0]:7.4f} {scale_matrix[2,1]:7.4f} {scale_matrix[2,2]:7.4f}]
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # 3D Statistics
                st.markdown("### 📈 Thống kê 3D")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔺 Vertices", len(vertices))
                with col2:
                    st.metric("📐 Faces", len(faces))
                with col3:
                    st.metric("🎨 Colors", len(colors))
                with col4:
                    st.metric("📏 Resolution", f"{mesh_resolution}x{mesh_resolution}")
                
                # Download 3D result
                buf_3d = io.BytesIO()
                rendered_3d.save(buf_3d, format='PNG')
                buf_3d.seek(0)
                
                st.download_button(
                    label="💾 Tải xuống ảnh 3D",
                    data=buf_3d,
                    file_name=f"3d_givens_{uploaded_file.name}",
                    mime="image/png"
                )
        
        else:
            st.info("👆 Nhấn nút 'Tạo hiệu ứng 3D' để xem kết quả")
    
    with tab3:
        st.subheader("📊 Ma trận & Công thức Givens Rotation")
        
        # Theory section
        st.markdown("### 🧮 Lý thuyết Ma trận Givens")
        
        st.markdown("""
        **Givens Rotation** là một phép biến đổi trực giao được sử dụng để xoay vector trong không gian 2D hoặc 3D.
        Ma trận Givens có tính chất đặc biệt là **trực giao** (orthogonal), nghĩa là G^T × G = I.
        """)
        
        # 2D Givens
        st.markdown("#### 🔄 Ma trận Givens 2D")
        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
        st.markdown("**Công thức tổng quát:**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
        st.code("""
G₂D(θ) = [cos(θ)  -sin(θ)]
         [sin(θ)   cos(θ)]

Với θ là góc xoay (radian)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("**Tính chất:**")
        st.markdown("""
        • **Trực giao**: G^T × G = I
        • **Det(G) = 1**: Bảo toàn thể tích
        • **Nghịch đảo**: G^(-1) = G^T = G(-θ)
        """)
        
        # 3D Givens
        st.markdown("#### 🌐 Ma trận Givens 3D")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Rotation quanh trục X:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code("""
Rx(θ) = [1    0       0   ]
        [0  cos(θ) -sin(θ)]
        [0  sin(θ)  cos(θ)]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Rotation quanh trục Y:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code("""
Ry(θ) = [ cos(θ) 0  sin(θ)]
        [   0    1    0   ]
        [-sin(θ) 0  cos(θ)]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("**Rotation quanh trục Z:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code("""
Rz(θ) = [cos(θ) -sin(θ) 0]
        [sin(θ)  cos(θ) 0]
        [  0       0    1]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Combined rotations
        st.markdown("#### 🔗 Kết hợp Rotations 3D")
        st.markdown("""
        Khi kết hợp nhiều rotation, thứ tự nhân ma trận rất quan trọng:
        """)
        
        st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
        st.code("""
Rotation Orders:
• ZYX: R = Rx(θx) × Ry(θy) × Rz(θz)  [Euler angles]
• XYZ: R = Rz(θz) × Ry(θy) × Rx(θx)  [Roll-Pitch-Yaw]
• ZXY: R = Ry(θy) × Rx(θx) × Rz(θz)  [Alternative]

Lưu ý: A × B ≠ B × A (không giao hoán)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Other transformations
        st.markdown("#### 🔧 Các phép biến đổi khác")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ma trận Scale:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code("""
S₃D = [sx  0   0 ]
      [0  sy   0 ]
      [0   0  sz ]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("**Ma trận Shear 2D:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code("""
Sh = [1   shx]
     [shy  1 ]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Ma trận Translation 3D:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code("""
T₃D = [1  0  0  tx]
      [0  1  0  ty]
      [0  0  1  tz]
      [0  0  0   1]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("**Ma trận Reflection:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code("""
Refl_x = [1   0]  Refl_y = [-1  0]
         [0  -1]           [0   1]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Applications
        st.markdown("### 🚀 Ứng dụng trong Computer Graphics")
        
        st.markdown("""
        **1. 🎮 Game Development:**
        • Xoay nhân vật, object trong game
        • Animation và chuyển động
        • Camera controls
        
        **2. 🎬 Computer Vision:**
        • Image registration và alignment
        • Object detection và tracking
        • Augmented Reality (AR)
        
        **3. 🏗️ 3D Modeling:**
        • Mesh transformations
        • Skeletal animation
        • Geometric modeling
        
        **4. 🔬 Scientific Computing:**
        • Numerical linear algebra
        • QR decomposition
        • Eigenvalue problems
        """)
        
        # Interactive demo
        st.markdown("### 🎯 Demo tương tác")
        
        demo_angle = st.slider("🔄 Góc xoay demo (độ)", 0, 360, 45, 15)
        demo_theta = np.radians(demo_angle)
        demo_matrix = givens_rotation_matrix_2d(demo_theta)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ma trận Givens:**")
            st.markdown('<div class="matrix-display">', unsafe_allow_html=True)
            st.code(f"""
G({demo_angle}°) = [{demo_matrix[0,0]:7.4f} {demo_matrix[0,1]:7.4f}]
           [{demo_matrix[1,0]:7.4f} {demo_matrix[1,1]:7.4f}]
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Tính chất:**")
            det = np.linalg.det(demo_matrix)
            st.markdown(f"• **Determinant:** {det:.6f}")
            st.markdown(f"• **cos({demo_angle}°):** {np.cos(demo_theta):.4f}")
            st.markdown(f"• **sin({demo_angle}°):** {np.sin(demo_theta):.4f}")
            st.markdown(f"• **Orthogonal:** {'✅' if np.allclose(np.dot(demo_matrix.T, demo_matrix), np.eye(2)) else '❌'}")
        
        # Performance notes
        st.markdown("### ⚡ Lưu ý Performance")
        
        st.markdown("""
        **Tối ưu hóa:**
        
        🟢 **Nhanh:**
        • Sử dụng NumPy vectorized operations
        • Pre-compute sin/cos values
        • Batch processing cho nhiều vertices
        
        🟡 **Trung bình:**
        • Loop qua từng vertex riêng lẻ
        • Tính toán realtime cho large meshes
        
        🔴 **Chậm:**
        • Python loops thuần túy
        • Không sử dụng matrix operations
        • Recompute matrices mỗi frame
        """)
        
        st.markdown('<div class="transform-info">', unsafe_allow_html=True)
        st.markdown("""
        💡 **Pro Tips:**
        • Kết hợp multiple transformations thành 1 ma trận duy nhất
        • Sử dụng homogeneous coordinates cho 3D transformations
        • Cache computed matrices khi có thể
        • Sử dụng GPU acceleration cho large datasets
        """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Landing page
    st.markdown("""
    ## 👋 Chào mừng đến với Givens Rotation Image Editor!
    
    ### 🎯 Tính năng chính:
    
    **🖼️ Chỉnh sửa 2D:**
    • Givens Rotation với góc tùy chỉnh
    • Shear, Scale, Reflection transformations
    • Brightness, Contrast, Saturation adjustments
    • Gaussian Blur effects
    
    **🌐 Hiệu ứng 3D:**
    • Tạo mesh 3D từ ảnh 2D với depth mapping
    • Áp dụng Givens rotation cho cả 3 trục X, Y, Z
    • Multiple rotation orders (ZYX, XYZ, etc.)
    • Enhanced lighting và shading
    • Perspective projection
    
    **📊 Visualization:**
    • Hiển thị ma trận transformations
    • Interactive demos
    • Real-time parameter adjustments
    
    ### 🚀 Cách sử dụng:
    1. **Upload ảnh** bằng cách nhấn nút "Chọn ảnh" ở sidebar
    2. **Chọn tab** để chỉnh sửa 2D hoặc tạo hiệu ứng 3D
    3. **Điều chỉnh tham số** bằng các slider
    4. **Xem kết quả** và download ảnh đã chỉnh sửa
    
    ### 📚 Về Givens Rotation:
    Givens Rotation là một phép biến đổi trực giao fundamental trong linear algebra, 
    được sử dụng rộng rãi trong computer graphics, computer vision, và scientific computing.
    
    ---
    **📁 Hãy upload một ảnh để bắt đầu!**
    """)
    
    # Example images section
    st.markdown("### 🖼️ Ví dụ kết quả:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original Image**")
        st.markdown("📷 Ảnh gốc")
    with col2:
        st.markdown("**2D Givens Rotation**")
        st.markdown("🔄 Xoay + chỉnh sửa 2D")
    with col3:
        st.markdown("**3D Mesh Effect**")
        st.markdown("🌐 Hiệu ứng 3D với depth")
    
    st.markdown("""
    ### 🔧 Yêu cầu hệ thống:
    - **Python 3.7+**
    - **Required:** Streamlit, NumPy, PIL
    - **Optional:** OpenCV (cho enhanced 3D effects), Matplotlib (cho advanced visualization)
    
    ### 📖 Supported Formats:
    PNG, JPG, JPEG, GIF, BMP, TIFF
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🎨 <strong>Givens Rotation Image Editor</strong> - Powered by Mathematical Transformations</p>
    <p>📊 Built with Streamlit • 🧮 Linear Algebra • 🎯 Computer Graphics</p>
</div>
""", unsafe_allow_html=True)
