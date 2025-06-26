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

# CSS với interactive 3D
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
    .canvas-container {
        position: relative;
        border: 2px solid #4A90E2;
        border-radius: 10px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        margin: 10px 0;
    }
    .control-panel {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .interactive-hint {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
        margin: 10px 0;
        text-align: center;
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
    faces = []
    
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
    
    # Tạo faces (triangles)
    for y in range(h-1):
        for x in range(w-1):
            # Chỉ số vertices
            i1 = y * w + x
            i2 = y * w + (x + 1)
            i3 = (y + 1) * w + x
            i4 = (y + 1) * w + (x + 1)
            
            # Hai triangles cho mỗi quad
            faces.append([i1, i2, i3])
            faces.append([i2, i4, i3])
    
    return np.array(vertices), np.array(colors), np.array(faces), (h, w)

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
    """Chiếu 3D lên 2D với perspective"""
    projected = []
    z_values = []
    
    for vertex in vertices_3d:
        x, y, z = vertex
        z_cam = z + distance
        if z_cam > 0.1:  # Tránh chia cho 0
            px = x / z_cam
            py = y / z_cam
        else:
            px, py = 0, 0
        projected.append([px, py])
        z_values.append(z_cam)
    
    return np.array(projected), np.array(z_values)

def render_3d_mesh_advanced(vertices_3d, colors, faces, projected_2d, z_values, image_size=600):
    """Render mesh 3D với depth sorting và wireframe"""
    img = Image.new('RGB', (image_size, image_size), (20, 25, 35))
    draw = ImageDraw.Draw(img)
    
    # Scale tọa độ 2D
    proj_scaled = projected_2d.copy()
    proj_scaled[:, 0] = (proj_scaled[:, 0] + 1) * image_size / 2
    proj_scaled[:, 1] = (proj_scaled[:, 1] + 1) * image_size / 2
    
    # Sort faces theo độ sâu (z-buffer đơn giản)
    face_depths = []
    for face in faces:
        avg_z = np.mean([z_values[i] for i in face])
        face_depths.append(avg_z)
    
    # Sắp xếp faces theo độ sâu (xa nhất trước)
    sorted_indices = np.argsort(face_depths)[::-1]
    
    # Vẽ faces
    for idx in sorted_indices:
        face = faces[idx]
        if len(face) >= 3:
            # Lấy tọa độ của 3 vertices đầu
            points = []
            face_colors = []
            valid = True
            
            for i in range(3):
                vertex_idx = face[i]
                if vertex_idx < len(proj_scaled):
                    x, y = proj_scaled[vertex_idx]
                    if 0 <= x < image_size and 0 <= y < image_size:
                        points.append((int(x), int(y)))
                        face_colors.append(colors[vertex_idx])
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            
            if valid and len(points) == 3:
                # Tính màu trung bình
                avg_color = np.mean(face_colors, axis=0)
                # Áp dụng shading đơn giản
                brightness = 0.3 + 0.7 * (face_depths[idx] / max(face_depths) if max(face_depths) > 0 else 1)
                final_color = tuple((avg_color * brightness * 255).astype(int))
                
                # Vẽ triangle
                try:
                    draw.polygon(points, fill=final_color, outline=(100, 120, 150))
                except:
                    pass
    
    # Vẽ wireframe
    for face in faces:
        if len(face) >= 3:
            points = []
            for i in range(3):
                vertex_idx = face[i]
                if vertex_idx < len(proj_scaled):
                    x, y = proj_scaled[vertex_idx]
                    if 0 <= x < image_size and 0 <= y < image_size:
                        points.append((int(x), int(y)))
            
            if len(points) == 3:
                try:
                    # Vẽ wireframe
                    for i in range(3):
                        start = points[i]
                        end = points[(i + 1) % 3]
                        draw.line([start, end], fill=(80, 100, 130), width=1)
                except:
                    pass
    
    return img

def generate_interactive_3d_html(vertices, colors, faces, mesh_size):
    """Tạo HTML với Three.js cho 3D interactive"""
    
    # Chuyển đổi dữ liệu sang format JSON
    vertices_js = vertices.tolist()
    colors_js = colors.tolist()
    faces_js = faces.tolist()
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive 3D Mesh</title>
        <style>
            body {{ margin: 0; overflow: hidden; background: #1a1a2e; }}
            #container {{ width: 100%; height: 500px; }}
            #controls {{ 
                position: absolute; 
                top: 10px; 
                left: 10px; 
                background: rgba(0,0,0,0.7); 
                color: white; 
                padding: 10px; 
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <div id="controls">
            <div>🖱️ Kéo chuột để xoay</div>
            <div>🔍 Cuộn chuột để zoom</div>
            <div>Vertices: {len(vertices)}</div>
            <div>Faces: {len(faces)}</div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            // Scene setup
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 500, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, 500);
            renderer.setClearColor(0x1a1a2e);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Create geometry
            const geometry = new THREE.BufferGeometry();
            
            // Vertices
            const vertices = {vertices_js};
            const verticesArray = new Float32Array(vertices.length * 3);
            for(let i = 0; i < vertices.length; i++) {{
                verticesArray[i * 3] = vertices[i][0];
                verticesArray[i * 3 + 1] = vertices[i][1]; 
                verticesArray[i * 3 + 2] = vertices[i][2];
            }}
            geometry.setAttribute('position', new THREE.BufferAttribute(verticesArray, 3));
            
            // Colors
            const colors = {colors_js};
            const colorsArray = new Float32Array(vertices.length * 3);
            for(let i = 0; i < colors.length; i++) {{
                colorsArray[i * 3] = colors[i][0];
                colorsArray[i * 3 + 1] = colors[i][1];
                colorsArray[i * 3 + 2] = colors[i][2];
            }}
            geometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
            
            // Faces
            const faces = {faces_js};
            const indices = [];
            for(let i = 0; i < faces.length; i++) {{
                indices.push(faces[i][0], faces[i][1], faces[i][2]);
            }}
            geometry.setIndex(indices);
            
            // Compute normals
            geometry.computeVertexNormals();
            
            // Materials
            const material = new THREE.MeshPhongMaterial({{ 
                vertexColors: true,
                side: THREE.DoubleSide,
                shininess: 30
            }});
            
            const wireframeMaterial = new THREE.MeshBasicMaterial({{ 
                color: 0x4a90e2,
                wireframe: true,
                transparent: true,
                opacity: 0.3
            }});
            
            // Meshes
            const mesh = new THREE.Mesh(geometry, material);
            const wireframe = new THREE.Mesh(geometry, wireframeMaterial);
            
            scene.add(mesh);
            scene.add(wireframe);
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Camera position
            camera.position.z = 3;
            
            // Mouse controls
            let mouseDown = false;
            let mouseX = 0;
            let mouseY = 0;
            let targetRotationX = 0;
            let targetRotationY = 0;
            let currentRotationX = 0;
            let currentRotationY = 0;
            
            renderer.domElement.addEventListener('mousedown', onMouseDown, false);
            renderer.domElement.addEventListener('mousemove', onMouseMove, false);
            renderer.domElement.addEventListener('mouseup', onMouseUp, false);
            renderer.domElement.addEventListener('wheel', onWheel, false);
            
            function onMouseDown(event) {{
                mouseDown = true;
                mouseX = event.clientX;
                mouseY = event.clientY;
            }}
            
            function onMouseMove(event) {{
                if (mouseDown) {{
                    const deltaX = event.clientX - mouseX;
                    const deltaY = event.clientY - mouseY;
                    
                    targetRotationY += deltaX * 0.01;
                    targetRotationX += deltaY * 0.01;
                    
                    mouseX = event.clientX;
                    mouseY = event.clientY;
                }}
            }}
            
            function onMouseUp(event) {{
                mouseDown = false;
            }}
            
            function onWheel(event) {{
                camera.position.z += event.deltaY * 0.01;
                camera.position.z = Math.max(1, Math.min(10, camera.position.z));
            }}
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                
                // Smooth rotation
                currentRotationX += (targetRotationX - currentRotationX) * 0.1;
                currentRotationY += (targetRotationY - currentRotationY) * 0.1;
                
                mesh.rotation.x = currentRotationX;
                mesh.rotation.y = currentRotationY;
                wireframe.rotation.x = currentRotationX;
                wireframe.rotation.y = currentRotationY;
                
                renderer.render(scene, camera);
            }}
            
            animate();
            
            // Handle resize
            window.addEventListener('resize', function() {{
                camera.aspect = window.innerWidth / 500;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, 500);
            }});
        </script>
    </body>
    </html>
    """
    
    return html_code

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
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Chỉnh sửa 2D", "🌐 Hiệu ứng 3D", "🎮 3D Tương tác", "📊 Ma trận"])
    
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
    
    # TAB 2: 3D EFFECTS (Static)
    with tab2:
        st.subheader("🌐 Hiệu ứng 3D với Givens Rotation")
        
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rx = st.slider("🔄 Rotation X", -180, 180, 15, 15)
            depth_scale = st.slider("🏔️ Độ sâu", 10, 100, 30, 10)
        with col2:
            ry = st.slider("🔄 Rotation Y", -180, 180, -20, 15)
            resolution = st.slider("🔍 Độ phân giải", 20, 60, 40, 10)
        with col3:
            rz = st.slider("🔄 Rotation Z", -180, 180, 0, 15)
            render_mode = st.selectbox("🎨 Chế độ render", ["Solid", "Wireframe", "Both"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Tạo hiệu ứng 3D", type="primary"):
            with st.spinner("Đang xử lý..."):
                # Tạo mesh 3D
                vertices, colors, faces, mesh_size = create_3d_mesh(
                    original_image, depth_scale, resolution
                )
                
                # Áp dụng rotation
                rotated_vertices, rotation_matrix = apply_3d_rotation(vertices, rx, ry, rz)
                
                # Chiếu lên 2D
                projected, z_values = project_3d_to_2d(rotated_vertices)
                
                # Render
                result_3d = render_3d_mesh_advanced(
                    rotated_vertices, colors, faces, projected, z_values
                )
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Ảnh gốc:**")
                    st.image(original_image, use_column_width=True)
                
                with col2:
                    st.markdown("**Hiệu ứng 3D:**")
                    st.image(result_3d, use_column_width=True)
                
                # Download 3D
                buf_3d = io.BytesIO()
                result_3d.save(buf_3d, format='PNG')
                st.download_button(
                    "💾 Tải xuống 3D",
                    buf_3d.getvalue(),
                    f"3d_{uploaded_file.name}",
                    "image/png"
                )
                
                # Lưu vào session state cho tab interactive
                st.session_state['mesh_data'] = {
                    'vertices': vertices,
                    'colors': colors,
                    'faces': faces,
                    'mesh_size': mesh_size
                }
    
    # TAB 3: 3D INTERACTIVE
    with tab3:
        st.subheader("🎮 3D Tương tác - Kéo chuột để xoay")
        
        st.markdown('<div class="interactive-hint">', unsafe_allow_html=True)
        st.markdown("🖱️ **Hướng dẫn:** Kéo chuột để xoay, cuộn chuột để zoom in/out")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Kiểm tra xem đã có mesh data chưa
        if 'mesh_data' not in st.session_state:
            st.info("📋 Hãy tạo mesh 3D ở tab '🌐 Hiệu ứng 3D' trước!")
            
            # Tạo mesh mặc định với độ phân giải thấp
            if st.button("🔧 Tạo mesh mặc định"):
                with st.spinner("Đang tạo mesh..."):
                    vertices, colors, faces, mesh_size = create_3d_mesh(
                        original_image, 30, 30
                    )
                    st.session_state['mesh_data'] = {
                        'vertices': vertices,
                        'colors': colors,
                        'faces': faces,
                        'mesh_size': mesh_size
                    }
                    st.rerun()
        else:
            # Lấy mesh data
            mesh_data = st.session_state['mesh_data']
            
            # Tạo HTML interactive
            interactive_html = generate_interactive_3d_html(
                mesh_data['vertices'],
                mesh_data['colors'], 
                mesh_data['faces'],
                mesh_data['mesh_size']
            )
            
            # Hiển thị 3D interactive
            st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
            st.components.v1.html(interactive_html, height=500)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Thông tin mesh
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Vertices", len(mesh_data['vertices']))
            with col2:
                st.metric("🔺 Faces", len(mesh_data['faces']))
            with col3:
                st.metric("📐 Resolution", f"{mesh_data['mesh_size'][0]}x{mesh_data['mesh_size'][1]}")
            with col4:
                if st.button("🔄 Reset mesh"):
                    del st.session_state['mesh_data']
                    st.rerun()
    
    # TAB 4: MATRICES
    with tab4:
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="matrix-box">', unsafe_allow_html=True)
            st.code(f"""
G({demo_angle}°) = [{demo_matrix[0,0]:7.4f}  {demo_matrix[0,1]:7.4f}]
           [{demo_matrix[1,0]:7.4f}  {demo_matrix[1,1]:7.4f}]

Determinant: {np.linalg.det(demo_matrix):.6f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Visualize rotation effect
            st.markdown("**Hiệu ứng xoay:**")
            # Tạo vector test
            test_vector = np.array([1, 0])
            rotated_vector = np.dot(demo_matrix, test_vector)
            
            # Plot đơn giản
            fig_data = f"""
Vector gốc: ({test_vector[0]:.2f}, {test_vector[1]:.2f})
Vector xoay: ({rotated_vector[0]:.2f}, {rotated_vector[1]:.2f})
Góc xoay: {demo_angle}°
            """
            st.text(fig_data)

else:
    # Landing page
    st.markdown("""
    ## 👋 Chào mừng đến với Givens Rotation Image Editor!
    
    ### 🎯 Tính năng mới:
    
    **🖼️ Chỉnh sửa 2D:**
    • Givens Rotation với góc tùy chỉnh
    • Điều chỉnh độ sáng, tương phản
    • Làm mờ Gaussian
    
    **🌐 Hiệu ứng 3D:**
    • Tạo mesh 3D từ ảnh
    • Rotation 3 trục với Givens
    • Depth sorting và wireframe
    
    **🎮 3D Tương tác:** ⭐ **MỚI**
    • Kéo chuột để xoay model 3D
    • Zoom in/out bằng scroll
    • Render real-time với Three.js
    • Lighting và shading nâng cao
    
    **📊 Ma trận:**
    • Hiển thị ma trận transformation
    • Demo tương tác
    • Công thức toán học
    
    ### 🚀 Cách sử dụng:
    1. **Upload ảnh** (PNG, JPG, JPEG)
    2. **Tab 2D:** Chỉnh sửa cơ bản với Givens rotation
    3. **Tab 3D:** Tạo hiệu ứng 3D static
    4. **Tab Tương tác:** Khám phá model 3D với chuột! 🖱️
    5. **Tab Ma trận:** Tìm hiểu lý thuyết
    
    ### 🎮 Điều khiển 3D:
    • **Kéo chuột:** Xoay model theo mọi hướng
    • **Scroll chuột:** Zoom in/out
    • **Real-time rendering** với WebGL
    
    **📁 Hãy upload ảnh để bắt đầu trải nghiệm 3D tương tác!**
    """)
    
    # Demo preview
    st.markdown("### 🎨 Preview tính năng:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **2D Rotation**
        - Ma trận Givens 2x2
        - Xoay ảnh mượt mà
        - Giữ nguyên chất lượng
        """)
    
    with col2:
        st.markdown("""
        **3D Mesh**
        - Depth từ brightness
        - Wireframe + Solid
        - Multiple rotations
        """)
    
    with col3:
        st.markdown("""
        **Interactive 3D** ⭐
        - Mouse controls
        - Real-time WebGL
        - Smooth animations
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🎨 <strong>Givens Rotation Image Editor v2.0</strong> - Now with Interactive 3D!<br>
    <small>Powered by Three.js & WebGL</small>
</div>
""", unsafe_allow_html=True)
