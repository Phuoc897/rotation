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

def givens_3d(angle, axis):
    """Tạo ma trận rotation 3D cho trục x, y, z"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis == 'y':
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis == 'z':
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

def create_3d_mesh(image, depth_scale=20, resolution=50):
    """Tạo mesh 3D cải tiến từ ảnh với chất lượng cao hơn"""
    # Resize ảnh với phương pháp LANCZOS để giữ chất lượng
    img_small = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    img_array = np.array(img_small)
    
    # Tạo depth map từ brightness với smooth filter
    if len(img_array.shape) == 3:
        # Sử dụng weighted average cho RGB
        depth = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    else:
        depth = img_array.copy()
    
    # Áp dụng Gaussian blur để smooth depth map
    depth = gaussian_blur(depth, sigma=0.😎
    
    # Normalize depth với scaling tốt hơn
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth_final = depth_normalized * (depth_scale / 100.0)
    
    # Tạo vertices với spacing đều
    vertices = []
    colors = []
    faces = []
    
    h, w = depth_final.shape
    
    # Scale factor cho tọa độ x, y
    scale_x = 2.0 / (w - 1)
    scale_y = 2.0 / (h - 1)
    
    for y in range(h):
        for x in range(w):
            # Tọa độ 3D với center tại (0,0)
            vertex_x = (x * scale_x) - 1.0
            vertex_y = 1.0 - (y * scale_y)  # Flip Y để match image orientation
            vertex_z = depth_final[y, x]
            
            vertices.append([vertex_x, vertex_y, vertex_z])
            
            # Màu từ ảnh gốc với interpolation
            if len(img_array.shape) == 3:
                colors.append(img_array[y, x] / 255.0)
            else:
                gray = img_array[y, x] / 255.0
                colors.append([gray, gray, gray])
    
    # Tạo faces với kiểm tra bounds
    for y in range(h-1):
        for x in range(w-1):
            # Chỉ số vertices
            i1 = y * w + x
            i2 = y * w + (x + 1)
            i3 = (y + 1) * w + x
            i4 = (y + 1) * w + (x + 1)
            
            # Kiểm tra bounds
            if i4 < len(vertices):
                # Hai triangles cho mỗi quad với winding order đúng
                faces.append([i1, i2, i3])  # Triangle 1
                faces.append([i2, i4, i3])  # Triangle 2
    
    return np.array(vertices), np.array(colors), np.array(faces), (h, w)

def gaussian_blur(image, sigma=1.0):
    """Áp dụng Gaussian blur đơn giản"""
    # Tạo kernel Gaussian
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    
    kernel = kernel / np.sum(kernel)
    
    # Áp dụng convolution
    padded = np.pad(image, center, mode='edge')
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return result

def apply_3d_rotation(vertices, rx, ry, rz):
    """Áp dụng rotation 3D với ma trận kết hợp"""
    # Chuyển sang radian
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)
    
    # Tạo ma trận rotation cho từng trục
    Rx = givens_3d(rx_rad, 'x')
    Ry = givens_3d(ry_rad, 'y')
    Rz = givens_3d(rz_rad, 'z')
    
    # Kết hợp rotation (Z * Y * X order)
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # Áp dụng rotation cho tất cả vertices
    rotated = np.dot(vertices, R.T)
    
    return rotated, R

def project_3d_to_2d(vertices_3d, distance=4.0, fov=60):
    """Chiếu 3D lên 2D với perspective projection cải tiến"""
    projected = []
    z_values = []
    
    # Tính toán projection parameters
    fov_rad = np.radians(fov)
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    for vertex in vertices_3d:
        x, y, z = vertex
        
        # Dịch chuyển z để camera ở khoảng cách distance
        z_cam = z + distance
        
        # Perspective projection với clipping
        if z_cam > 0.01:  # Near clipping plane
            # Sử dụng proper perspective projection
            px = (x * f) / z_cam
            py = (y * f) / z_cam
        else:
            # Clip vertices behind camera
            px, py = 0, 0
            z_cam = 0.01
            
        projected.append([px, py])
        z_values.append(z_cam)
    
    return np.array(projected), np.array(z_values)

def calculate_face_normal(v1, v2, v3):
    """Tính normal vector của face"""
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal)
    if length > 0:
        return normal / length
    return np.array([0, 0, 1])

def render_3d_mesh_advanced(vertices_3d, colors, faces, projected_2d, z_values, image_size=800):
    """Render mesh 3D với improved shading và anti-aliasing"""
    
    # Tạo image với background gradient
    img = Image.new('RGBA', (image_size, image_size), (15, 20, 30, 255))
    draw = ImageDraw.Draw(img)
    
    # Scale và center tọa độ 2D
    proj_scaled = projected_2d.copy()
    
    # Tìm bounds của projection
    valid_mask = z_values > 0.01
    if np.any(valid_mask):
        valid_proj = projected_2d[valid_mask]
        min_x, max_x = np.min(valid_proj[:, 0]), np.max(valid_proj[:, 0])
        min_y, max_y = np.min(valid_proj[:, 1]), np.max(valid_proj[:, 1])
        
        # Scale để fit trong image với margin
        margin = 0.1
        scale_factor = min((1 - 2*margin) / (max_x - min_x), (1 - 2*margin) / (max_y - min_y))
        scale_factor = min(scale_factor, 1.0)  # Không scale up quá 1.0
        
        # Center và scale
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        proj_scaled[:, 0] = (proj_scaled[:, 0] - center_x) * scale_factor
        proj_scaled[:, 1] = (proj_scaled[:, 1] - center_y) * scale_factor
    
    # Convert to screen coordinates
    proj_scaled[:, 0] = (proj_scaled[:, 0] + 1) * image_size / 2
    proj_scaled[:, 1] = (1 - proj_scaled[:, 1]) * image_size / 2  # Flip Y for screen coords
    
    # Sort faces theo depth (z-buffer)
    face_depths = []
    face_centers = []
    valid_faces = []
    
    for i, face in enumerate(faces):
        if len(face) >= 3 and all(idx < len(z_values) for idx in face[:3]):
            # Tính depth trung bình
            face_z_values = [z_values[idx] for idx in face[:3]]
            avg_z = np.mean(face_z_values)
            face_depths.append(avg_z)
            
            # Tính center của face
            face_verts = [vertices_3d[idx] for idx in face[:3]]
            center = np.mean(face_verts, axis=0)
            face_centers.append(center)
            valid_faces.append(i)
    
    if not face_depths:
        return img
        
    # Sort faces (xa nhất trước - painter's algorithm)
    sorted_indices = np.argsort(face_depths)[::-1]
    
    # Light direction
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Render faces
    for sort_idx in sorted_indices:
        face_idx = valid_faces[sort_idx]
        face = faces[face_idx]
        
        if len(face) >= 3:
            # Get vertices for triangle
            points = []
            face_colors = []
            face_verts_3d = []
            valid = True
            
            for i in range(3):
                vertex_idx = face[i]
                if vertex_idx < len(proj_scaled) and vertex_idx < len(colors):
                    x, y = proj_scaled[vertex_idx]
                    # Check bounds with some tolerance
                    if -10 <= x <= image_size + 10 and -10 <= y <= image_size + 10:
                        points.append((max(0, min(image_size-1, int(x))), 
                                     max(0, min(image_size-1, int(y)))))
                        face_colors.append(colors[vertex_idx])
                        face_verts_3d.append(vertices_3d[vertex_idx])
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            
            if valid and len(points) == 3 and len(face_verts_3d) == 3:
                # Calculate lighting
                normal = calculate_face_normal(
                    face_verts_3d[0], face_verts_3d[1], face_verts_3d[2]
                )
                
                # Lambertian shading
                light_intensity = max(0.2, np.dot(normal, light_dir))
                
                # Calculate average color
                avg_color = np.mean(face_colors, axis=0)
                
                # Apply lighting
                final_color = avg_color * light_intensity
                final_color = np.clip(final_color * 255, 0, 255).astype(int)
                
                # Convert to tuple
                fill_color = tuple(final_color)
                outline_color = tuple((final_color * 0.😎.astype(int))
                
                # Render triangle
                try:
                    # Check if triangle has area
                    p1, p2, p3 = points
                    area = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    if area > 1:  # Only render if triangle has significant area
                        draw.polygon(points, fill=fill_color, outline=outline_color, width=1)
                except Exception as e:
                    continue
    
    return img

def generate_interactive_3d_html(vertices, colors, faces, mesh_size):
    """Tạo HTML với Three.js cải tiến cho 3D interactive"""
    
    # Chuyển đổi dữ liệu sang format JSON
    vertices_js = vertices.tolist()
    colors_js = colors.tolist()
    faces_js = faces.tolist()
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive 3D Mesh - Enhanced</title>
        <style>
            body {{ 
                margin: 0; 
                overflow: hidden; 
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            #container {{ 
                width: 100%; 
                height: 100vh; 
                position: relative;
            }}
            #controls {{ 
                position: absolute; 
                top: 20px; 
                left: 20px; 
                background: rgba(0,0,0,0.8); 
                color: white; 
                padding: 15px; 
                border-radius: 8px;
                font-size: 14px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }}
            #stats {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .control-item {{
                margin: 5px 0;
                display: flex;
                align-items: center;
            }}
            .emoji {{
                margin-right: 8px;
                font-size: 16px;
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <div id="controls">
            <div class="control-item">
                <span class="emoji">🖱️</span>
                <span>Drag to rotate</span>
            </div>
            <div class="control-item">
                <span class="emoji">🔍</span>
                <span>Scroll to zoom</span>
            </div>
            <div class="control-item">
                <span class="emoji">📐</span>
                <span>Right-click to pan</span>
            </div>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                <div>Resolution: {mesh_size[0]}x{mesh_size[1]}</div>
            </div>
        </div>
        <div id="stats">
            <div>Vertices: {len(vertices):,}</div>
            <div>Faces: {len(faces):,}</div>
            <div>FPS: <span id="fps">--</span></div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
            // Performance monitoring
            let frameCount = 0;
            let lastTime = Date.now();
            
            // Scene setup
            const scene = new THREE.Scene();
            scene.fog = new THREE.Fog(0x0f0f23, 5, 15);
            
            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
            
            const renderer = new THREE.WebGLRenderer({{ 
                antialias: true,
                alpha: true,
                powerPreference: "high-performance"
            }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setClearColor(0x0f0f23, 1);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Create geometry with proper error handling
            const geometry = new THREE.BufferGeometry();
            
            try {{
                // Vertices
                const vertices = {vertices_js};
                if (vertices.length === 0) throw new Error("No vertices data");
                
                const verticesArray = new Float32Array(vertices.length * 3);
                for(let i = 0; i < vertices.length; i++) {{
                    if (vertices[i] && vertices[i].length >= 3) {{
                        verticesArray[i * 3] = vertices[i][0] || 0;
                        verticesArray[i * 3 + 1] = vertices[i][1] || 0; 
                        verticesArray[i * 3 + 2] = vertices[i][2] || 0;
                    }}
                }}
                geometry.setAttribute('position', new THREE.BufferAttribute(verticesArray, 3));
                
                // Colors
                const colors = {colors_js};
                const colorsArray = new Float32Array(vertices.length * 3);
                for(let i = 0; i < colors.length && i < vertices.length; i++) {{
                    if (colors[i] && colors[i].length >= 3) {{
                        colorsArray[i * 3] = Math.max(0, Math.min(1, colors[i][0] || 0));
                        colorsArray[i * 3 + 1] = Math.max(0, Math.min(1, colors[i][1] || 0));
                        colorsArray[i * 3 + 2] = Math.max(0, Math.min(1, colors[i][2] || 0));
                    }} else {{
                        // Default gray color
                        colorsArray[i * 3] = 0.5;
                        colorsArray[i * 3 + 1] = 0.5;
                        colorsArray[i * 3 + 2] = 0.5;
                    }}
                }}
                geometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
                
                // Faces/Indices
                const faces = {faces_js};
                const indices = [];
                for(let i = 0; i < faces.length; i++) {{
                    if (faces[i] && faces[i].length >= 3) {{
                        const v1 = faces[i][0];
                        const v2 = faces[i][1];
                        const v3 = faces[i][2];
                        
                        // Validate indices
                        if (v1 >= 0 && v2 >= 0 && v3 >= 0 && 
                            v1 < vertices.length && v2 < vertices.length && v3 < vertices.length) {{
                            indices.push(v1, v2, v3);
                        }}
                    }}
                }}
                
                if (indices.length === 0) throw new Error("No valid faces");
                geometry.setIndex(indices);
                
                // Compute normals for proper lighting
                geometry.computeVertexNormals();
                
            }} catch(error) {{
                console.error("Geometry creation error:", error);
                // Create fallback geometry
                const fallbackGeometry = new THREE.BoxGeometry(1, 1, 1);
                geometry.copy(fallbackGeometry);
            }}
            
            // Materials
            const material = new THREE.MeshPhongMaterial({{ 
                vertexColors: true,
                side: THREE.DoubleSide,
                shininess: 60,
                specular: 0x222222,
                transparent: false,
                flatShading: false
            }});
            
            const wireframeMaterial = new THREE.MeshBasicMaterial({{ 
                color: 0x4a90e2,
                wireframe: true,
                transparent: true,
                opacity: 0.15,
                depthWrite: false
            }});
            
            // Meshes
            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            
            const wireframe = new THREE.Mesh(geometry, wireframeMaterial);
            wireframe.renderOrder = 1;
            
            scene.add(mesh);
            scene.add(wireframe);
            
            // Enhanced lighting setup
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            scene.add(ambientLight);
            
            const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
            mainLight.position.set(2, 2, 2);
            mainLight.castShadow = true;
            mainLight.shadow.mapSize.width = 1024;
            mainLight.shadow.mapSize.height = 1024;
            scene.add(mainLight);
            
            const fillLight = new THREE.DirectionalLight(0x4466aa, 0.3);
            fillLight.position.set(-1, -1, 1);
            scene.add(fillLight);
            
            // Camera positioning
            camera.position.set(0, 0, 4);
            camera.lookAt(0, 0, 0);
            
            // Enhanced controls
            let isDragging = false;
            let isPanning = false;
            let previousMousePosition = {{ x: 0, y: 0 }};
            let targetRotation = {{ x: 0, y: 0 }};
            let currentRotation = {{ x: 0, y: 0 }};
            let targetPosition = {{ x: 0, y: 0, z: 4 }};
            let currentPosition = {{ x: 0, y: 0, z: 4 }};
            
            // Mouse event handlers
            renderer.domElement.addEventListener('mousedown', onMouseDown);
            renderer.domElement.addEventListener('mousemove', onMouseMove);
            renderer.domElement.addEventListener('mouseup', onMouseUp);
            renderer.domElement.addEventListener('wheel', onWheel);
            renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
            
            function onMouseDown(event) {{
                if (event.button === 0) {{ // Left mouse button
                    isDragging = true;
                }} else if (event.button === 2) {{ // Right mouse button
                    isPanning = true;
                }}
                previousMousePosition.x = event.clientX;
                previousMousePosition.y = event.clientY;
            }}
            
            function onMouseMove(event) {{
                const deltaX = event.clientX - previousMousePosition.x;
                const deltaY = event.clientY - previousMousePosition.y;
                
                if (isDragging) {{
                    targetRotation.y += deltaX * 0.005;
                    targetRotation.x += deltaY * 0.005;
                    targetRotation.x = Math.max(-Math.PI/2, Math.min(Math.PI/2, targetRotation.x));
                }} else if (isPanning) {{
                    const panSpeed = 0.002 * targetPosition.z;
                    targetPosition.x -= deltaX * panSpeed;
                    targetPosition.y += deltaY * panSpeed;
                }}
                
                previousMousePosition.x = event.clientX;
                previousMousePosition.y = event.clientY;
            }}
            
            function onMouseUp(event) {{
                isDragging = false;
                isPanning = false;
            }}
            
            function onWheel(event) {{
                const zoomSpeed = 0.1;
                targetPosition.z += event.deltaY * zoomSpeed * 0.01;
                targetPosition.z = Math.max(1, Math.min(20, targetPosition.z));
                event.preventDefault();
            }}
            
            // Animation loop with performance monitoring
            function animate() {{
                requestAnimationFrame(animate);
                
                // Update FPS counter
                frameCount++;
                const currentTime = Date.now();
                if (currentTime - lastTime >= 1000) {{
                    document.getElementById('fps').textContent = frameCount;
                    frameCount = 0;
                    lastTime = currentTime;
                }}
                
                // Smooth interpolation
                const lerpFactor = 0.1;
                
                currentRotation.x += (targetRotation.x - currentRotation.x) * lerpFactor;
                currentRotation.y += (targetRotation.y - currentRotation.y) * lerpFactor;
                
                currentPosition.x += (targetPosition.x - currentPosition.x) * lerpFactor;
                currentPosition.y += (targetPosition.y - currentPosition.y) * lerpFactor;
                currentPosition.z += (targetPosition.z - currentPosition.z) * lerpFactor;
                
                // Apply transformations
                mesh.rotation.x = currentRotation.x;
                mesh.rotation.y = currentRotation.y;
                wireframe.rotation.x = currentRotation.x;
                wireframe.rotation.y = currentRotation.y;
                
                camera.position.set(currentPosition.x, currentPosition.y, currentPosition.z);
                camera.lookAt(0, 0, 0);
                
                renderer.render(scene, camera);
            }}
            
            // Handle window resize
            function handleResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}
            
            window.addEventListener('resize', handleResize);
            
            // Start animation
            animate();
            
            // Add keyboard controls
            document.addEventListener('keydown', function(event) {{
                switch(event.code) {{
                    case 'KeyR':
                        // Reset view
                        targetRotation = {{ x: 0, y: 0 }};
                        targetPosition = {{ x: 0, y: 0, z: 4 }};
                        break;
                    case 'KeyW':
                        // Toggle wireframe
                        wireframe.visible = !wireframe.visible;
                        break;
                }}
            }});
            
        </script>
    </body>
    </html>
    """
    
    return html_code

# Hàm chính để test
def process_image_to_3d(image_path, rotation_x=0, rotation_y=0, rotation_z=0):
    """Xử lý ảnh thành 3D mesh với các cải tiến"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Tạo mesh 3D
    vertices, colors, faces, mesh_size = create_3d_mesh(image, depth_scale=15, resolution=60)
    
    # Áp dụng rotation
    if rotation_x != 0 or rotation_y != 0 or rotation_z != 0:
        vertices, rotation_matrix = apply_3d_rotation(vertices, rotation_x, rotation_y, rotation_z)
    
    # Project 3D to 2D
    projected, z_values = project_3d_to_2d(vertices, distance=4.5, fov=50)
    
    # Render 2D image
    rendered_img = render_3d_mesh_advanced(vertices, colors, faces, projected, z_values)
    
    # Tạo HTML interactive
    html_content = generate_interactive_3d_html(vertices, colors, faces, mesh_size)
    
    return rendered_img, html_content

# Example usage:
# rendered_image, html_code = process_image_to_3d("your_image.jpg", 
#                                                rotation_x=15, 
#                                                rotation_y=25, 
#                                                rotation_z=0)

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
