{rotation_matrix[1,0]:7.4f}   {rotation_matrix[1,1]:7.4f} ]

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
