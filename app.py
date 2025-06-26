# ... (phần đầu code giữ nguyên như bạn cung cấp)

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
