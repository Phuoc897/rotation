import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st
import plotly.graph_objects as go

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        # Tạo lưới tọa độ
        y, x = np.meshgrid(range(self.height), range(self.width), indexing='ij')
        z = np.zeros_like(x)
        self.pixels = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    def givens_matrix(self, i, j, theta):
        if i == j or i < 0 or j < 0 or i > 2 or j > 2:
            raise ValueError("Invalid Givens indices")
        if i > j:
            i, j = j, i
        G = np.eye(3)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = s, -s
        return G

    def centering_image(self, pixels):
        center = np.array([self.width/2, self.height/2, 0])
        return pixels - center

    def rotate_image_2d(self, angle=0):
        h, w = self.image.shape[:2]
        rad = np.deg2rad(angle)
        cos, sin = np.abs(np.cos(rad)), np.abs(np.sin(rad))
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(self.image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

    def givens_rotation_3d(self, a, t, g):
        R_x = self.givens_matrix(1, 2, a)
        R_y = self.givens_matrix(0, 2, t)
        R_z = self.givens_matrix(0, 1, g)
        pts = self.centering_image(self.pixels)
        return pts @ R_z @ R_y @ R_x

    def initialize_projection(self, max_angle):
        d = max(self.height, self.width)
        self.focal_length = d * 1.5 * (1 + max_angle/90)
        cx, cy = self.width/2, self.height/2
        self.camera_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def project_points(self, pts3d):
        cam = pts3d.copy()
        cam[:, 2] += self.focal_length * 2
        valid_mask = cam[:, 2] > 0.1
        cam = cam[valid_mask]
        if len(cam) == 0:
            return np.array([]), valid_mask
        x_proj = cam[:, 0] / cam[:, 2]
        y_proj = cam[:, 1] / cam[:, 2]
        u = self.focal_length * x_proj + self.camera_matrix[0, 2]
        v = self.focal_length * y_proj + self.camera_matrix[1, 2]
        pts2d = np.column_stack((u, v)).astype(int)
        if len(pts2d) > 0:
            min_u, min_v = pts2d.min(axis=0)
            if min_u < 0:
                pts2d[:, 0] -= min_u
            if min_v < 0:
                pts2d[:, 1] -= min_v
        return pts2d, valid_mask

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0):
        a, t, g = np.deg2rad([alpha, theta, gamma])
        pts3d = self.givens_rotation_3d(a, t, g)
        max_angle = max(abs(alpha), abs(theta), abs(gamma))
        self.initialize_projection(max_angle)
        pts2d, valid_mask = self.project_points(pts3d)
        if len(pts2d) == 0:
            # Trả về ảnh transparent nếu không có điểm
            return np.zeros((1, 1, 4), dtype=np.uint8)
        H, W = pts2d[:, 1].max() + 1, pts2d[:, 0].max() + 1
        H, W = max(H, 1), max(W, 1)
        # Tạo canvas RGBA, khởi tạo alpha = 0 (transparent)
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        # Gán các pixel RGB và thiết lập alpha
        valid_pixels = self.pixels[valid_mask]
        for idx in range(len(valid_pixels)):
            x, y = int(valid_pixels[idx][0]), int(valid_pixels[idx][1])
            u, v = pts2d[idx]
            if 0 <= x < self.width and 0 <= y < self.height and 0 <= u < W and 0 <= v < H:
                pixel = self.image[y, x]
                canvas[v, u, :3] = pixel
                canvas[v, u, 3] = 255  # Opaque
        return canvas

# --------------------- Giao diện Streamlit ---------------------
st.set_page_config(page_title="Xoay ảnh 2D & 3D", layout="wide")
st.title("🎨 Ứng dụng Xoay ảnh và Chỉnh sáng")

sidebar = st.sidebar
mode = sidebar.radio("Chế độ xoay", ["2D", "3D"])
brightness = sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)

if mode == "2D":
    angle = sidebar.slider("Góc xoay (độ)", -180, 180, 0)
else:
    alpha = sidebar.slider("Alpha (X)", -45, 45, 0)
    theta = sidebar.slider("Theta (Y)", -45, 45, 0)
    gamma = sidebar.slider("Gamma (Z)", -45, 45, 0)

uploaded = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("Định dạng file không được hỗ trợ hoặc file bị lỗi.")
    else:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.subheader("Ảnh gốc")
        st.image(img, width=300)
        rot = ImageRotation(img)
        if mode == "2D":
            if sidebar.button("Xoay 2D") or angle != 0:
                out = rot.rotate_image_2d(angle)
                out = cv2.convertScaleAbs(out, alpha=brightness, beta=0)
                st.subheader(f"Kết quả 2D: Góc={angle}°, Độ sáng={brightness}")
                st.image(out, width=300)
        else:
            if sidebar.button("Xoay 3D") or (alpha or theta or gamma):
                with st.spinner("Đang xoay ảnh 3D..."):
                    out = rot.rotate_image_3d(alpha, theta, gamma)
                    # Áp dụng độ sáng chỉ cho kênh RGB
                    rgba = out.astype(np.float32)
                    rgba[..., :3] = np.clip(rgba[..., :3] * brightness, 0, 255)
                    out = rgba.astype(np.uint8)
                    st.subheader(f"Kết quả 3D (no background)")
                    st.image(out, width=400)
                    if st.checkbox("Interactive (Plotly) - chỉ di chuyển vùng ảnh"):
                        fig = go.Figure()
                        fig.add_layout_image(
                            dict(
                                source=out,
                                x=0, y=0,
                                sizex=out.shape[1], sizey=out.shape[0],
                                xref="x", yref="y",
                                layer="above"
                            )
                        )
                        fig.update_xaxes(visible=False, range=[0, out.shape[1]])
                        fig.update_yaxes(visible=False, range=[out.shape[0], 0])
                        fig.update_layout(
                            width=400, height=400,
                            margin=dict(l=0, r=0, t=0, b=0),
                            dragmode='pan'
                        )
                        st.plotly_chart(fig, use_container_width=False)
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

# Sample images
gdown_ui = st.expander("Tải ảnh mẫu")
with gdown_ui:
    if st.button("Tải ảnh mẫu qua Google Drive"):
        samples = [
            ("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE", "sample1.jpg"),
            ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ", "sample2.jpg")
        ]
        for gid, fname in samples:
            url = f"https://drive.google.com/uc?id={gid}"
            gdown.download(url, fname, quiet=True)
        st.success("Tải xong ảnh mẫu!")

st.markdown("---")
st.markdown("**Hướng dẫn sử dụng:**")
st.markdown("- **2D**: Xoay ảnh theo góc đơn giản")
st.markdown("- **3D**: Xoay ảnh với nền trong suốt và chỉ vùng ảnh có thể kéo")
