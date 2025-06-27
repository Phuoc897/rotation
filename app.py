import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st
import plotly.graph_objects as go
import gc

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        y, x = np.meshgrid(range(self.height), range(self.width), indexing='ij')
        z = np.zeros_like(x)
        self.pixels = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(np.float32).T

    def givens_matrix(self, i, j, theta):
        if i == j or i < 0 or j < 0 or i > 2 or j > 2:
            raise ValueError("Invalid Givens indices")
        if i > j:
            i, j = j, i
        G = np.eye(3, dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = s, -s
        return G

    def centering_image(self, pixels):
        center = np.array([self.width/2, self.height/2, 0], dtype=np.float32)
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
            return np.ones_like(self.image) * 255
        H, W = pts2d[:, 1].max() + 1, pts2d[:, 0].max() + 1
        H, W = max(H, 1), max(W, 1)
        canvas = np.ones((H, W, self.image.shape[2]), dtype=self.image.dtype) * 255 if self.image.ndim == 3 else np.ones((H, W), dtype=self.image.dtype) * 255
        valid_pixels = self.pixels[valid_mask]
        return assign_pixels_nb(valid_pixels, pts2d, self.image, canvas)

@nb.njit
def assign_pixels_nb(pixels, pts2d, img, out):
    H, W = out.shape[:2]
    for i in range(len(pixels)):
        x, y = int(pixels[i][0]), int(pixels[i][1])
        u, v = pts2d[i]
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and 0 <= u < W and 0 <= v < H:
            if img.ndim == 3:
                for c in range(img.shape[2]):
                    out[v, u, c] = img[y, x, c]
            else:
                out[v, u] = img[y, x]
    return out

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Xoay ảnh 2D & 3D", layout="wide")
st.title("🎨 Ứng dụng Xoay ảnh và Chỉnh sáng")

sidebar = st.sidebar
che_do = sidebar.radio("Chế độ xoay", ["2D", "3D"])
do_sang = sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)

if che_do == "2D":
    goc = sidebar.slider("Góc xoay (độ)", -180, 180, 0)
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

        max_dim = 800
        h, w = img.shape[:2]
        scale = max_dim / max(h, w, max_dim)
        if scale < 1:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        st.subheader("Ảnh gốc")
        st.image(img, width=300)

        rotation = ImageRotation(img)

        if che_do == "2D":
            if sidebar.button("Xoay 2D"):
                try:
                    out = rotation.rotate_image_2d(goc)
                    out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                    st.subheader(f"Kết quả 2D: Góc={goc}°, Độ sáng={do_sang}")
                    st.image(out, width=300)
                except Exception as e:
                    st.error(f"Lỗi xoay 2D: {str(e)}")
        else:
            if sidebar.button("Xoay 3D"):
                try:
                    with st.spinner("Đang xoay ảnh 3D..."):
                        out = rotation.rotate_image_3d(alpha, theta, gamma)
                        out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                        st.subheader(f"Kết quả 3D: α={alpha}°, θ={theta}°, γ={gamma}°, Độ sáng={do_sang}")
                        st.image(out, width=400)
                        if st.checkbox("Hiển thị interactive (Plotly)"):
                            fig = go.Figure(go.Image(z=out))
                            fig.update_layout(width=400, height=400, margin=dict(l=0, r=0, t=0, b=0), dragmode='pan', title="Ảnh 3D (có thể zoom/pan)")
                            fig.update_xaxes(visible=False)
                            fig.update_yaxes(visible=False)
                            st.plotly_chart(fig, use_container_width=False)
                except Exception as e:
                    st.error(f"Lỗi xoay 3D: {str(e)}")

        del rotation, out, img
        gc.collect()
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")
