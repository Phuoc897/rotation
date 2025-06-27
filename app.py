import numpy as np
import cv2
import numba as nb
import streamlit as st
import plotly.graph_objects as go
import gc

# -------------------- Helper --------------------
def resize_image(img, max_dim=512):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

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

# -------------------- Core --------------------
class ImageRotation:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]
        y, x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        self.pixels = np.vstack((x.ravel(), y.ravel(), np.zeros_like(x).ravel())).astype(np.float32).T

    def rotate_2d(self, angle):
        h, w = self.image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    def rotate_3d(self, alpha, theta, gamma):
        a, t, g = np.deg2rad([alpha, theta, gamma])
        R_x = self._givens(1, 2, a)
        R_y = self._givens(0, 2, t)
        R_z = self._givens(0, 1, g)
        center = np.array([self.width / 2, self.height / 2, 0], dtype=np.float32)
        pts = self.pixels - center
        rotated = pts @ R_z @ R_y @ R_x

        fl = max(self.height, self.width) * 1.2
        cam = rotated.copy()
        cam[:, 2] += fl
        mask = cam[:, 2] > 0.1
        cam = cam[mask]

        x = fl * cam[:, 0] / cam[:, 2] + self.width / 2
        y = fl * cam[:, 1] / cam[:, 2] + self.height / 2
        pts2d = np.stack((x, y), axis=1).astype(int)

        max_h = int(np.max(pts2d[:, 1]) + 1)
        max_w = int(np.max(pts2d[:, 0]) + 1)

        out = np.ones((max_h, max_w, self.image.shape[2]), dtype=np.uint8) * 255 if self.image.ndim == 3 else np.ones((max_h, max_w), dtype=np.uint8) * 255
        return assign_pixels_nb(self.pixels[mask], pts2d, self.image, out)

    def _givens(self, i, j, theta):
        G = np.eye(3, dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = s, -s
        return G

# -------------------- UI --------------------
st.set_page_config(page_title="Xoay ảnh nhẹ RAM", layout="wide")
st.title("🖼️ Xoay ảnh 2D & 3D (Tối ưu RAM)")

mode = st.sidebar.radio("Chế độ", ["2D", "3D"])
b = st.sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)

uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_image(img, 512)

        st.subheader("Ảnh gốc")
        st.image(img, width=300)

        rotator = ImageRotation(img)

        if mode == "2D":
            angle = st.sidebar.slider("Góc xoay", -180, 180, 0)
            if st.sidebar.button("Xoay 2D"):
                out = rotator.rotate_2d(angle)
                out = cv2.convertScaleAbs(out, alpha=b)
                st.image(out, caption=f"2D: {angle}°", width=300)
        else:
            alpha = st.sidebar.slider("X (pitch)", -45, 45, 0)
            theta = st.sidebar.slider("Y (yaw)", -45, 45, 0)
            gamma = st.sidebar.slider("Z (roll)", -45, 45, 0)
            if st.sidebar.button("Xoay 3D"):
                with st.spinner("Xoay ảnh 3D..."):
                    out = rotator.rotate_3d(alpha, theta, gamma)
                    out = cv2.convertScaleAbs(out, alpha=b)
                    st.image(out, caption=f"3D: X={alpha}, Y={theta}, Z={gamma}", width=350)

        del rotator, out, img
        gc.collect()
else:
    st.info("Vui lòng tải ảnh để bắt đầu.")
