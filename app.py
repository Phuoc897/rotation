import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st
import plotly.graph_objects as go

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        # image already resized and in RGB
        self.image = image
        self.height, self.width = image.shape[:2]

        # tạo meshgrid bằng float32 để giảm bộ nhớ
        y, x = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width,  dtype=np.float32),
            indexing='ij'
        )
        z = np.zeros_like(x, dtype=np.float32)
        self.pixels = np.vstack((
            x.ravel(),
            y.ravel(),
            z.ravel()
        )).T  # dtype float32

    def givens_matrix(self, i, j, theta):
        if i == j or i < 0 or j < 0 or i > 2 or j > 2:
            raise ValueError("Invalid Givens indices")
        if i > j:
            i, j = j, i
        G = np.eye(3, dtype=np.float32)
        c = np.cos(theta).astype(np.float32)
        s = np.sin(theta).astype(np.float32)
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
        return cv2.warpAffine(
            self.image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderValue=(255,255,255)
        )

    def givens_rotation_3d(self, a, t, g):
        R_x = self.givens_matrix(1, 2, a)
        R_y = self.givens_matrix(0, 2, t)
        R_z = self.givens_matrix(0, 1, g)
        pts = self.centering_image(self.pixels)
        return pts @ R_z @ R_y @ R_x

    def initialize_projection(self, max_angle):
        d = max(self.height, self.width)
        self.focal_length = float(d * 1.5 * (1 + max_angle/90))
        cx, cy = self.width/2, self.height/2
        self.camera_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def project_points(self, pts3d):
        cam = pts3d.copy()
        cam[:, 2] += self.focal_length * 2
        valid = cam[:, 2] > 0.1
        cam = cam[valid]
        if cam.size == 0:
            return np.empty((0,2), int), valid
        x_proj = cam[:,0] / cam[:,2]
        y_proj = cam[:,1] / cam[:,2]
        u = self.focal_length * x_proj + self.camera_matrix[0,2]
        v = self.focal_length * y_proj + self.camera_matrix[1,2]
        pts2d = np.vstack((u, v)).T.astype(np.int32)
        min_u, min_v = pts2d.min(axis=0)
        if min_u < 0: pts2d[:,0] -= int(min_u)
        if min_v < 0: pts2d[:,1] -= int(min_v)
        return pts2d, valid

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0):
        a, t, g = np.deg2rad([alpha, theta, gamma], dtype=np.float32)
        pts3d = self.givens_rotation_3d(a, t, g)
        self.initialize_projection(max(abs(alpha), abs(theta), abs(gamma)))
        pts2d, mask = self.project_points(pts3d)
        if pts2d.size == 0:
            return np.ones_like(self.image) * 255
        H, W = pts2d[:,1].max()+1, pts2d[:,0].max()+1
        canvas = np.full(
            (max(H,1), max(W,1), *self.image.shape[2:]),
            255, dtype=self.image.dtype
        )
        return assign_pixels_nb(self.pixels[mask], pts2d, self.image, canvas)

@nb.njit(parallel=True)
def assign_pixels_nb(pixels, pts2d, img, out):
    H, W = out.shape[:2]
    for i in nb.prange(pixels.shape[0]):
        x, y = int(pixels[i,0]), int(pixels[i,1])
        u, v = pts2d[i]
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and 0 <= u < W and 0 <= v < H:
            if img.ndim == 3:
                out[v, u] = img[y, x]
            else:
                out[v, u] = img[y, x]
    return out

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Xoay ảnh 2D & 3D", layout="wide", initial_sidebar_state="expanded")
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

uploaded = st.file_uploader("Tải ảnh lên", type=['jpg','jpeg','png','bmp','tiff'])

if uploaded:
    # Đọc ảnh
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is None:
        st.error("Định dạng file không được hỗ trợ hoặc file bị lỗi.")
    else:
        # Chuyển màu
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ——— THÊM: resize ảnh về max 800px cạnh dài nhất ———
        h, w = img.shape[:2]
        max_side = 800
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # ——————————————————————————

        # Hiển thị ảnh gốc
        st.subheader("Ảnh gốc")
        st.image(img, width=300)

        # Khởi tạo xoay
        rotation = ImageRotation(img)

        if che_do == "2D":
            if sidebar.button("Xoay 2D") or goc != 0:
                try:
                    out = rotation.rotate_image_2d(goc)
                    out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                    st.subheader(f"Kết quả 2D: Góc={goc}°, Độ sáng={do_sang}")
                    st.image(out, width=300)
                except Exception as e:
                    st.error(f"Lỗi xoay 2D: {e}")
        else:
            if sidebar.button("Xoay 3D") or alpha!=0 or theta!=0 or gamma!=0:
                try:
                    out = rotation.rotate_image_3d(alpha, theta, gamma)
                    out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                    st.subheader(f"Kết quả 3D: α={alpha}°, θ={theta}°, γ={gamma}°, Độ sáng={do_sang}")
                    st.image(out, width=400)
                except Exception as e:
                    st.error(f"Lỗi xoay 3D: {e}")

else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

# Sample images
with st.expander("Tải ảnh mẫu"):
    if st.button("Tải ảnh mẫu qua Google Drive"):
        try:
            samples = [
                ("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE", "sample1.jpg"),
                ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ", "sample2.jpg")
            ]
            for gid, fname in samples:
                url = f"https://drive.google.com/uc?id={gid}"
                gdown.download(url, fname, quiet=True)
            st.success("Tải xong ảnh mẫu!")
        except Exception as e:
            st.error(f"Lỗi tải ảnh mẫu: {e}")
