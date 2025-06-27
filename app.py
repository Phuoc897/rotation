import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st
import gc

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray, step: int = 2):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.step = step
        # Tạo lưới tọa độ giảm chi tiết để tiết kiệm RAM
        y, x = np.meshgrid(
            np.arange(0, self.height, step),
            np.arange(0, self.width, step),
            indexing='ij'
        )
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
        return cv2.warpAffine(
            self.image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR, borderValue=(255,255,255)
        )

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
        canvas = np.ones((max(H,1), max(W,1), *self.image.shape[2:]), dtype=self.image.dtype) * 255
        valid_pixels = self.pixels[valid_mask]
        return assign_pixels_nb(valid_pixels, pts2d, self.image, canvas)

@nb.njit(parallel=True)
def assign_pixels_nb(pixels, pts2d, img, out):
    H, W = out.shape[:2]
    for i in nb.prange(len(pixels)):
        x, y = int(pixels[i][0]), int(pixels[i][1])
        u, v = pts2d[i]
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and 0 <= u < W and 0 <= v < H:
            if img.ndim == 3:
                for c in range(img.shape[2]):
                    out[v, u, c] = img[y, x, c]
            else:
                out[v, u] = img[y, x]
    return out

# --------------------- Streamlit Interface ---------------------
st.set_page_config(page_title="Xoay ảnh 2D & 3D", layout="wide")
st.title("🎨 Ứng dụng Xoay ảnh và Chỉnh sáng (Optimized)")

sidebar = st.sidebar
che_do = sidebar.radio("Chế độ xoay", ["2D", "3D"])
do_sang = sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)

if che_do == "2D":
    goc = sidebar.slider("Góc xoay (độ)", -180, 180, 0)
else:
    alpha = sidebar.slider("Alpha (Pitch)", -45, 45, 0)
    theta = sidebar.slider("Theta (Yaw)", -45, 45, 0)
    gamma = sidebar.slider("Gamma (Roll)", -45, 45, 0)

uploaded = st.file_uploader("Tải ảnh lên", type=['jpg','jpeg','png','bmp','tiff'])
MAX_DIM = 512

if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("Định dạng file không được hỗ trợ hoặc file bị lỗi.")
        st.stop()
    # Xử lý alpha channel
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize giảm tải theo MAX_DIM
    if max(img.shape[:2]) > MAX_DIM:
        scale = MAX_DIM / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    st.subheader("Ảnh gốc (resized)")
    st.image(img, width=300)

    rotation = ImageRotation(img, step=2)
    if che_do == "2D":
        if sidebar.button("Xoay 2D") or goc != 0:
            out = rotation.rotate_image_2d(goc)
    else:
        if sidebar.button("Xoay 3D") or (alpha or theta or gamma):
            with st.spinner("Đang xoay ảnh 3D..."):
                out = rotation.rotate_image_3d(alpha, theta, gamma)
    # Áp sáng và hiển thị
    if 'out' in locals():
        out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
        st.subheader("Kết quả")
        st.image(out, width=350)
        # Nút tải xuống
        is_success, buffer = cv2.imencode('.png', out)
        if is_success:
            st.download_button("📥 Tải ảnh kết quả", buffer.tobytes(), file_name='rotated_output.png', mime='image/png')
        # Dọn dẹp
        del rotation, out
        gc.collect()
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

# Phần ảnh mẫu
with st.expander("Tải ảnh mẫu"):
    if st.button("Tải ảnh mẫu qua Google Drive"):
        samples = [
            ("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE", "sample1.jpg"),
            ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ", "sample2.jpg"),
        ]
        for gid, fname in samples:
            url = f"https://drive.google.com/uc?id={gid}"
            gdown.download(url, fname, quiet=True)
        st.success("Tải xong ảnh mẫu!")

st.markdown("---")
st.markdown("**Hướng dẫn:**")
st.markdown("- 2D: Xoay theo góc đơn giản")
st.markdown("- 3D: Xoay theo 3 trục (XYZ)")
