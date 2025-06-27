import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st  # <-- Added missing import

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        # Prepare pixel coordinates (x: row, y: col, z=0)
        x, y = np.meshgrid(range(self.height), range(self.width), indexing='ij')
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
        center = np.mean(pixels, axis=0)
        return pixels - center

    def rotate_image_2d(self, angle=0):
        """
        Xoay 2D sử dụng OpenCV, mở rộng canvas để giữ nguyên trọn vẹn ảnh.
        """
        h, w = self.image.shape[:2]
        rad = np.deg2rad(angle)
        cos, sin = np.abs(np.cos(rad)), np.abs(np.sin(rad))
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        center_orig = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center_orig, angle, 1.0)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            self.image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderValue=(255, 255, 255)
        )
        return rotated

    def givens_rotation_3d(self, alpha, theta, gamma):
        R_x = self.givens_matrix(0, 2, alpha)
        R_y = self.givens_matrix(1, 2, theta)
        R_z = self.givens_matrix(0, 1, gamma)
        pts_centered = self.centering_image(self.pixels)
        return pts_centered @ R_x @ R_y @ R_z

    def initialize_projection(self, max_angle):
        max_dim = max(self.height, self.width)
        factor = 1 + max_angle / 90
        self.focal_length = max_dim * 1.2 * factor
        cx, cy = self.height / 2, self.width / 2
        self.camera_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def project_points(self, pts3d):
        tvec = np.array([0, 0, self.focal_length * 1.5], dtype=np.float32)
        cam = pts3d.T + tvec.reshape(3, 1)
        x, y = cam[0] / cam[2], cam[1] / cam[2]
        fx = self.camera_matrix[0, 0]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        u, v = fx * x + cx, fx * y + cy
        pts2d = np.vstack((u, v)).T.astype(int)
        pts2d -= pts2d.min(axis=0)
        return pts2d

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0):
        """
        Xoay 3D bằng Givens và chiếu xuống 2D.
        """
        a, t, g = np.deg2rad([alpha, theta, gamma])
        rotated3d = self.givens_rotation_3d(a, t, g)
        max_ang = max(abs(alpha), abs(theta), abs(gamma))
        self.initialize_projection(max_ang)
        pts2d = self.project_points(rotated3d)

        h_out, w_out = pts2d[:, 0].max() + 1, pts2d[:, 1].max() + 1
        channels = 3 if self.image.ndim == 3 else 1
        canvas = np.ones((h_out, w_out, channels), dtype=self.image.dtype) * 255
        return assign_pixels_nb(self.pixels, pts2d, self.image, canvas)

@nb.njit(parallel=True)
def assign_pixels_nb(pixels, pts2d, img, out):
    for i in nb.prange(pixels.shape[0]):
        x, y = int(pixels[i, 0]), int(pixels[i, 1])
        u, v = pts2d[i, 0], pts2d[i, 1]
        if img.ndim == 3:
            for c in range(img.shape[2]):
                out[u, v, c] = img[x, y, c]
        else:
            out[u, v] = img[x, y]
    return out

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Image Rotation", layout="wide")
st.title("🎨 Image Rotation with Givens Transform")

mode = st.sidebar.radio("Rotation Mode", ["2D", "3D"])
uploaded = st.file_uploader("Upload an image", type=None)  # Allow all types

if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("Định dạng file không được hỗ trợ hoặc file bị lỗi.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        st.subheader("Original Image")
        st.image(img, use_column_width=True)

        if mode == "2D":
            angle2d = st.sidebar.slider("Angle (degrees)", -180, 180, 0)
            if st.sidebar.button("Rotate 2D"):
                with st.spinner("Rotating 2D..."):
                    out2d = ImageRotation(img).rotate_image_2d(angle2d)
                    st.subheader(f"2D Rotated (θ={angle2d}°)")
                    st.image(out2d, use_column_width=True)
        else:
            alpha = st.sidebar.slider("Alpha (X-axis)", -90, 90, 0)
            theta = st.sidebar.slider("Theta (Y-axis)", -90, 90, 0)
            gamma = st.sidebar.slider("Gamma (Z-axis)", -90, 90, 0)
            if st.sidebar.button("Rotate 3D"):
                with st.spinner("Rotating 3D..."):
                    out3d = ImageRotation(img).rotate_image_3d(alpha, theta, gamma)
                    st.subheader(f"3D Rotated (α={alpha}°, θ={theta}°, γ={gamma}°)")
                    st.image(out3d, use_column_width=True)
else:
    st.info("Please upload an image to begin.")

# Download samples
with st.expander("Download Sample Images"):
    if st.button("Download via gdown"):
        samples = [
            ("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE", "sample1.jpg"),
            ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ", "sample2.jpg")
        ]
        for gid, fname in samples:
            url = f"https://drive.google.com/uc?id={gid}"
            gdown.download(url, fname, quiet=True)
        st.success("Samples downloaded.")
