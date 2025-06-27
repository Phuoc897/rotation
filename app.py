import streamlit as st
import numpy as np
import cv2
import numba as nb
from PIL import Image
import gdown

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        # Prepare pixel coordinates
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
        G[i, i] = c; G[j, j] = c
        G[i, j] = s; G[j, i] = -s
        return G

    def centering_image(self, pixels):
        center = np.mean(pixels, axis=0)
        return pixels - center

    def givens_rotation(self, pixels, alpha, theta, gamma):
        R_x = self.givens_matrix(0, 2, alpha)
        R_y = self.givens_matrix(1, 2, theta)
        R_z = self.givens_matrix(0, 1, gamma)
        pixels_centered = self.centering_image(pixels)
        return pixels_centered @ R_x @ R_y @ R_z

    def initialize_projection(self, max_angle):
        max_dim = max(self.height, self.width)
        angle_factor = 1 + max_angle / 90
        self.focal_length = max_dim * 1.2 * angle_factor
        cx, cy = self.height / 2, self.width / 2
        self.camera_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def project_points(self, points_3d):
        tvec = np.array([0, 0, self.focal_length * 1.5], dtype=np.float32)
        X_cam = points_3d.T + tvec.reshape(3,1)
        x, y = X_cam[0] / X_cam[2], X_cam[1] / X_cam[2]
        fx = self.camera_matrix[0, 0]
        cx = self.camera_matrix[0, 2]; cy = self.camera_matrix[1, 2]
        u = fx * x + cx; v = fx * y + cy
        pts2d = np.vstack((u, v)).T.astype(int)
        pts2d -= pts2d.min(axis=0)
        return pts2d

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0):
        # Convert degrees to radians
        a, t, g = np.deg2rad([alpha, theta, gamma])
        # Rotate
        rotated = self.givens_rotation(self.pixels.copy(), a, t, g)
        # Projection setup
        max_ang = max(abs(alpha), abs(theta), abs(gamma))
        self.initialize_projection(max_ang)
        pts2d = self.project_points(rotated)
        # Prepare output canvas
        h_out = pts2d[:,0].max() + 1
        w_out = pts2d[:,1].max() + 1
        channels = 3 if self.image.ndim == 3 else 1
        canvas = np.ones((h_out, w_out, channels), dtype=self.image.dtype) * 255
        # Map pixels
        canvas = assign_pixels_nb(self.pixels, pts2d, self.image, canvas)
        return canvas

@nb.njit(parallel=True)
def assign_pixels_nb(pixels, pts2d, img, out):
    for i in nb.prange(pixels.shape[0]):
        x, y = int(pixels[i,0]), int(pixels[i,1])
        u, v = pts2d[i,0], pts2d[i,1]
        if img.ndim == 3:
            for c in range(img.shape[2]):
                out[u, v, c] = img[x, y, c]
        else:
            out[u, v] = img[x, y]
    return out

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="3D Image Rotation", layout="wide")
st.title("🎨 3D Image Rotation with Givens Transform")

# Sidebar controls
st.sidebar.header("Parameters")
alpha = st.sidebar.slider("Alpha (X-axis)", -90, 90, 0)
theta = st.sidebar.slider("Theta (Y-axis)", -90, 90, 0)
gamma = st.sidebar.slider("Gamma (Z-axis)", -90, 90, 0)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    if st.sidebar.button("Rotate Image"):
        with st.spinner("Rotating..."):
            rot = ImageRotation(img).rotate_image_3d(alpha, theta, gamma)
            st.subheader(f"Rotated Image (α={alpha}, θ={theta}, γ={gamma})")
            st.image(rot, use_column_width=True)
else:
    st.info("Please upload an image to begin.")

# Example: Download sample images
with st.expander("Download Sample Images"):
    if st.button("Download Samples via gdown"):
        urls = [("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE", "sample1.jpg"),
                ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ", "sample2.jpg")]
        for gid, fname in urls:
            gdown.download(id=gid, output=fname, quiet=True)
        st.success("Sample images downloaded.")
