import streamlit as st
import numpy as np
import cv2
import numba as nb
import gdown
import requests
from PIL import Image
from io import BytesIO

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        x, y = np.meshgrid(range(self.height), range(self.width), indexing='ij')
        z = np.zeros_like(x)
        self.pixels = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # ... các phương thức givens_matrix, centering_image, rotate_image_2d, 
    # givens_rotation_3d, initialize_projection, project_points, rotate_image_3d 
    # và assign_pixels_nb giữ nguyên như trước ...

# copy lại toàn bộ class và hàm assign_pixels_nb ở đây
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
st.set_page_config(page_title="Image Rotation", layout="wide")
st.title("🎨 Image Rotation with Givens Transform")

mode = st.sidebar.radio("Rotation Mode", ["2D", "3D"])

# 1) Nhập URL
url = st.text_input("Or enter an image URL here:")

# 2) Hoặc upload file
uploaded = st.file_uploader("Or upload an image file")

img = None
raw_bytes = None

# Nếu có URL, thử fetch
if url:
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        raw_bytes = resp.content
    except Exception as e:
        st.error(f"Không tải được ảnh từ URL: {e}")
        raw_bytes = None

# Nếu có upload, lấy raw bytes
if uploaded:
    raw_bytes = uploaded.read()

# Nếu đã có raw_bytes, decode
if raw_bytes:
    data = np.frombuffer(raw_bytes, np.uint8)
    img_raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img_raw is None:
        # fallback Pillow
        try:
            pil_img = Image.open(BytesIO(raw_bytes))
            pil_img = pil_img.convert("RGB")
            img = np.array(pil_img)
        except Exception:
            st.error("Định dạng file/URL không được hỗ trợ hoặc file bị lỗi.")
            st.stop()
    else:
        # chuyển về RGB
        if img_raw.ndim == 2:
            img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        elif img_raw.shape[2] == 4:
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # Hiển thị và xử lý
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
    st.info("Vui lòng nhập URL hoặc upload ảnh để bắt đầu.")

# Download samples
with st.expander("Download Sample Images"):
    if st.button("Download via gdown"):
        samples = [
            ("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE", "sample1.jpg"),
            ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ", "sample2.jpg")
        ]
        for gid, fname in samples:
            gdown.download(id=gid, output=fname, quiet=True)
        st.success("Samples downloaded.")
