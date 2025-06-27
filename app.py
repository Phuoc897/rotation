import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st
import plotly.graph_objects as go  # For interactive pan/zoom

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
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
        return pixels - np.mean(pixels, axis=0)

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
        R_x = self.givens_matrix(0, 2, a)
        R_y = self.givens_matrix(1, 2, t)
        R_z = self.givens_matrix(0, 1, g)
        pts = self.centering_image(self.pixels)
        return pts @ R_x @ R_y @ R_z

    def initialize_projection(self, max_angle):
        d = max(self.height, self.width)
        self.focal_length = d * 1.2 * (1 + max_angle/90)
        cx, cy = self.height/2, self.width/2
        self.camera_matrix = np.array([[self.focal_length,0,cx],[0,self.focal_length,cy],[0,0,1]], dtype=np.float32)

    def project_points(self, pts3d):
        cam = pts3d.T + np.array([0,0,self.focal_length*1.5]).reshape(3,1)
        x, y = cam[0]/cam[2], cam[1]/cam[2]
        fx = self.camera_matrix[0,0]
        u = fx*x + self.camera_matrix[0,2]
        v = fx*y + self.camera_matrix[1,2]
        pts2d = np.vstack((u, v)).T.astype(int)
        return pts2d - pts2d.min(axis=0)

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0):
        a, t, g = np.deg2rad([alpha, theta, gamma])
        pts3d = self.givens_rotation_3d(a, t, g)
        self.initialize_projection(max(abs(alpha), abs(theta), abs(gamma)))
        pts2d = self.project_points(pts3d)
        H, W = pts2d[:,0].max()+1, pts2d[:,1].max()+1
        canvas = np.ones((H, W, (3 if self.image.ndim==3 else 1)), dtype=self.image.dtype)*255
        return assign_pixels_nb(self.pixels, pts2d, self.image, canvas)

@nb.njit(parallel=True)
def assign_pixels_nb(pixels, pts2d, img, out):
    for i in nb.prange(pixels.shape[0]):
        x,y = pixels[i]
        u,v = pts2d[i]
        if img.ndim==3:
            for c in range(img.shape[2]): out[u,v,c] = img[x,y,c]
        else:
            out[u,v] = img[x,y]
    return out

# --------------------- Giao diện Streamlit ---------------------
st.set_page_config(page_title="Xoay ảnh 2D & 3D", layout="wide", initial_sidebar_state="expanded")
st.title("🎨 Ứng dụng Xoay ảnh và Chỉnh sáng")

sidebar = st.sidebar
che_do = sidebar.radio("Chế độ xoay", ["2D", "3D"])
do_sang = sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)
if che_do == "2D":
    goc = sidebar.slider("Góc xoay (độ)", -180, 180, 0)
else:
    alpha = sidebar.slider("Alpha (trục X, °)", -90, 90, 0)
    theta = sidebar.slider("Theta (trục Y, °)", -90, 90, 0)
    gamma = sidebar.slider("Gamma (trục Z, °)", -90, 90, 0)

uploaded = st.file_uploader("Tải ảnh lên", type=None)
if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("Định dạng file không được hỗ trợ hoặc file bị lỗi.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim==3 else img
        st.subheader("Ảnh gốc")
        st.image(img, width=300)

        if che_do == "2D" and sidebar.button("Xoay 2D"):
            out = ImageRotation(img).rotate_image_2d(goc)
            out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
            st.subheader(f"2D: Góc={goc}°, Độ sáng={do_sang}")
            st.image(out, width=300)

        if che_do == "3D" and sidebar.button("Xoay 3D"):
            out = ImageRotation(img).rotate_image_3d(alpha, theta, gamma)
            out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
            # Use Plotly Image trace for interactivity
            fig = go.Figure(go.Image(z=out))
            fig.update_layout(
                width=400, height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                dragmode='pan'
            )
            st.subheader(f"3D: α={alpha}°, θ={theta}°, γ={gamma}°, Độ sáng={do_sang}")
            st.plotly_chart(fig, use_container_width=False)
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

with st.expander("Tải ảnh mẫu"):
    if st.button("Tải qua gdown"):
        samples = [("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE","sample1.jpg"),
                   ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ","sample2.jpg")]
        for gid, fname in samples:
            gdown.download(f"https://drive.google.com/uc?id={gid}", fname, quiet=True)
        st.success("Tải xong ảnh mẫu.")
