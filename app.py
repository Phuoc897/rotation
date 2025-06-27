import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray, block_size: int = 256):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.block_size = block_size

    @st.experimental_memo
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

    def centering(self, pts):
        center = np.array([self.w/2, self.h/2, 0], dtype=np.float32)
        return pts - center

    def initialize_projection(self, max_angle):
        d = max(self.h, self.w)
        self.f = float(d * 1.5 * (1 + max_angle/90))
        self.cx, self.cy = self.w/2, self.h/2

    def project(self, pts3d):
        cam = pts3d.copy()
        cam[:, 2] += self.f * 2
        valid = cam[:, 2] > 0.1
        cam = cam[valid]
        if cam.size == 0:
            return np.empty((0,2), dtype=np.int32), valid
        x_p = cam[:,0] / cam[:,2]
        y_p = cam[:,1] / cam[:,2]
        u = self.f * x_p + self.cx
        v = self.f * y_p + self.cy
        pts2d = np.vstack((u, v)).T.astype(np.int32)
        return pts2d, valid

    def rotate_image_2d(self, angle=0):
        # Gốc algorithm giữ nguyên
        h, w = self.image.shape[:2]
        rad = np.deg2rad(angle)
        cos, sin = np.abs(np.cos(rad)), np.abs(np.sin(rad))
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(self.image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0):
        # Gốc algorithm tính R_x,R_y,R_z
        a, t, g = np.deg2rad([alpha, theta, gamma], dtype=np.float32)
        R_x = self.givens_matrix(1, 2, a)
        R_y = self.givens_matrix(0, 2, t)
        R_z = self.givens_matrix(0, 1, g)
        max_ang = max(abs(alpha), abs(theta), abs(gamma))
        self.initialize_projection(max_ang)

        # Tính canvas size qua block-processing nhưng giữ thuật toán gốc
        max_u = max_v = 0
        for y0 in range(0, self.h, self.block_size):
            for x0 in range(0, self.w, self.block_size):
                ys = min(self.block_size, self.h - y0)
                xs = min(self.block_size, self.w - x0)
                # create tile points
                yy, xx = np.meshgrid(
                    np.arange(ys, dtype=np.float32),
                    np.arange(xs, dtype=np.float32), indexing='ij'
                )
                zz = np.zeros_like(xx)
                pts = np.vstack((xx.ravel()+x0, yy.ravel()+y0, zz.ravel())).T
                pts_c = self.centering(pts)
                pts3d = pts_c @ R_z @ R_y @ R_x
                pts2d, _ = self.project(pts3d)
                if pts2d.size:
                    max_u = max(max_u, pts2d[:,0].max())
                    max_v = max(max_v, pts2d[:,1].max())
        Hc, Wc = max_v+1, max_u+1
        canvas = np.full((Hc, Wc, *self.image.shape[2:]), 255, dtype=self.image.dtype)

        # Gán pixel từng block theo gốc algorithm
        for y0 in range(0, self.h, self.block_size):
            for x0 in range(0, self.w, self.block_size):
                ys = min(self.block_size, self.h - y0)
                xs = min(self.block_size, self.w - x0)
                yy, xx = np.meshgrid(
                    np.arange(ys, dtype=np.float32),
                    np.arange(xs, dtype=np.float32), indexing='ij'
                )
                zz = np.zeros_like(xx)
                pts = np.vstack((xx.ravel()+x0, yy.ravel()+y0, zz.ravel())).T
                pts_c = self.centering(pts)
                pts3d = pts_c @ R_z @ R_y @ R_x
                pts2d, mask = self.project(pts3d)
                valid_src = pts.astype(int)[mask]
                for idx, (u, v) in enumerate(pts2d):
                    x_s, y_s = valid_src[idx]
                    canvas[v, u] = self.image[y_s, x_s]
        return canvas

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Xoay ảnh 2D & 3D", layout="wide")
st.title("🎨 Ứng dụng Xoay ảnh và Chỉnh sáng")

sidebar = st.sidebar
do_sang = sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)
che_do = sidebar.radio("Chế độ xoay", ["2D","3D"])
if che_do == "2D":
    goc = sidebar.slider("Góc xoay (°)", -180,180,0)
else:
    alpha = sidebar.slider("Alpha (X)", -45,45,0)
    theta = sidebar.slider("Theta (Y)", -45,45,0)
    gamma = sidebar.slider("Gamma (Z)", -45,45,0)

uploaded = st.file_uploader("Tải ảnh lên", type=['jpg','png','jpeg','bmp','tiff'])
if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("File không hợp lệ!")
    else:
        if img.ndim==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize input
        max_side=800
        h,w=img.shape[:2]
        scl=min(1.0, max_side/max(h,w))
        if scl<1.0:
            img=cv2.resize(img,None,fx=scl,fy=scl,interpolation=cv2.INTER_AREA)
        st.subheader("Ảnh gốc")
        st.image(img, width=300)
        rot=ImageRotation(img)
        if che_do=="2D":
            if sidebar.button("Xoay 2D") or goc!=0:
                out=rot.rotate_image_2d(goc)
                out=cv2.convertScaleAbs(out,alpha=do_sang)
                st.subheader(f"KQ 2D: góc={goc}°, sáng={do_sang}")
                st.image(out, width=300)
        else:
            if sidebar.button("Xoay 3D") or alpha or theta or gamma:
                out=rot.rotate_image_3d(alpha,theta,gamma)
                out=cv2.convertScaleAbs(out,alpha=do_sang)
                st.subheader(f"KQ 3D: α={alpha}°, θ={theta}°, γ={gamma}°, sáng={do_sang}")
                st.image(out, width=400)
else:
    st.info("Vui lòng tải ảnh lên.")

with st.expander("Tải ảnh mẫu"):
    if st.button("Tải ảnh mẫu qua Drive"):
        samples=[
            ("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE","sample1.jpg"),
            ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ","sample2.jpg")
        ]
        for gid,fname in samples:
            url=f"https://drive.google.com/uc?id={gid}"
            gdown.download(url,fname,quiet=True)
        st.success("Đã tải mẫu!")
