import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image.copy()
        self.h, self.w = image.shape[:2]
        # Tính sẵn 4 góc
        self.corners = np.array([
            [0, 0, 0],
            [self.w, 0, 0],
            [self.w, self.h, 0],
            [0, self.h, 0]
        ], dtype=np.float32)

    def givens_matrix(self, i, j, theta):
        if i == j:
            raise ValueError("Invalid Givens indices")
        G = np.eye(3, dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i] = c; G[j, j] = c
        G[i, j] = s; G[j, i] = -s
        return G

    def rotate_3d_corners(self, alpha, theta, gamma):
        # radian
        a, t, g = np.deg2rad(alpha), np.deg2rad(theta), np.deg2rad(gamma)
        Rz = self.givens_matrix(0,1, g)
        Ry = self.givens_matrix(0,2, t)
        Rx = self.givens_matrix(1,2, a)
        # Trung tâm ảnh
        center = np.array([self.w/2, self.h/2, 0], dtype=np.float32)
        # Dịch về tâm, xoay, dịch lại
        pts = (self.corners - center) @ (Rz @ Ry @ Rx).T + center
        return pts

    @st.cache_data(show_spinner=False)
    def rotate_image_3d(self, alpha=0, theta=0, gamma=0, transparent_bg=True):
        # 1) Xoay 4 góc
        dst_corners_3d = self.rotate_3d_corners(alpha, theta, gamma)
        # 2) Chiếu phối cảnh: giả sử focal = max(w,h)
        f = max(self.w, self.h)
        # Dịch Z để có điểm trước camera
        src_z = np.zeros((4,1), dtype=np.float32) + f*3
        dst_z = dst_corners_3d[:,2:3] + f*3
        # Tính u,v: u = f*x/z + cx; cx = w/2
        src_uv = np.hstack([
            f * (self.corners[:,0:1]-self.w/2) / src_z + self.w/2,
            f * (self.corners[:,1:2]-self.h/2) / src_z + self.h/2
        ]).astype(np.float32)
        dst_uv = np.hstack([
            f * (dst_corners_3d[:,0:1]-self.w/2) / dst_z + self.w/2,
            f * (dst_corners_3d[:,1:2]-self.h/2) / dst_z + self.h/2
        ]).astype(np.float32)
        # 3) Lấy H perspective
        H, _ = cv2.findHomography(src_uv, dst_uv)
        # Kích thước canvas mới (chọn sao cho đủ chứa điểm đích)
        u_min, v_min = dst_uv.min(axis=0)
        u_max, v_max = dst_uv.max(axis=0)
        width_new  = int(np.ceil(u_max - u_min))
        height_new = int(np.ceil(v_max - v_min))
        # Dịch chuyển so cho tất cả + để >=0
        T = np.array([[1,0,-u_min],[0,1,-v_min],[0,0,1]], dtype=np.float32)
        Ht = T @ H
        # 4) WarpPerspective
        if transparent_bg and self.image.ndim==3:
            # output RGBA
            canvas = cv2.warpPerspective(
                np.dstack([self.image, np.full((self.h,self.w),255,dtype=np.uint8)]),
                Ht, (width_new, height_new),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
            )
        else:
            canvas = cv2.warpPerspective(
                self.image, Ht, (width_new, height_new),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255)
            )
        return canvas

    def rotate_image_2d(self, angle=0, bg=(255,255,255)):
        h, w = self.image.shape[:2]
        rad = np.deg2rad(angle)
        cos, sin = np.abs(np.cos(rad)), np.abs(np.sin(rad))
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        M[0,2] += (new_w - w)/2
        M[1,2] += (new_h - h)/2
        return cv2.warpAffine(self.image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=bg)

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="3D Rotate (Optimized)", layout="wide")
st.title("🎨 Xoay ảnh 3D — Cực nhanh, cực nhẹ")

uploaded = st.file_uploader("Chọn ảnh (.jpg, .png, .bmp, .tiff)", type=['jpg','jpeg','png','bmp','tiff'])
if not uploaded:
    st.info("Tải lên ảnh để thử ngay!")
    st.stop()

# Đọc ảnh
data = np.frombuffer(uploaded.read(), np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
if img is None:
    st.error("Không đọc được ảnh. Có thể file bị hỏng.")
    st.stop()
if img.ndim==3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rotator = ImageRotation(img)

mode = st.sidebar.radio("Chế độ", ["2D", "3D"])
brightness = st.sidebar.slider("Độ sáng alpha", 0.1, 2.0, 1.0)
transparent = st.sidebar.checkbox("Nền trong suốt (3D)", True)

if mode=="2D":
    angle = st.sidebar.slider("Góc xoay 2D", -180, 180, 0)
    out = rotator.rotate_image_2d(angle)
else:
    st.sidebar.markdown("### Xoay 3 trục")
    a = st.sidebar.slider("Alpha (X)", -45, 45, 0)
    t = st.sidebar.slider("Theta (Y)", -45, 45, 0)
    g = st.sidebar.slider("Gamma (Z)", -45, 45, 0)
    out = rotator.rotate_image_3d(a, t, g, transparent)

# Áp dụng ánh sáng
out = cv2.convertScaleAbs(out, alpha=brightness, beta=0)

# Hiển thị
col1, col2 = st.columns(2)
with col1:
    st.subheader("Ảnh gốc")
    st.image(img, use_column_width=True)
with col2:
    st.subheader("Kết quả")
    st.image(out, use_column_width=True)
