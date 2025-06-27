import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st
import plotly.graph_objects as go

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        # Sửa lỗi: tạo lưới tọa độ đúng
        y, x = np.meshgrid(range(self.height), range(self.width), indexing='ij')
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
        return cv2.warpAffine(self.image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

    def givens_rotation_3d(self, a, t, g):
        # Ma trận xoay quanh trục X (pitch)
        R_x = self.givens_matrix(1, 2, a)
        # Ma trận xoay quanh trục Y (yaw)
        R_y = self.givens_matrix(0, 2, t)
        # Ma trận xoay quanh trục Z (roll)
        R_z = self.givens_matrix(0, 1, g)
        
        pts = self.centering_image(self.pixels)
        # Áp dụng các phép xoay theo thứ tự Z-Y-X
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
        # Dịch chuyển các điểm về phía trước camera
        cam = pts3d.copy()
        cam[:, 2] += self.focal_length * 2
        
        # Lọc các điểm có z > 0 (trước camera)
        valid_mask = cam[:, 2] > 0.1
        cam = cam[valid_mask]
        
        if len(cam) == 0:
            return np.array([]), valid_mask
        
        # Phép chiếu phối cảnh
        x_proj = cam[:, 0] / cam[:, 2]
        y_proj = cam[:, 1] / cam[:, 2]
        
        # Chuyển đổi sang tọa độ pixel
        u = self.focal_length * x_proj + self.camera_matrix[0, 2]
        v = self.focal_length * y_proj + self.camera_matrix[1, 2]
        
        pts2d = np.column_stack((u, v)).astype(int)
        
        # Dịch chuyển để đảm bảo tọa độ dương
        if len(pts2d) > 0:
            min_u, min_v = pts2d.min(axis=0)
            if min_u < 0:
                pts2d[:, 0] -= min_u
            if min_v < 0:
                pts2d[:, 1] -= min_v
        
        return pts2d, valid_mask

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0):
        # Chuyển đổi sang radian
        a, t, g = np.deg2rad([alpha, theta, gamma])
        
        # Áp dụng phép xoay 3D
        pts3d = self.givens_rotation_3d(a, t, g)
        
        # Khởi tạo thông số chiếu
        max_angle = max(abs(alpha), abs(theta), abs(gamma))
        self.initialize_projection(max_angle)
        
        # Chiếu xuống 2D
        pts2d, valid_mask = self.project_points(pts3d)
        
        if len(pts2d) == 0:
            # Trả về ảnh trắng nếu không có điểm hợp lệ
            return np.ones_like(self.image) * 255
        
        # Tạo canvas đầu ra
        H, W = pts2d[:, 1].max() + 1, pts2d[:, 0].max() + 1
        H, W = max(H, 1), max(W, 1)  # Đảm bảo kích thước tối thiểu
        
        if self.image.ndim == 3:
            canvas = np.ones((H, W, self.image.shape[2]), dtype=self.image.dtype) * 255
        else:
            canvas = np.ones((H, W), dtype=self.image.dtype) * 255
        
        # Gán pixel với mask hợp lệ
        valid_pixels = self.pixels[valid_mask]
        return assign_pixels_nb(valid_pixels, pts2d, self.image, canvas)

@nb.njit(parallel=True)
def assign_pixels_nb(pixels, pts2d, img, out):
    H, W = out.shape[:2]
    for i in nb.prange(len(pixels)):
        x, y = int(pixels[i][0]), int(pixels[i][1])
        u, v = pts2d[i]
        
        # Kiểm tra giới hạn
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and 0 <= u < W and 0 <= v < H:
            if img.ndim == 3:
                for c in range(img.shape[2]):
                    out[v, u, c] = img[y, x, c]
            else:
                out[v, u] = img[y, x]
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
    alpha = sidebar.slider("Alpha (xoay X - pitch, °)", -45, 45, 0)
    theta = sidebar.slider("Theta (xoay Y - yaw, °)", -45, 45, 0)
    gamma = sidebar.slider("Gamma (xoay Z - roll, °)", -45, 45, 0)

uploaded = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

if uploaded:
    # Đọc ảnh
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        st.error("Định dạng file không được hỗ trợ hoặc file bị lỗi.")
    else:
        # Chuyển đổi màu sắc nếu cần
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Hiển thị ảnh gốc
        st.subheader("Ảnh gốc")
        st.image(img, width=300)
        
        # Xử lý xoay ảnh
        rotation = ImageRotation(img)
        
        if che_do == "2D":
            if sidebar.button("Xoay 2D") or goc != 0:
                try:
                    out = rotation.rotate_image_2d(goc)
                    out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                    st.subheader(f"Kết quả 2D: Góc={goc}°, Độ sáng={do_sang}")
                    st.image(out, width=300)
                except Exception as e:
                    st.error(f"Lỗi xoay 2D: {str(e)}")
        
        else:  # 3D mode
            if sidebar.button("Xoay 3D") or alpha != 0 or theta != 0 or gamma != 0:
                try:
                    with st.spinner("Đang xoay ảnh 3D..."):
                        out = rotation.rotate_image_3d(alpha, theta, gamma)
                        out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                        
                        st.subheader(f"Kết quả 3D: α={alpha}°, θ={theta}°, γ={gamma}°, Độ sáng={do_sang}")
                        
                        # Hiển thị bằng st.image thay vì plotly để đảm bảo hoạt động
                        st.image(out, width=400)
                        
                        # Tùy chọn: Thêm hiển thị plotly interactive
                        if st.checkbox("Hiển thị interactive (Plotly)"):
                            fig = go.Figure(go.Image(z=out))
                            fig.update_layout(
                                width=400, height=400,
                                margin=dict(l=0, r=0, t=0, b=0),
                                dragmode='pan',
                                title="Ảnh 3D (có thể zoom/pan)"
                            )
                            fig.update_xaxes(visible=False)
                            fig.update_yaxes(visible=False)
                            st.plotly_chart(fig, use_container_width=False)
                            
                except Exception as e:
                    st.error(f"Lỗi xoay 3D: {str(e)}")
                    st.write("Chi tiết lỗi:", e)
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

# Phần tải ảnh mẫu
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
            st.error(f"Lỗi tải ảnh mẫu: {str(e)}")

st.markdown("---")
st.markdown("**Hướng dẫn sử dụng:**")
st.markdown("- **2D**: Xoay ảnh theo góc đơn giản")
st.markdown("- **3D**: Xoay ảnh theo 3 trục (X, Y, Z)")
st.markdown("- **Alpha (X)**: Xoay lên/xuống (pitch)")
st.markdown("- **Theta (Y)**: Xoay trái/phải (yaw)")  
st.markdown("- **Gamma (Z)**: Xoay nghiêng (roll)")
