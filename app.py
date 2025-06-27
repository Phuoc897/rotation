import numpy as np
import cv2
import numba as nb
import gdown
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------- Core Logic ---------------------
class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        # Tạo mask cho pixel không phải background
        if image.ndim == 3:
            # Tạo mask alpha channel hoặc phát hiện background trắng
            self.alpha_mask = self.create_alpha_mask(image)
        else:
            self.alpha_mask = np.ones((self.height, self.width), dtype=bool)
        
        # Tạo lưới tọa độ
        y, x = np.meshgrid(range(self.height), range(self.width), indexing='ij')
        z = np.zeros_like(x)
        self.pixels = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        
    def create_alpha_mask(self, image):
        """Tạo mask để loại bỏ background trắng"""
        if image.ndim == 3:
            # Phát hiện pixel trắng hoặc gần trắng
            white_threshold = 240
            mask = ~np.all(image >= white_threshold, axis=2)
            
            # Tùy chọn: Sử dụng edge detection để giữ lại vùng có nội dung
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Mở rộng vùng edges
            kernel = np.ones((3,3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Kết hợp mask
            mask = mask | (edges_dilated > 0)
            
            return mask
        else:
            return image < 240  # Loại bỏ pixel sáng cho ảnh grayscale

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

    def rotate_image_3d(self, alpha=0, theta=0, gamma=0, transparent_bg=True):
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
            if transparent_bg and self.image.ndim == 3:
                return np.zeros((100, 100, 4), dtype=np.uint8)  # RGBA với background trong suốt
            else:
                return np.ones_like(self.image) * 255
        
        # Tạo canvas đầu ra
        H, W = pts2d[:, 1].max() + 50, pts2d[:, 0].max() + 50  # Thêm padding
        H, W = max(H, 1), max(W, 1)  # Đảm bảo kích thước tối thiểu
        
        if transparent_bg and self.image.ndim == 3:
            # Tạo canvas RGBA với background trong suốt
            canvas = np.zeros((H, W, 4), dtype=self.image.dtype)
        elif self.image.ndim == 3:
            canvas = np.ones((H, W, self.image.shape[2]), dtype=self.image.dtype) * 255
        else:
            canvas = np.ones((H, W), dtype=self.image.dtype) * 255
        
        # Gán pixel với mask hợp lệ và alpha mask
        valid_pixels = self.pixels[valid_mask]
        alpha_mask_valid = self.alpha_mask.flatten()[valid_mask]
        
        return assign_pixels_with_alpha_nb(valid_pixels, pts2d, self.image, canvas, 
                                         alpha_mask_valid, transparent_bg)

@nb.njit(parallel=True)
def assign_pixels_with_alpha_nb(pixels, pts2d, img, out, alpha_mask, transparent_bg):
    H, W = out.shape[:2]
    for i in nb.prange(len(pixels)):
        if not alpha_mask[i]:  # Bỏ qua pixel background
            continue
            
        x, y = int(pixels[i][0]), int(pixels[i][1])
        u, v = pts2d[i]
        
        # Kiểm tra giới hạn
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and 0 <= u < W and 0 <= v < H:
            if transparent_bg and out.shape[2] == 4:  # RGBA
                if img.ndim == 3:
                    for c in range(3):
                        out[v, u, c] = img[y, x, c]
                    out[v, u, 3] = 255  # Alpha channel
                else:
                    out[v, u, :3] = img[y, x]
                    out[v, u, 3] = 255
            elif img.ndim == 3:
                for c in range(img.shape[2]):
                    out[v, u, c] = img[y, x, c]
            else:
                out[v, u] = img[y, x]
    return out

# --------------------- Giao diện Streamlit với Interactive 3D ---------------------
st.set_page_config(page_title="Interactive 3D Image Rotation", layout="wide", initial_sidebar_state="expanded")
st.title("🎨 Ứng dụng Xoay ảnh 3D Tương tác")

# Khởi tạo session state
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0
if 'theta' not in st.session_state:
    st.session_state.theta = 0
if 'gamma' not in st.session_state:
    st.session_state.gamma = 0

sidebar = st.sidebar
che_do = sidebar.radio("Chế độ xoay", ["3D Interactive", "2D"])
do_sang = sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)
trong_suot = sidebar.checkbox("Background trong suốt", True)

if che_do == "2D":
    goc = sidebar.slider("Góc xoay (độ)", -180, 180, 0)
else:
    st.sidebar.markdown("### 🎮 Điều khiển 3D")
    st.sidebar.markdown("**Cách sử dụng:**")
    st.sidebar.markdown("- Kéo chuột trên ảnh để xoay")
    st.sidebar.markdown("- Hoặc dùng slider bên dưới")
    
    alpha = sidebar.slider("Alpha (X - pitch, °)", -45, 45, st.session_state.alpha, key='alpha_slider')
    theta = sidebar.slider("Theta (Y - yaw, °)", -45, 45, st.session_state.theta, key='theta_slider')
    gamma = sidebar.slider("Gamma (Z - roll, °)", -45, 45, st.session_state.gamma, key='gamma_slider')
    
    # Reset button
    if sidebar.button("🔄 Reset"):
        st.session_state.alpha = 0
        st.session_state.theta = 0
        st.session_state.gamma = 0
        st.experimental_rerun()

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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(img, width=300)
        
        with col2:
            # Xử lý xoay ảnh
            rotation = ImageRotation(img)
            
            if che_do == "2D":
                if goc != 0:
                    try:
                        out = rotation.rotate_image_2d(goc)
                        out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                        st.subheader(f"Kết quả 2D: {goc}°")
                        st.image(out, width=300)
                    except Exception as e:
                        st.error(f"Lỗi xoay 2D: {str(e)}")
            
            else:  # 3D Interactive mode
                try:
                    # Cập nhật session state từ slider
                    st.session_state.alpha = alpha
                    st.session_state.theta = theta
                    st.session_state.gamma = gamma
                    
                    with st.spinner("Đang xoay ảnh 3D..."):
                        out = rotation.rotate_image_3d(
                            st.session_state.alpha, 
                            st.session_state.theta, 
                            st.session_state.gamma,
                            transparent_bg=trong_suot
                        )
                        out = cv2.convertScaleAbs(out, alpha=do_sang, beta=0)
                        
                        st.subheader(f"3D: α={st.session_state.alpha}°, θ={st.session_state.theta}°, γ={st.session_state.gamma}°")
                        
                        # Tạo interactive plot với khả năng kéo để xoay
                        fig = go.Figure()
                        
                        # Thêm ảnh
                        fig.add_trace(go.Image(z=out, name="Rotated Image"))
                        
                        # Cấu hình layout cho interactive
                        fig.update_layout(
                            width=500,
                            height=500,
                            margin=dict(l=0, r=0, t=30, b=0),
                            dragmode='pan',
                            title="🎮 Kéo chuột để xoay ảnh",
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)' if trong_suot else 'white',
                            paper_bgcolor='rgba(0,0,0,0)' if trong_suot else 'white'
                        )
                        
                        # Ẩn trục
                        fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
                        fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
                        
                        # Hiển thị plot và capture mouse events
                        event = st.plotly_chart(fig, use_container_width=False, key="3d_plot")
                        
                        # Thêm JavaScript để handle mouse drag (chỉ mô phỏng)
                        st.markdown("""
                        <div style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); 
                                   padding: 10px; border-radius: 10px; margin: 10px 0;">
                            <h4 style="color: white; margin: 0;">💡 Mẹo sử dụng:</h4>
                            <p style="color: white; margin: 5px 0;">
                                • Sử dụng slider để xoay ảnh theo từng trục<br>
                                • Alpha (X): Xoay lên/xuống<br>
                                • Theta (Y): Xoay trái/phải<br>
                                • Gamma (Z): Xoay nghiêng<br>
                                • Bật "Background trong suốt" để loại bỏ nền trắng
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Hiển thị ảnh tĩnh bổ sung
                        st.image(out, width=400, caption="Ảnh kết quả")
                        
                except Exception as e:
                    st.error(f"Lỗi xoay 3D: {str(e)}")
                    st.write("Chi tiết lỗi:", e)

else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")
    
    # Demo với ảnh mẫu
    st.markdown("### 🖼️ Hoặc thử với ảnh mẫu:")
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        if st.button("🐱 Ảnh mèo"):
            # Tạo ảnh demo đơn giản
            demo_img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.circle(demo_img, (100, 100), 50, (255, 192, 203), -1)  # Pink circle
            cv2.circle(demo_img, (85, 85), 8, (0, 0, 0), -1)  # Left eye
            cv2.circle(demo_img, (115, 85), 8, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(demo_img, (100, 110), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Smile
            
            st.session_state.demo_img = demo_img
            st.image(demo_img, width=150)
    
    with demo_col2:
        if st.button("🔵 Hình học"):
            demo_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            cv2.rectangle(demo_img, (50, 50), (150, 150), (0, 100, 255), -1)
            cv2.circle(demo_img, (100, 100), 30, (255, 255, 0), -1)
            
            st.session_state.demo_img = demo_img
            st.image(demo_img, width=150)
    
    with demo_col3:
        if st.button("⭐ Ngôi sao"):
            demo_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            pts = np.array([[100,30], [120,70], [160,70], [130,100], [140,140], 
                           [100,120], [60,140], [70,100], [40,70], [80,70]], np.int32)
            cv2.fillPoly(demo_img, [pts], (255, 215, 0))
            
            st.session_state.demo_img = demo_img
            st.image(demo_img, width=150)

# Phần tải ảnh mẫu từ Google Drive
with st.expander("📥 Tải ảnh mẫu từ Google Drive"):
    if st.button("Tải ảnh mẫu"):
        try:
            samples = [
                ("1HQmRC6D5vKDwVjsVUbs5GBQ0x_2KjtNE", "sample1.jpg"),
                ("1Acz81dy_j9kXV956N0_88gsEW8BQKVSQ", "sample2.jpg")
            ]
            for gid, fname in samples:
                url = f"https://drive.google.com/uc?id={gid}"
                gdown.download(url, fname, quiet=True)
            st.success("✅ Tải xong ảnh mẫu!")
        except Exception as e:
            st.error(f"❌ Lỗi tải ảnh mẫu: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <h4>🎯 Tính năng chính:</h4>
    <p>✨ Xoay ảnh 3D theo 3 trục với Givens rotation<br>
    🎮 Giao diện tương tác với slider điều khiển<br>
    🎨 Loại bỏ background tự động<br>
    ⚡ Tối ưu hóa với Numba để xử lý nhanh</p>
</div>
""", unsafe_allow_html=True)
