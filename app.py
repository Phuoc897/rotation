import numpy as np
import cv2
import streamlit as st
import gc

def resize_image(img, max_dim=512):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

st.set_page_config(page_title="Xoay ảnh 2D & 3D (Efficient)", layout="wide")
st.title("🎨 Xoay 2D & 3D bằng Perspective Transform")

# Sidebar
sidebar = st.sidebar
mode = sidebar.radio("Chế độ xoay", ["2D", "3D"])
brightness = sidebar.slider("Độ sáng", 0.1, 2.0, 1.0, 0.1)

if mode == "2D":
    angle = sidebar.slider("Góc xoay (°)", -180, 180, 0)
else:
    pitch = np.deg2rad(sidebar.slider("Pitch (X)", -45, 45, 0))
    yaw   = np.deg2rad(sidebar.slider("Yaw (Y)",   -45, 45, 0))
    roll  = np.deg2rad(sidebar.slider("Roll (Z)",  -45, 45, 0))

uploaded = st.file_uploader("Tải ảnh lên", type=['jpg','jpeg','png','bmp','tiff'])

if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("Không hỗ trợ định dạng hoặc file lỗi.")
        st.stop()
    # Xử lý alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to save memory
    img = resize_image(img, max_dim=512)
    st.subheader("Ảnh gốc")
    st.image(img, width=300)

    h, w = img.shape[:2]
    # 2D rotation
    if mode == "2D":
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        dst = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    else:
        # 3D rotation via homography
        # Camera matrix
        f = max(h, w)
        K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]])
        # Rotation matrices
        Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        Rz = np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
        R = Rz @ Ry @ Rx
        # Homography
        H = K @ (R - np.array([[0,0,0],[0,0,0],[0,0,1]])) @ np.linalg.inv(K)
        dst = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

    # Adjust brightness
    dst = cv2.convertScaleAbs(dst, alpha=brightness, beta=0)
    st.subheader("Kết quả")
    st.image(dst, width=300)
    # Download
    _, buf = cv2.imencode('.png', dst)
    st.download_button("📥 Tải ảnh", buf.tobytes(), "output.png", mime='image/png')

    # Cleanup
    gc.collect()
else:
    st.info("Tải lên ảnh để xoay ngay!")
