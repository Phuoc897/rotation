import numpy as np
import cv2
import streamlit as st
import gc

# ------------------ Helper ------------------
def resize_image(img, max_dim=512):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

@st.cache_resource(ttl=1800, max_entries=2)
def load_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return resize_image(img)

@st.cache_data(ttl=600)
def rotate_image_2d(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

# ------------------ UI ------------------
st.set_page_config(page_title="Xoay ảnh tiết kiệm RAM", layout="centered")
st.title("📷 Xoay ảnh 2D (Tối ưu RAM)")

angle = st.sidebar.slider("Góc xoay (độ)", -180, 180, 0)
bright = st.sidebar.slider("Độ sáng", 0.5, 2.0, 1.0, 0.1)

uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        img = load_image(uploaded.read())
        st.image(img, caption="Ảnh gốc", width=300)

        if st.button("Xoay ảnh"):
            out = rotate_image_2d(img, angle)
            out = cv2.convertScaleAbs(out, alpha=bright)
            st.image(out, caption=f"Ảnh đã xoay: {angle}°", width=300)

            del out
            gc.collect()
    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

st.markdown("---")
st.markdown("**Gợi ý:** App này được tối ưu để tránh vượt giới hạn bộ nhớ trên Streamlit Community Cloud bằng cách giảm kích thước ảnh và cache hợp lý.")
