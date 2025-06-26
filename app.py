import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import io
import math

# Cấu hình trang
st.set_page_config(
    page_title="🌐 Givens Rotation 3D Editor",
    page_icon="🌐",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .control-panel {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .matrix-display {
        background: #1e1e1e;
        color: #00ff41;
        font-family: 'Courier New', monospace;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 11px;
        line-height: 1.3;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌐 Givens Rotation 3D Editor</h1>
    <p>Áp dụng phép biến đổi Givens Rotation 3D để tạo hiệu ứng không gian ba chiều</p>
</div>
""", unsafe_allow_html=True)

# Upload ảnh
uploaded_file = st.file_uploader(
    "📁 Chọn ảnh để áp dụng Givens Rotation 3D", 
    type=['png', 'jpg', 'jpeg'],
    help="Hỗ trợ file PNG, JPG, JPEG"
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Ảnh gốc", use_column_width=True)
    except:
        st.error("Ảnh không hợp lệ")

# Phần còn lại của xử lý ảnh vẫn giữ nguyên ở dưới đây...
