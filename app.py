from flask import Flask, render_template_string, request, jsonify
import os
import cv2
import numpy as np
import numba as nb
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import uuid
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ImageRotation:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.channel = self.image.shape[2] if len(self.image.shape) > 2 else 1

        # Tạo tọa độ cho từng pixel trong ảnh
        y, x = np.meshgrid(range(self.height), range(self.width))
        z = np.zeros_like(x)
        self.pixels = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    def givens_matrix(self, i, j, theta):
        if i < 0 or i > 2 or j < 0 or j > 2 or i == j:
            raise ValueError("Indices i and j must be different and in the range [0, 2].")
        if i > j:
            i, j = j, i

        G = np.identity(3)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = s
        G[j, i] = -s
        return G

    def givens_matrix_2d(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def givens_rotation_2d(self, theta):
        try:
            theta_rad = np.deg2rad(theta)
            rotation_matrix = self.givens_matrix_2d(theta_rad)
            
            center_x, center_y = self.height // 2, self.width // 2
            
            # Tạo tọa độ pixel 2D
            y_coords, x_coords = np.meshgrid(range(self.width), range(self.height))
            coords = np.stack([x_coords.flatten() - center_x, y_coords.flatten() - center_y])
            
            # Xoay tọa độ
            rotated_coords = rotation_matrix @ coords
            rotated_coords[0] += center_x
            rotated_coords[1] += center_y
            
            # Tính kích thước ảnh đầu ra
            min_x, max_x = int(np.min(rotated_coords[0])), int(np.max(rotated_coords[0]))
            min_y, max_y = int(np.min(rotated_coords[1])), int(np.max(rotated_coords[1]))
            
            output_height = max_x - min_x + 1
            output_width = max_y - min_y + 1
            
            # Kiểm tra kích thước hợp lệ
            if output_height <= 0 or output_width <= 0:
                raise ValueError("Invalid output dimensions")
            
            # Điều chỉnh tọa độ để không âm
            rotated_coords[0] -= min_x
            rotated_coords[1] -= min_y
            
            # Tạo ảnh đầu ra
            if len(self.image.shape) == 3:
                output_image = np.zeros((output_height, output_width, self.image.shape[2]), dtype=self.image.dtype)
            else:
                output_image = np.zeros((output_height, output_width), dtype=self.image.dtype)
            
            # Map pixel values với interpolation cơ bản
            for i in range(len(coords[0])):
                orig_x = i // self.width
                orig_y = i % self.width
                new_x = int(rotated_coords[0][i])
                new_y = int(rotated_coords[1][i])
                
                if 0 <= new_x < output_height and 0 <= new_y < output_width:
                    if 0 <= orig_x < self.height and 0 <= orig_y < self.width:
                        if len(self.image.shape) == 3:
                            output_image[new_x, new_y] = self.image[orig_x, orig_y]
                        else:
                            output_image[new_x, new_y] = self.image[orig_x, orig_y]
            
            return output_image
        except Exception as e:
            logger.error(f"Error in 2D rotation: {str(e)}")
            raise

    def givens_rotation(self, pixels, alpha, theta, gamma):
        try:
            Ry = self.givens_matrix(1, 2, theta)
            Rx = self.givens_matrix(0, 2, alpha)
            Rz = self.givens_matrix(0, 1, gamma)

            pixels = self.centering_image(pixels)
            return pixels @ Rx @ Ry @ Rz
        except Exception as e:
            logger.error(f"Error in 3D rotation: {str(e)}")
            raise

    def centering_image(self, pixels):
        center = np.mean(pixels, axis=0)
        return pixels - center

    def projectPoints(self, object_point, camera_matrix):
        try:
            tvec = np.array([0, 0, self.focal_length * 1.5], dtype=np.float32)
            X_cam = object_point.T + tvec.reshape(3, 1)

            # Tránh chia cho 0
            Z_cam = np.where(X_cam[2] != 0, X_cam[2], 1e-8)
            
            x = X_cam[0] / Z_cam
            y = X_cam[1] / Z_cam

            fx = camera_matrix[0, 0]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            u = fx * x + cx
            v = fx * y + cy
            return np.stack([u, v], axis=1)
        except Exception as e:
            logger.error(f"Error in point projection: {str(e)}")
            raise

    def initialize_projection_parameters(self, max_angle):
        if max_angle < 0 or max_angle > 90:
            raise ValueError("max_angle must be in the range [0, 90].")

        max_dim = max(self.image.shape[0], self.image.shape[1])
        angle_factor = 1 + max_angle / 90
        self.focal_length = max_dim * 1.2 * angle_factor
        self.center = (self.height/2, self.width/2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

    def rotate_image_3d(self, alpha: int = 0, theta: int = 0, gamma: int = 0):
        try:
            # Kiểm tra tham số đầu vào
            if not all(-90 <= angle <= 90 for angle in [alpha, theta, gamma]):
                raise ValueError("Góc xoay phải nằm trong khoảng [-90, 90] độ.")

            self.alpha = np.deg2rad(alpha)
            self.theta = np.deg2rad(theta)
            self.gamma = np.deg2rad(gamma)

            rotated_pixels = self.givens_rotation(self.pixels.copy(), self.alpha, self.theta, self.gamma)
            self.initialize_projection_parameters(np.max(np.abs([alpha, theta, gamma])))

            self.projected_points = self.projectPoints(rotated_pixels, self.camera_matrix).astype(int)
            
            # Kiểm tra projected_points có hợp lệ không
            if self.projected_points.size == 0:
                raise ValueError("No valid projected points")
                
            self.projected_points -= np.min(self.projected_points, axis=0)

            max_height = np.max(self.projected_points[:, 0]) + 1
            max_width = np.max(self.projected_points[:, 1]) + 1

            # Kiểm tra kích thước hợp lệ
            if max_height <= 0 or max_width <= 0:
                raise ValueError("Invalid output dimensions")

            if len(self.image.shape) > 2:
                self.output_image = np.full((max_height, max_width, self.image.shape[2]), 255, dtype=self.image.dtype)
            else:
                self.output_image = np.full((max_height, max_width), 255, dtype=self.image.dtype)

            self.output_image = assign_pixels_parallel(self.pixels, self.projected_points, self.image, self.output_image)
            return self.output_image
        except Exception as e:
            logger.error(f"Error in 3D rotation: {str(e)}")
            raise

@nb.njit(parallel=True)
def assign_pixels_parallel(pixels, projected_points, image, output_image):
    for i in nb.prange(len(pixels)):
        orig_x = int(pixels[i, 0])
        orig_y = int(pixels[i, 1])
        proj_x = projected_points[i, 0]
        proj_y = projected_points[i, 1]

        # Kiểm tra bounds
        if (0 <= orig_x < image.shape[0] and 0 <= orig_y < image.shape[1] and
            0 <= proj_x < output_image.shape[0] and 0 <= proj_y < output_image.shape[1]):
            
            if image.ndim == 3:
                for c in range(image.shape[2]):
                    output_image[proj_x, proj_y, c] = image[orig_x, orig_y, c]
            else:
                output_image[proj_x, proj_y] = image[orig_x, orig_y]

    return output_image

def adjust_brightness(image, brightness):
    """Điều chỉnh độ sáng của ảnh"""
    try:
        if brightness == 0:
            return image
        
        # Chuyển đổi sang float để tránh overflow
        image_float = image.astype(np.float32)
        
        # Điều chỉnh độ sáng
        adjusted = image_float + brightness
        
        # Đảm bảo giá trị pixel trong khoảng [0, 255]
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    except Exception as e:
        logger.error(f"Error in brightness adjustment: {str(e)}")
        raise

def image_to_base64(image_array):
    """Chuyển đổi numpy array thành base64 string"""
    try:
        if image_array is None or image_array.size == 0:
            raise ValueError("Invalid image array")
            
        if len(image_array.shape) == 3:
            image_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        else:
            image_pil = Image.fromarray(image_array, mode='L')
        
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Processing Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .content {
            padding: 30px;
        }

        .upload-section {
            margin-bottom: 30px;
            text-align: center;
        }

        .file-input {
            display: none;
        }

        .file-input-button {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .file-input-button:hover {
            transform: translateY(-2px);
        }

        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .control-panel {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #e9ecef;
        }

        .control-panel h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .control-group {
            margin-bottom: 15px;
        }

        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }

        .control-group input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }

        .control-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4CAF50;
            cursor: pointer;
        }

        .value-display {
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            margin-left: 10px;
            min-width: 50px;
            text-align: center;
        }

        .apply-button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
        }

        .apply-button:hover:not(:disabled) {
            transform: translateY(-2px);
        }

        .apply-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .image-container {
            text-align: center;
        }

        .image-container h4 {
            margin-bottom: 15px;
            color: #333;
        }

        .image-wrapper {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 20px;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fafafa;
        }

        .image-wrapper img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
        }

        .placeholder {
            color: #999;
            font-style: italic;
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .controls {
                grid-template-columns: 1fr;
            }
            .images {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🖼️ Image Processing Tool</h1>
            <p>Xoay ảnh Givens 2D/3D và điều chỉnh độ sáng</p>
        </div>

        <div class="content">
            <div class="upload-section">
                <input type="file" id="fileInput" class="file-input" accept="image/*">
                <button class="file-input-button" onclick="document.getElementById('fileInput').click()">
                    📁 Chọn ảnh để upload
                </button>
            </div>

            <div class="controls">
                <div class="control-panel">
                    <h3>🔄 Xoay 2D</h3>
                    <div class="control-group">
                        <label for="angle2d">Góc xoay:</label>
                        <input type="range" id="angle2d" min="-180" max="180" value="0">
                        <span class="value-display" id="angle2dValue">0°</span>
                    </div>
                    <button class="apply-button" id="apply2d" disabled>Áp dụng 2D</button>
                </div>

                <div class="control-panel">
                    <h3>🌐 Xoay 3D</h3>
                    <div class="control-group">
                        <label for="alpha">Alpha (X):</label>
                        <input type="range" id="alpha" min="-90" max="90" value="0">
                        <span class="value-display" id="alphaValue">0°</span>
                    </div>
                    <div class="control-group">
                        <label for="theta">Theta (Y):</label>
                        <input type="range" id="theta" min="-90" max="90" value="0">
                        <span class="value-display" id="thetaValue">0°</span>
                    </div>
                    <div class="control-group">
                        <label for="gamma">Gamma (Z):</label>
                        <input type="range" id="gamma" min="-90" max="90" value="0">
                        <span class="value-display" id="gammaValue">0°</span>
                    </div>
                    <button class="apply-button" id="apply3d" disabled>Áp dụng 3D</button>
                </div>

                <div class="control-panel">
                    <h3>💡 Độ sáng</h3>
                    <div class="control-group">
                        <label for="brightness">Độ sáng:</label>
                        <input type="range" id="brightness" min="-100" max="100" value="0">
                        <span class="value-display" id="brightnessValue">0</span>
                    </div>
                    <button class="apply-button" id="applyBrightness" disabled>Áp dụng độ sáng</button>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Đang xử lý...</p>
            </div>

            <div class="images">
                <div class="image-container">
                    <h4>Ảnh gốc</h4>
                    <div class="image-wrapper" id="originalImageWrapper">
                        <span class="placeholder">Chưa có ảnh</span>
                    </div>
                </div>
                <div class="image-container">
                    <h4>Ảnh đã xử lý</h4>
                    <div class="image-wrapper" id="processedImageWrapper">
                        <span class="placeholder">Chưa có ảnh</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFilename = null;

        // File upload
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentFilename = data.filename;
                    document.getElementById('originalImageWrapper').innerHTML = 
                        `<img src="${data.image}" alt="Original Image">`;
                    enableButtons();
                    resetControls();
                } else {
                    showError(data.error);
                }
            })
            .catch(error => {
                showError('Lỗi upload: ' + error.message);
            });
        });

        // Update value displays
        document.getElementById('angle2d').addEventListener('input', function() {
            document.getElementById('angle2dValue').textContent = this.value + '°';
        });

        document.getElementById('alpha').addEventListener('input', function() {
            document.getElementById('alphaValue').textContent = this.value + '°';
        });

        document.getElementById('theta').addEventListener('input', function() {
            document.getElementById('thetaValue').textContent = this.value + '°';
        });

        document.getElementById('gamma').addEventListener('input', function() {
            document.getElementById('gammaValue').textContent = this.value + '°';
        });

        document.getElementById('brightness').addEventListener('input', function() {
            document.getElementById('brightnessValue').textContent = this.value;
        });

        // Apply 2D rotation
        document.getElementById('apply2d').addEventListener('click', function() {
            if (!currentFilename) return;
            
            const angle = document.getElementById('angle2d').value;
            processImage({
                filename: currentFilename,
                type: '2d',
                angle: parseInt(angle)
            });
        });

        // Apply 3D rotation
        document.getElementById('apply3d').addEventListener('click', function() {
            if (!currentFilename) return;
            
            const alpha = document.getElementById('alpha').value;
            const theta = document.getElementById('theta').value;
            const gamma = document.getElementById('gamma').value;
            
            processImage({
                filename: currentFilename,
                type: '3d',
                alpha: parseInt(alpha),
                theta: parseInt(theta),
                gamma: parseInt(gamma)
            });
        });

        // Apply brightness
        document.getElementById('applyBrightness').addEventListener('click', function() {
            if (!currentFilename) return;
            
            const brightness = document.getElementById('brightness').value;
            processImage({
                filename: currentFilename,
                type: 'brightness',
                brightness: parseInt(brightness)
            });
        });

        function processImage(data) {
            showLoading(true);
            
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                showLoading(false);
                if (result.success) {
                    document.getElementById('processedImageWrapper').innerHTML = 
                        `<img src="${result.image}" alt="Processed Image">`;
                } else {
                    showError(result.error);
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Lỗi xử lý: ' + error.message);
            });
        }

        function enableButtons() {
            document.getElementById('apply2d').disabled = false;
            document.getElementById('apply3d').disabled = false;
            document.getElementById('applyBrightness').disabled = false;
        }

        function resetControls() {
            document.getElementById('angle2d').value = 0;
            document.getElementById('alpha').value = 0;
            document.getElementById('theta').value = 0;
            document.getElementById('gamma').value = 0;
            document.getElementById('brightness').value = 0;
            
            document.getElementById('angle2dValue').textContent = '0°';
            document.getElementById('alphaValue').textContent = '0°';
            document.getElementById('thetaValue').textContent = '0°';
            document.getElementById('gammaValue').textContent = '0°';
            document.getElementById('brightnessValue').textContent = '0';
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            document.querySelector('.content').insertBefore(errorDiv, document.querySelector('.controls'));
            
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            # Tạo tên file unique để tránh xung đột
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Đọc ảnh
            image = cv2.imread(filepath)
            if image is None:
                os.remove(filepath)  # Xóa file không hợp lệ
                return jsonify({'success': False, 'error': 'Cannot read image file'})
            
            # Resize ảnh nếu quá lớn
            max_size = 600
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                cv2.imwrite(filepath, image)
            
            image_base64 = image_to_base64(image)
            
            return jsonify({
                'success': True,
                'image': image_base64,
                'filename': filename
            })
        
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'success': False, 'error': f'Upload error: {str(e)}'})

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        filename = data.get('filename')
        process_type = data.get('type')
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        # Đọc ảnh
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'success': False, 'error': 'Cannot read image file'})
        
        processed_image = None
        
        if process_type == '2d':
            angle = data.get('angle', 0)
            rotation = ImageRotation(image)
            processed_image = rotation.givens_rotation_2d(angle)
            
        elif process_type == '3d':
            alpha = data.get('alpha', 0)
            theta = data.get('theta', 0)
            gamma = data.get('gamma', 0)
            
            rotation = ImageRotation(image)
            processed_image = rotation.rotate_image_3d(alpha, theta, gamma)
            
        elif process_type == 'brightness':
            brightness = data.get('brightness', 0)
            processed_image = adjust_brightness(image, brightness)
            
        else:
            return jsonify({'success': False, 'error': 'Invalid process type'})
        
        if processed_image is None or processed_image.size == 0:
            return jsonify({'success': False, 'error': 'Processing failed'})
        
        # Chuyển đổi sang base64
        image_base64 = image_to_base64(processed_image)
        
        return jsonify({
            'success': True,
            'image': image_base64
        })
    
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return jsonify({'success': False, 'error': f'Processing error: {str(e)}'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Kiểm tra các dependency cần thiết
        import cv2
        import numpy as np
        import numba as nb
        from PIL import Image
        
        logger.info("Starting Flask application...")
        logger.info("All dependencies loaded successfully")
        
        # Chạy app với debug mode cho development
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        print(f"Error: Missing required dependency - {str(e)}")
        print("Please install missing packages using pip:")
        print("pip install opencv-python numpy numba pillow flask")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error starting application: {str(e)}")
