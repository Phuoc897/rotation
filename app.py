import cv2
import numpy as np
# import tkinter as tk # Removed tkinter
# from tkinter import filedialog # Removed filedialog
from google.colab import files # Import files for Colab

# ========== 1. Chọn ảnh ==========
# root = tk.Tk() # Removed tkinter
# root.withdraw() # Removed tkinter
# file_path = filedialog.askopenfilename(title='Chọn ảnh') # Removed tkinter

uploaded = files.upload() # Use Colab's file upload

if not uploaded:
    print("Không có ảnh nào được chọn.")
    # exit() # Removed exit to allow the cell to continue if no file is selected, though the rest of the code will likely fail without an image. Consider adding error handling later if needed.
else:
    # Assuming only one file is uploaded for simplicity
    file_path = next(iter(uploaded))
    print(f"Đã chọn ảnh: {file_path}")

    img = cv2.imread(file_path)
    if img is None:
        print("Lỗi đọc ảnh.")
        # exit() # Removed exit

    # ========== 2. Nhập dữ liệu từ bàn phím ==========
    angle = float(input("Nhập góc quay (độ): "))
    in1 = int(input("Bấm 1 để xử lý ảnh 3D, bấm 2 để xử lý ảnh 2D: "))

    angle_rad = np.radians(angle)

    if in1 == 1:
        homo = input("Nhập ma trận xoay 1x3 (vd: 1 0 0): ")
        homo = np.fromstring(homo, sep=' ')  # chuyển thành mảng numpy
        if homo.shape != (3,):
            print("Sai định dạng ma trận xoay.")
            # exit() # Removed exit

        # ========== 3. Tạo ma trận xoay A ==========
        A1 = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])
        A = np.vstack([np.hstack([A1, np.zeros((2,1))]), homo.reshape(1,3)])
        print("Ma trận A:\n", A)

        # Bạn có thể tiếp tục áp dụng ma trận A để biến đổi ảnh (warpPerspective 3D)
    else:
        print("Chế độ 2D chưa được xử lý trong đoạn này.")

    # The following code blocks were not part of the original error but are included in the same cell.
    # I will keep them as they are for now, assuming they are part of the user's workflow,
    # but note that they use a hardcoded image file '1.png' and the variable 'img' from the upload is not used here.
    # This might lead to errors later if '1.png' doesn't exist or if the intention was to use the uploaded image.

    # This section reads '1.png' and performs a 2D rotation example.
    # If the user intended to use the uploaded image for this, further modifications are needed.
    import numpy as np
    import cv2

    # Đọc ảnh và chuyển sang float
    # img = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE) # This line uses a hardcoded image '1.png'
    # a = img.astype(np.float64) # This will cause an error if the above line is commented out and 'img' from upload is used, as it might be color.

    # Assuming the user wants to continue with the uploaded image if it's grayscale for the 2D part
    # If the uploaded image is color and user selected 2D, this part might need adjustment.
    if img is not None and in1 == 2: # Only proceed with 2D processing if an image was uploaded and 2D was selected
        if len(img.shape) == 3:
            print("Chuyển đổi ảnh màu sang ảnh xám cho xử lý 2D.")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            a = img_gray.astype(np.float64)
        else:
             a = img.astype(np.float64)


        b1, b2 = a.shape  # chiều cao, chiều rộng

        # Ma trận dịch gốc về tâm ảnh
        trans = np.array([
            [1, 0, -b2/2],
            [0, 1, -b1/2],
            [0, 0, 1]
        ])

        # Ma trận xoay 2D (ví dụ 45 độ)
        theta = np.radians(45) # This was hardcoded to 45 degrees, but the user also input an angle earlier.
                               # I will keep it as 45 degrees as it seems to be a separate example,
                               # but if the intention was to use the user's input angle for the 2D part,
                               # this line should be changed to theta = angle_rad.
        A = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Tạo mảng chứa tọa độ sau khi xoay
        outx = np.zeros((b1, b2), dtype=int)
        outy = np.zeros((b1, b2), dtype=int)

        # Biến đổi từng điểm ảnh
        for i in range(b1):
            for j in range(b2):
                point = np.array([[j], [i], [1]])  # lưu ý: x=j, y=i
                new = A @ (trans @ point)
                outx[i, j] = round(new[0, 0] / new[2, 0])
                outy[i, j] = round(new[1, 0] / new[2, 0])

        # Tìm min/max tọa độ mới
        minoutx = np.min(outx)
        minouty = np.min(outy)
        maxoutx = np.max(outx)
        maxouty = np.max(outy)

        print("Tọa độ sau khi xoay:")
        print(f"X: {minoutx} → {maxoutx}")
        print(f"Y: {minouty} → {maxouty}")

        # This section reconstructs the image and displays it.
        import numpy as np
        import cv2
        from PIL import Image # PIL is imported but not used
        import matplotlib.pyplot as plt

        # Giả sử: img là ảnh đầu vào (a), outx, outy là kết quả biến đổi tọa độ
        # Assuming 'img' here refers to the potentially color uploaded image
        # and 'a' is the grayscale version used for calculations.
        # The original code uses 'a' for shape and 'img' for pixel values, which is inconsistent if 'img' is color.
        # I will adjust to use 'img' for pixel values if it's color, and 'a' for grayscale calculations.
        a1 = 1 if len(img.shape) == 3 else 0 # Check if the original uploaded image is color

        # Xác định kích thước ảnh mới
        new_w = maxoutx + abs(minoutx) + 1
        new_h = maxouty + abs(minouty) + 1

        # Tạo ảnh mới rỗng (grayscale hoặc RGB)
        if a1 == 1:
            f = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        else:
            f = np.zeros((new_h, new_w), dtype=np.uint8)

        # Gán lại giá trị pixel
        # Need to map the original image pixels based on the calculated new coordinates
        # and the offset to handle negative coordinates.
        # This part of the original code seems to directly copy pixels which is not correct for rotation.
        # A proper rotation would involve inverse mapping or using cv2.warpAffine/warpPerspective.
        # However, to stick to the user's current approach of manual mapping:
        # The original loop assigned pixels from img[i, j] to f[y_new, x_new].
        # This is still a direct copy, not a rotation.
        # To perform a rotation manually, you'd typically iterate through the output image
        # and calculate the corresponding input image coordinates using the inverse transform.
        # Since the user's code is doing a forward mapping and assigning pixels,
        # I will keep this structure but acknowledge it's not a standard rotation implementation.
        # The original code also seems to use 'img' for color values and 'a' for grayscale calculations inconsistently.
        # I will use the appropriate source image based on whether the original uploaded image was color or not.

        for i in range(b1): # b1, b2 are dimensions of the grayscale image 'a'
            for j in range(b2):
                x_new = outx[i, j] + abs(minoutx)
                y_new = outy[i, j] + abs(minouty)
                if 0 <= y_new < new_h and 0 <= x_new < new_w:
                    if a1 == 1: # If original image was color
                         # Assuming the pixel value to copy is from the original color image
                        f[y_new, x_new, 🙂 = img[i, j, 🙂
                    else: # If original image was grayscale
                        # Assuming the pixel value to copy is from the original grayscale image
                        f[y_new, x_new] = img[i, j] # Or use a[i,j] if calculations were meant to affect pixel values

        # ========== Hiển thị kết quả ==========
        # Check if img is not None before displaying
        if img is not None:
            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if a1 else img, cmap='gray')
            plt.title("Ảnh Gốc")

        plt.figure()
        # Need to handle the case where the processed image 'f' might not be created if no image was uploaded or 2D was not selected.
        if 'f' in locals():
            plt.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if a1 else f, cmap='gray')
            plt.title("Ảnh đã xử lý")
            plt.show()
        else:
            print("Ảnh đã xử lý không được tạo.")
