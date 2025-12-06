import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CẤU HÌNH
# ==========================================
# Đường dẫn ảnh bạn muốn test
TEST_IMAGE_PATH = './output_easyocr_final/imgs_plate/175816_ID5_p1_TEXT_0NR0.jpg' 

# Khởi tạo EasyOCR
print(">>> Đang khởi tạo EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True) 

# ==========================================
# 2. PIPELINE XỬ LÝ ẢNH MỚI (V2 - Tối ưu cho ảnh mờ)
# ==========================================
def preprocess_license_plate_debug(image):
    """
    Xử lý ảnh và trả về từng bước để hiển thị Debug
    Chiến lược: Zoom -> Gray -> CLAHE -> Sharpen (Bỏ Threshold)
    """
    steps = {}
    
    # Bước 0: Ảnh gốc
    steps['1. Original'] = image
    
    # Bước 1: Phóng to ảnh (Zoom x4)
    # Dùng INTER_LANCZOS4 cho kết quả sắc nét hơn CUBIC
    h, w = image.shape[:2]
    image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    steps['2. Zoomed (Lanczos4)'] = image
    
    # Bước 2: Chuyển xám (Grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps['3. Grayscale'] = gray
    
    # Bước 3: Khử nhiễu bảo toàn cạnh (Bilateral Filter)
    # Giảm nhiễu hạt nhưng không làm mờ cạnh chữ
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Bước 4: Tăng tương phản thông minh (CLAHE)
    # Giúp chữ nổi bật khỏi nền mà không bị cháy sáng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(blur)
    steps['4. Contrast (CLAHE)'] = contrast
    
    # Bước 5: Làm sắc nét (Sharpening)
    # Thay vì Threshold làm đứt nét, ta dùng kernel làm nét để chữ đậm hơn
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(contrast, -1, kernel)
    steps['5. Final Sharpened'] = sharpened
    
    # Lưu ý: Ta trả về ảnh XÁM (Grayscale), EasyOCR đọc ảnh xám tốt hơn ảnh nhị phân vỡ nét
    return sharpened, steps

# ==========================================
# 3. CHẠY THỬ NGHIỆM
# ==========================================

def run_test():
    # Load ảnh
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Không tìm thấy file {TEST_IMAGE_PATH}.")
        # Tạo ảnh mẫu nếu không tìm thấy file
        img = np.zeros((60, 200, 3), dtype=np.uint8)
        cv2.putText(img, "29H1-123.45", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        img = cv2.imread(TEST_IMAGE_PATH)

    if img is None:
        print("Lỗi đọc ảnh! File có thể bị hỏng.")
        return

    # Chạy xử lý
    processed_img, debug_steps = preprocess_license_plate_debug(img)

    # Chạy OCR
    print(">>> Đang đọc OCR...")
    try:
        # allowlist: Chỉ cho phép các ký tự này xuất hiện (Số, Chữ hoa, dấu gạch, chấm)
        # paragraph=False: Đọc từng dòng đơn lẻ
        # canvas_size: Tăng kích thước vùng đệm để tránh mất chữ ở mép
        results = reader.readtext(processed_img, 
                                  detail=0, 
                                  allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-',
                                  paragraph=False,
                                  canvas_size=2048)
        
        raw_text = "".join(results)
        
        # Format text
        clean_text = raw_text.upper().replace(' ', '').replace('.', '').replace('-', '')
        
        print(f"\n=== KẾT QUẢ ===")
        print(f"Raw OCR : {raw_text}")
        print(f"Cleaned : {clean_text}")
        
    except Exception as e:
        print(f"Lỗi OCR: {e}")
        clean_text = "Error"

    # Hiển thị trực quan (Matplotlib)
    titles = list(debug_steps.keys())
    images = list(debug_steps.values())
    
    # Tính toán lưới hiển thị
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        
        # Chuyển BGR sang RGB để plt hiển thị đúng màu
        if len(images[i].shape) == 3:
            show_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cmap = None
        else:
            show_img = images[i]
            cmap = 'gray'
            
        plt.imshow(show_img, cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
        
    plt.suptitle(f"Quy trình xử lý V2 - Kết quả: {clean_text}", fontsize=16, color='red')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()