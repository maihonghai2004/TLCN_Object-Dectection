# reset_models.py
import torch
import gc

def reset_all_models():
    """
    Reset toàn bộ model đang nằm trong RAM/GPU.
    Xóa biến global, giải phóng bộ nhớ Python + GPU cache.
    """

    print("🔄 Reset toàn bộ model...")

    # Danh sách biến global phổ biến mà các dự án hay sử dụng
    global_vars = [
        "GLOBAL_ESRGAN_MODEL",
        "GLOBAL_ESRGAN_NET",
        "GLOBAL_EASYOCR_READER",
        "GLOBAL_CLAHE_MODEL",
        "GLOBAL_YOLO_MODEL",
        "GLOBAL_ZERO_DCE",
        "GLOBAL_BM3D_MODEL"
    ]

    for var in global_vars:
        if var in globals():
            try:
                del globals()[var]
                print(f"   ➤ Deleted: {var}")
            except:
                pass

    # Thu gom rác Python
    freed = gc.collect()
    print(f"🧹 GC collected: {freed} objects")

    # Giải phóng VRAM GPU nếu có
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("🔥 GPU VRAM cleared")

    print("✔ Reset hoàn tất — RAM/GPU đã sạch.")

# Nếu chạy file trực tiếp
if __name__ == "__main__":
    reset_all_models()
