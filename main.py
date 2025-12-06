import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

# ==========================================
# 1. CẤU HÌNH VÀ KHỞI TẠO
# ==========================================

VIDEO_PATH = '1.mp4'

# Thư mục lưu trữ (Giữ cấu trúc 3 thư mục)
BASE_OUTPUT_DIR = "vi_pham_output"
DIR_FULL = os.path.join(BASE_OUTPUT_DIR, "imgs_full")
DIR_VEHICLE = os.path.join(BASE_OUTPUT_DIR, "imgs_vehicle")
DIR_PLATE = os.path.join(BASE_OUTPUT_DIR, "imgs_plate")

for d in [DIR_FULL, DIR_VEHICLE, DIR_PLATE]:
    os.makedirs(d, exist_ok=True)

# Load Models
print("Đang tải các model...")
# Lưu ý: Code ví dụ có ocr_model nhưng bạn chưa cung cấp file ocr weights 
# nên tôi chỉ dùng plate_model để detect và lưu ảnh biển số.
light_model = YOLO("./runs_light/runs/detect/train/weights/best.pt") 
vehicle_model = YOLO("./yolo11n.pt")
plate_model = YOLO("./runs_plate/runs/detect/yolo11_run/weights/best.pt")

# Cấu hình vùng vi phạm (Polygon)
VIOLATION_ZONE = np.array([
    [700, 600],   # Điểm trên trái
    [1200, 600],  # Điểm trên phải
    [1200, 700],  # Điểm dưới phải
    [500, 700]    # Điểm dưới trái
], np.int32)

# Cấu hình tối ưu
FRAME_SKIP = 3        # Xử lý 1 frame, bỏ qua 2 frame (Tăng tốc độ)
RESIZE_WIDTH = 1280   # Kích thước hiển thị

# Lưu lịch sử vị trí xe (Dictionary lưu list toạ độ theo track_id)
vehicle_positions = defaultdict(list)
violated_ids = set()

# ==========================================
# 2. CÁC HÀM HỖ TRỢ
# ==========================================

def get_center(box):
    """Tính tâm của bounding box (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = map(int, box)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_inside_zone(point, zone):
    """Kiểm tra điểm có nằm trong vùng Polygon không"""
    return cv2.pointPolygonTest(zone, point, False) >= 0

def draw_zone(image, zone, color):
    """Vẽ vùng vi phạm"""
    cv2.polylines(image, [zone], True, color, 2)
    # Tô màu bán trong suốt
    overlay = image.copy()
    cv2.fillPoly(overlay, [zone], color)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================

def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Lấy thông số video gốc
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tính toán kích thước hiển thị
    RESIZE_HEIGHT = int(RESIZE_WIDTH * orig_h / orig_w)
    
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Tạo frame hiển thị (copy để không vẽ đè lên frame gốc)
        display_frame = frame.copy()

        # ------------------------------------------------
        # A. XỬ LÝ ĐÈN GIAO THÔNG
        # ------------------------------------------------
        is_red_light = False
        # Detect đèn (Dùng frame gốc để chính xác nhất)
        light_results = light_model(frame, verbose=False, imgsz=1280, conf=0.2)[0]
        
        for box in light_results.boxes:
            lbl = light_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if "red" in lbl.lower() and conf > 0.4:
                is_red_light = True
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(display_frame, f"RED {conf:.2f}", (x1, y1-10), 0, 0.6, (0,0,255), 2)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Vẽ vùng vi phạm (Đỏ nếu đèn đỏ, Xanh nếu đèn xanh)
        zone_color = (0, 0, 255) if is_red_light else (0, 255, 0)
        draw_zone(display_frame, VIOLATION_ZONE, zone_color)

        # ------------------------------------------------
        # B. XỬ LÝ XE & VI PHẠM (Logic Tracking)
        # ------------------------------------------------
        if is_red_light:
            # Sử dụng mode track để ID xe ổn định
            results = vehicle_model.track(frame, persist=True, classes=[2,3,5,7], verbose=False)
            
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    tid = int(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Điểm chạm đất (giữa đáy xe)
                    bottom_center = (int((x1 + x2) / 2), y2)
                    
                    # Cập nhật lịch sử vị trí
                    vehicle_positions[tid].append(bottom_center)
                    if len(vehicle_positions[tid]) > 20: 
                        vehicle_positions[tid].pop(0)

                    # LOGIC KIỂM TRA VI PHẠM
                    # Xe chưa bị bắt lỗi AND điểm chạm nằm trong vùng cấm
                    if tid not in violated_ids:
                        if is_inside_zone(bottom_center, VIOLATION_ZONE):
                            
                            # Kiểm tra thêm hướng di chuyển (nếu cần): 
                            # if len(vehicle_positions[tid]) > 2 and vehicle_positions[tid][-1][1] < vehicle_positions[tid][-2][1]: ...
                            
                            timestamp = datetime.now().strftime("%H%M%S_%f")
                            print(f"!!! VI PHẠM: Xe ID {tid} lúc {timestamp}")
                            violated_ids.add(tid)

                            # --- LƯU BẰNG CHỨNG ---
                            base_name = f"{timestamp}_ID{tid}"
                            
                            # 1. Ảnh toàn cảnh
                            cv2.imwrite(f"{DIR_FULL}/{base_name}.jpg", frame)
                            
                            # 2. Ảnh xe
                            veh_img = frame[y1:y2, x1:x2]
                            if veh_img.size > 0:
                                cv2.imwrite(f"{DIR_VEHICLE}/{base_name}.jpg", veh_img)
                                
                                # 3. Detect & Lưu biển số
                                p_results = plate_model(veh_img, verbose=False)
                                p_cnt = 0
                                for pr in p_results:
                                    for pbox in pr.boxes:
                                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                                        p_img = veh_img[py1:py2, px1:px2]
                                        if p_img.size > 0:
                                            p_cnt += 1
                                            cv2.imwrite(f"{DIR_PLATE}/{base_name}_p{p_cnt}.jpg", p_img)

                    # Vẽ hiển thị
                    color = (0, 0, 255) if tid in violated_ids else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"ID:{tid}", (x1, y1-5), 0, 0.6, color, 2)
                    cv2.circle(display_frame, bottom_center, 4, (255, 0, 255), -1)

        # ------------------------------------------------
        # C. HIỂN THỊ (RESIZE)
        # ------------------------------------------------
        # Resize frame hiển thị để nhẹ máy (Frame xử lý vẫn giữ nguyên gốc)
        view_frame = cv2.resize(display_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        
        cv2.putText(view_frame, f"Light: {'RED' if is_red_light else 'GREEN'}", (20, 40), 0, 1, zone_color, 2)
        cv2.putText(view_frame, f"Violations: {len(violated_ids)}", (20, 80), 0, 1, (0, 165, 255), 2)
        
        cv2.imshow('Traffic Violation Detection', view_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Hoàn tất. Kết quả lưu tại {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    process_video()