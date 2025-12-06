import cv2
import os
import numpy as np
import re
import easyocr
import csv
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO
# ==========================================
VIDEO_PATH = 'traffic_simulation_output.mp4'
BASE_OUTPUT_DIR = "output_standard"
CSV_FILE = "traffic_violations.csv"

# Regex biển số Việt Nam
VN_PLATE_PATTERN = r"^[0-9]{2}[A-Z]{1}\d{3,4}$|^[0-9]{2}[A-Z]{1}\d{3}\.\d{2}$"

# Thư mục lưu ảnh
DIR_IMGS_FULL = os.path.join(BASE_OUTPUT_DIR, "imgs_full")
DIR_IMGS_VEH = os.path.join(BASE_OUTPUT_DIR, "imgs_vehicles")
DIR_IMGS_PLATE_RAW = os.path.join(BASE_OUTPUT_DIR, "imgs_plate_raw")
DIR_IMGS_PLATE_PROC = os.path.join(BASE_OUTPUT_DIR, "imgs_plate_processed")

for d in [DIR_IMGS_FULL, DIR_IMGS_VEH, DIR_IMGS_PLATE_RAW, DIR_IMGS_PLATE_PROC]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["Timestamp", "Vehicle_ID", "Type", "License_Plate", "Confidence", "Image_Frame", "Image_Vehicle", "Image_Plate_Raw", "Image_Plate_Processed"])

print(">>> [INIT] Khởi tạo EasyOCR...")
reader = easyocr.Reader(['vi', 'en'], gpu=True)

print(">>> [INIT] Tải model YOLO...")
try:
    light_model = YOLO("./runs_light/runs/detect/train/weights/best.pt")
    vehicle_model = YOLO("./yolo11n.pt")
    plate_model = YOLO("./runs_plate/runs/detect/yolo11_run/weights/best.pt")
except Exception as e:
    print(f"LỖI LOAD MODEL: {e}")
    exit()

# ==========================================
# 2. XỬ LÝ ẢNH (preprocess + OCR)
# ==========================================
def preprocess_plate_standard(img):
    if img is None or img.size == 0: return None
    img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    try:
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    except Exception:
        img = cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return gray

def sort_ocr_results(results):
    return sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))

def clean_plate_text(text):
    if text is None: return ""
    text = text.upper()
    text = text.replace(' ', '').replace('.', '').replace('-', '')
    replacements = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    cleaned = "".join([replacements.get(c, c) for c in text if c.isalnum()])
    return cleaned

def perform_ocr_standard(image):
    processed = preprocess_plate_standard(image)
    if processed is None: return "Error", None
    try:
        results = reader.readtext(processed, detail=1, paragraph=False, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not results: return "Unknown", processed
        
        sorted_res = sort_ocr_results(results)
        lines = []
        curr_y = None
        curr_line = ""
        for box, txt, conf in sorted_res:
            ctext = clean_plate_text(txt)
            if len(ctext) == 0: continue
            cy = sum([p[1] for p in box]) / 4.0
            if curr_y is None:
                curr_y = cy
                curr_line = ctext
            elif abs(cy - curr_y) < 25:
                curr_line += ctext
            else:
                lines.append(curr_line)
                curr_line = ctext
                curr_y = cy
        if curr_line: lines.append(curr_line)
        return "".join(lines), processed
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Error", processed

# ==========================================
# 3. LOGIC TRACKING & VI PHẠM
# ==========================================
VIOLATION_ZONE = np.array([[200, 700], [1600, 700], [1800, 850], [200, 850]], np.int32)
violated_ids = set()
recent_violations_log = []
violated_info = {} 
DUPLICATE_TIME = 3.0
IOU_THRESHOLD = 0.5

def is_inside_zone(point):
    return cv2.pointPolygonTest(VIOLATION_ZONE, point, False) >= 0

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter/union if union > 0 else 0

def is_duplicate(box, now):
    global recent_violations_log
    recent_violations_log = [log for log in recent_violations_log if (now - log[1]).total_seconds() < DUPLICATE_TIME]
    for old_box, _ in recent_violations_log:
        if calculate_iou(box, old_box) > IOU_THRESHOLD:
            return True
    return False

def draw_info(frame, zone_color, is_red):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [VIOLATION_ZONE], zone_color)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [VIOLATION_ZONE], True, zone_color, 2)
    text = "RED LIGHT" if is_red else "GREEN"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 3)

# ==========================================
# 4. MAIN LOOP
# ==========================================
def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Không mở được video")
        return

    WINDOW_NAME = "Traffic Standard System"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_cnt = 0
    paused = False 
    display = None 
    
    print("--- BẮT ĐẦU CHẠY ---")
    print(">>> Nhấn phím 'P' hoặc 'SPACE' để Tạm Dừng/Tiếp Tục")
    print(">>> Nhấn phím 'Q' hoặc tắt cửa sổ để Thoát")

    while True:
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Cửa sổ đã bị đóng. Dừng chương trình.")
                break
        except Exception: pass

        if not paused:
            if cap.isOpened():
                success, frame = cap.read()
                if not success: 
                    print("Hết video.")
                    break

                frame_cnt += 1
                if frame_cnt % 3 != 0: continue

                display = frame.copy()
                curr_time = datetime.now()

                # --- 1. Detect Light ---
                is_red = False
                roi = frame[0:400, 0:600]
                try:
                    for r in light_model(roi, verbose=False, conf=0.4):
                        for b in r.boxes:
                            cls_idx = int(b.cls[0])
                            name = light_model.names.get(cls_idx, "")
                            if 'red' in name.lower(): is_red = True
                except Exception: pass

                zone_color = (0,0,255) if is_red else (0,255,0)
                draw_info(display, zone_color, is_red)

                # --- 2. Track Vehicle ---
                try:
                    results = vehicle_model.track(frame, persist=True, verbose=False, classes=[2,3,5,7], tracker="bytetrack.yaml")
                except Exception as e:
                    results = []

                if results and len(results) > 0 and getattr(results[0].boxes, "id", None) is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy()

                    for box, tid, cls in zip(boxes, ids, clss):
                        x1, y1, x2, y2 = map(int, box)
                        tid = int(tid)
                        center = (int((x1+x2)/2), int((y1+y2)/2))
                        curr_box = [x1, y1, x2, y2]
                        
                        v_type = vehicle_model.names[int(cls)] if int(cls) in vehicle_model.names else str(int(cls))

                        # ==========================
                        # LOGIC VI PHẠM 
                        # ==========================
                        if is_red and tid not in violated_ids:
                            if is_inside_zone(center) and not is_duplicate(curr_box, curr_time):
                                violated_ids.add(tid)
                                recent_violations_log.append((curr_box, curr_time))
                                
                                # Lưu ý: Tôi đã bỏ dòng print ở đây để chờ có biển số mới in

                                ts = curr_time.strftime("%Y%m%d_%H%M%S")
                                base_name = f"{ts}_ID{tid}"
                                full_path = os.path.join(DIR_IMGS_FULL, f"{base_name}.jpg")
                                cv2.imwrite(full_path, frame)

                                # Crop logic
                                y1c, y2c = max(0, y1), max(0, y2)
                                x1c, x2c = max(0, x1), max(0, x2)
                                veh_w = x2c - x1c
                                veh_h = y2c - y1c
                                veh_img = frame[y1c:y2c, x1c:x2c]
                                veh_path = ""
                                
                                best_text = "Unknown"
                                plate_raw_path = ""
                                plate_proc_path = ""
                                plate_relative_box = None 

                                if veh_img is not None and veh_img.size > 0:
                                    veh_path = os.path.join(DIR_IMGS_VEH, f"{base_name}_veh.jpg")
                                    cv2.imwrite(veh_path, veh_img)

                                    try: p_res = plate_model(veh_img, verbose=False, conf=0.3)
                                    except: p_res = []

                                    plate_boxes = []
                                    for pr in p_res:
                                        for pb in pr.boxes:
                                            try:
                                                coords = pb.xyxy[0].cpu().numpy()
                                                confp = float(pb.conf[0].cpu().numpy())
                                                px1, py1, px2, py2 = map(int, coords)
                                                plate_boxes.append((confp, px1, py1, px2, py2))
                                            except: continue
                                    
                                    if plate_boxes:
                                        plate_boxes.sort(reverse=True, key=lambda x: x[0])
                                        _, px1, py1, px2, py2 = plate_boxes[0]
                                        px1, py1 = max(0, px1), max(0, py1)
                                        px2, py2 = min(veh_img.shape[1], px2), min(veh_img.shape[0], py2)
                                        
                                        # Tính tọa độ tương đối
                                        if veh_w > 0 and veh_h > 0:
                                            rel_x = px1 / veh_w
                                            rel_y = py1 / veh_h
                                            rel_w = (px2 - px1) / veh_w
                                            rel_h = (py2 - py1) / veh_h
                                            plate_relative_box = (rel_x, rel_y, rel_w, rel_h)

                                        if px2 - px1 > 0 and py2 - py1 > 0:
                                            plate_crop = veh_img[py1:py2, px1:px2]
                                            plate_raw_path = os.path.join(DIR_IMGS_PLATE_RAW, f"{base_name}_plate_raw.jpg")
                                            cv2.imwrite(plate_raw_path, plate_crop)

                                            text, proc_img = perform_ocr_standard(plate_crop)
                                            if proc_img is not None:
                                                plate_proc_path = os.path.join(DIR_IMGS_PLATE_PROC, f"{base_name}_plate_proc.jpg")
                                                try: cv2.imwrite(plate_proc_path, proc_img)
                                                except: pass

                                            cleaned = clean_plate_text(text)
                                            if re.match(VN_PLATE_PATTERN, cleaned): best_text = cleaned
                                            elif len(cleaned) >= 4: best_text = cleaned
                                            else: best_text = "Unknown"
                                    else:
                                        # Fallback
                                        hh, ww = veh_img.shape[:2]
                                        band = veh_img[int(hh*0.6):min(hh, int(hh*0.95)), int(ww*0.05):int(ww*0.95)]
                                        if band is not None and band.size > 0:
                                            t, dbg = perform_ocr_standard(band)
                                            if dbg is not None:
                                                try: cv2.imwrite(os.path.join(DIR_IMGS_PLATE_PROC, f"{base_name}_fb_proc.jpg"), dbg)
                                                except: pass
                                            if t and len(t) >= 4: best_text = t
                                
                                # [UPDATE] In thông báo đầy đủ SAU KHI đã có best_text (biển số)
                                time_str = curr_time.strftime("%H:%M:%S %d/%m/%Y")
                                print(f"\n!!! VI PHẠM: ID {tid} ({v_type}) | Biển số: {best_text} | Thời gian: {time_str}")

                                # Lưu thông tin
                                violated_info[tid] = {
                                    "type": v_type,
                                    "plate": best_text,
                                    "rel_box": plate_relative_box
                                }

                                with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                                    csv.writer(f).writerow([curr_time.strftime("%Y-%m-%d %H:%M:%S"), tid, v_type, best_text, "0.00", full_path, veh_path, plate_raw_path, plate_proc_path])

                        # ==========================
                        # LOGIC VẼ HÌNH
                        # ==========================
                        color = (0,0,255) if tid in violated_ids else (0,255,0)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                        if tid in violated_info:
                            info = violated_info[tid]
                            label = f"ID:{tid} | {info['type']} | {info['plate']}"
                            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display, (x1, y1 - 25), (x1 + t_size[0], y1), color, -1)
                            cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            if info["rel_box"] is not None:
                                rx, ry, rw, rh = info["rel_box"]
                                curr_veh_w = x2 - x1
                                curr_veh_h = y2 - y1
                                
                                pbx1 = int(x1 + rx * curr_veh_w)
                                pby1 = int(y1 + ry * curr_veh_h)
                                pbx2 = int(pbx1 + rw * curr_veh_w)
                                pby2 = int(pby1 + rh * curr_veh_h)
                                
                                cv2.rectangle(display, (pbx1, pby1), (pbx2, pby2), (255, 255, 0), 2) 
                        else:
                            cv2.putText(display, f"ID: {tid}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if display is not None:
            if paused:
                overlay = display.copy()
                cv2.rectangle(overlay, (50, 200), (450, 300), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
                cv2.putText(display, "TAM DUNG (PAUSED)", (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p') or key == ord(' '): paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn tất.")

if __name__ == "__main__":
    process_video()