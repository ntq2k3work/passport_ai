import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

def detect_head_region(image_path, model_path="models/best_re_final.onnx", conf_threshold=0.5):
    """
    Nhận diện vùng đầu trong ảnh bằng ONNX model
    Returns: bounding box [x1, y1, x2, y2] hoặc None nếu không tìm thấy
    """
    try:
        # Load ONNX model
        opt_session = ort.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=EP_list)
        
        # Lấy thông tin input/output
        model_inputs = session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        input_height, input_width = input_shape[2:]
        
        model_output = session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]
        
        # Đọc và preprocess ảnh
        img = cv2.imread(image_path)
        image_height, image_width = img.shape[:2]
        
        # Resize và normalize ảnh
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (input_width, input_height))
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        
        # Chạy inference
        outputs = session.run(output_names, {input_names[0]: input_tensor})[0]
        predictions = np.squeeze(outputs).T
        
        # Filter theo confidence threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]
        
        if len(predictions) == 0:
            return None
            
        # Lấy detection có confidence cao nhất
        best_idx = np.argmax(scores)
        best_prediction = predictions[best_idx]
        
        # Lấy bounding box
        boxes = best_prediction[:4]
        
        # Rescale box về tọa độ ảnh gốc
        input_shape_array = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape_array, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)
        
        # Chuyển từ [x, y, w, h] sang [x1, y1, x2, y2]
        x, y, w, h = boxes
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2
        
        return [x1, y1, x2, y2]
        
    except Exception as e:
        print(f"Lỗi nhận diện: {e}")
        return None

def handle_convert_visa(
    input_image_path, output_image_path, bbox, photo_type='2x3', dpi=300, head_ratio=0.75
):
    """
    Căn chỉnh ảnh visa dựa trên vùng đầu đã nhận diện:
    - Nhận ảnh gốc và bounding box vùng đầu
    - Mở rộng vùng cắt để bao gồm cả thân
    - Scale để vùng đầu chiếm ~75% chiều cao ảnh chuẩn
    - Căn giữa trong nền trắng chuẩn
    
    input_image_path: ảnh gốc
    output_image_path: nơi lưu ảnh kết quả
    bbox: bounding box vùng đầu [x1, y1, x2, y2]
    photo_type: '2x3' hoặc '3x4'
    dpi: dots per inch (mặc định 300)
    head_ratio: tỉ lệ chiều cao đầu trên tổng chiều cao ảnh (mặc định 0.75)
    """
    # Định nghĩa kích thước chuẩn theo mm
    photo_sizes = {
        '2x3': (20, 30),  # mm
        '3x4': (30, 40),
    }
    if photo_type not in photo_sizes:
        raise ValueError('photo_type phải là "2x3" hoặc "3x4"')
    width_mm, height_mm = photo_sizes[photo_type]

    # Chuyển đổi sang pixel
    def mm_to_px(mm):
        inch = mm / 25.4
        return int(round(inch * dpi))
    width_px = mm_to_px(width_mm)
    height_px = mm_to_px(height_mm)

    # Đọc ảnh gốc
    img = cv2.imread(input_image_path)
    h_img, w_img = img.shape[:2]

    # Xử lý bounding box
    if len(bbox) == 4:
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
        else:  # [x, y, w, h]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
    else:
        raise ValueError("bbox phải có 4 giá trị")

    # Tính kích thước vùng đầu
    head_width = x2 - x1
    head_height = y2 - y1
    
    # Mở rộng vùng cắt để bao gồm cả thân
    # Ước tính chiều cao thân = 2-3 lần chiều cao đầu
    body_height_ratio = 2.5  # Tỷ lệ thân/đầu
    total_height = int(head_height * (1 + body_height_ratio))
    
    # Tính vị trí cắt mới
    # Đầu ở 1/3 trên của vùng cắt
    head_y_ratio = 0.33  # Vị trí đầu trong vùng cắt
    crop_y1 = max(0, y1 - int(head_height * head_y_ratio))
    crop_y2 = min(h_img, crop_y1 + total_height)
    
    # Căn giữa theo chiều ngang
    crop_x1 = max(0, x1 - (head_width // 2))
    crop_x2 = min(w_img, crop_x1 + head_width * 2)  # Chiều rộng gấp đôi đầu
    
    # Cắt vùng người (đầu + thân)
    person_region = img[crop_y1:crop_y2, crop_x1:crop_x2]
    h_person, w_person = person_region.shape[:2]

    # Tính chiều cao đầu mong muốn trong ảnh chuẩn
    target_head_height = int(height_px * head_ratio)
    
    # Tính tỷ lệ scale để đạt được chiều cao đầu mong muốn
    # Vị trí đầu trong vùng cắt
    head_in_crop_y = int(h_person * head_y_ratio)
    head_in_crop_height = int(head_height * (h_person / total_height))
    
    # Scale để đầu có chiều cao mong muốn
    scale = target_head_height / head_in_crop_height
    new_w = int(w_person * scale)
    new_h = int(h_person * scale)
    
    # Kiểm tra và điều chỉnh scale nếu ảnh quá lớn
    if new_w > width_px or new_h > height_px:
        # Tính scale để vừa với nền
        scale_w = width_px / w_person
        scale_h = height_px / h_person
        scale = min(scale_w, scale_h)
        new_w = int(w_person * scale)
        new_h = int(h_person * scale)
        print(f"Điều chỉnh scale từ {target_head_height / head_in_crop_height:.3f} xuống {scale:.3f} để vừa với nền")
    
    # Resize vùng người
    person_resized = cv2.resize(person_region, (new_w, new_h))

    # Tạo nền trắng chuẩn
    result = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    # Căn giữa vùng người trên nền trắng
    start_x = (width_px - new_w) // 2
    start_y = (height_px - new_h) // 2
    result[start_y:start_y+new_h, start_x:start_x+new_w] = person_resized

    # Lưu ảnh kết quả
    cv2.imwrite(output_image_path, result)
    print(f"Đã lưu ảnh chuẩn tại: {output_image_path}")
    print(f"Kích thước nền: {width_px}x{height_px} px")
    print(f"Vùng đầu gốc: ({x1},{y1}) -> ({x2},{y2})")
    print(f"Vùng cắt người: ({crop_x1},{crop_y1}) -> ({crop_x2},{crop_y2})")
    print(f"Kích thước người resize: {new_w}x{new_h} px")
    print(f"Tỉ lệ đầu/ảnh: {target_head_height/height_px*100:.1f}%")
    print(f"Vị trí đặt: ({start_x}, {start_y})")
    pass

def mm_to_px(mm, dpi=300):
    """
    Chuyển đổi milimét sang pixel dựa trên DPI.
    mm: chiều dài milimét
    dpi: dots per inch (mặc định 300)
    """
    inch = mm / 25.4
    px = int(round(inch * dpi))
    return px

# Ví dụ sử dụng cho các loại ảnh visa chuẩn quốc tế
visa_photo_sizes = [
    {"name": "2x3 cm", "width_mm": 20, "height_mm": 30},
    {"name": "3x4 cm", "width_mm": 30, "height_mm": 40},
]

def print_visa_photo_pixel_sizes(dpi=300):
    print(f"Kích thước pixel cho các loại ảnh visa ở {dpi} DPI:")
    for size in visa_photo_sizes:
        w_px = mm_to_px(size["width_mm"], dpi)
        h_px = mm_to_px(size["height_mm"], dpi)
        print(f"{size['name']}: {w_px}x{h_px} px")

# Nếu muốn chạy thử:
if __name__ == "__main__":
    print_visa_photo_pixel_sizes()
