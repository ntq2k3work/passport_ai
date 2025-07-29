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

def detect_person_region(image_path, model_path="models/best_re_final.onnx", conf_threshold=0.5):
    """
    Nhận diện vùng người trong ảnh bằng ONNX model
    Returns: bounding box [x1, y1, x2, y2] của toàn bộ người hoặc None nếu không tìm thấy
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

def extract_head_region_from_person(person_bbox, image_shape):
    """
    Trích xuất vùng đầu từ vùng người đã nhận diện
    person_bbox: [x1, y1, x2, y2] của toàn bộ người
    image_shape: (height, width) của ảnh
    Returns: [x1, y1, x2, y2] của vùng đầu
    """
    x1, y1, x2, y2 = person_bbox
    person_height = y2 - y1
    person_width = x2 - x1
    
    # Ước tính vùng đầu chiếm khoảng 1/3 chiều cao của người
    head_height_ratio = 0.33
    head_width_ratio = 0.8  # Đầu thường rộng hơn thân một chút
    
    # Tính kích thước vùng đầu
    head_height = int(person_height * head_height_ratio)
    head_width = int(person_width * head_width_ratio)
    
    # Tính vị trí vùng đầu (ở phía trên của người)
    head_x1 = x1 + (person_width - head_width) // 2
    head_y1 = y1
    head_x2 = head_x1 + head_width
    head_y2 = head_y1 + head_height
    
    return [head_x1, head_y1, head_x2, head_y2]

def handle_convert_visa(
    input_image_path, output_image_path, bbox, 
    size_px=None, size_mm=None, dpi=300,
    top_margin_mm=0, bottom_margin_mm=0, left_margin_mm=0, right_margin_mm=0,
    background_color='white'
):
    """
    Căn chỉnh ảnh visa dựa trên vùng đầu đã nhận diện với các tham số mới:
    
    input_image_path: ảnh gốc
    output_image_path: nơi lưu ảnh kết quả
    bbox: bounding box vùng đầu [x1, y1, x2, y2]
    size_px: tuple (width_px, height_px) - kích thước pixel đầu ra
    size_mm: tuple (width_mm, height_mm) - kích thước mm
    dpi: dots per inch (mặc định 300)
    top_margin_mm: khoảng cách từ đỉnh đầu tới lề trên của ảnh kích thước đã chọn (mm)
    bottom_margin_mm: khoảng cách từ cằm tới lề dưới của ảnh kích thước đã chọn (mm)
    left_margin_mm: khoảng cách từ mặt tới lề trái của ảnh kích thước đã chọn (mm)
    right_margin_mm: khoảng cách từ mặt tới lề phải của ảnh kích thước đã chọn (mm)
    background_color: màu nền ('white', 'light_blue', etc.)
    """
    
    # Xác định kích thước đầu ra
    if size_px:
        width_px, height_px = size_px
    elif size_mm:
        width_mm, height_mm = size_mm
        # Chuyển đổi từ mm sang pixel
        def mm_to_px(mm):
            inch = mm / 25.4
            return int(round(inch * dpi))
        width_px = mm_to_px(width_mm)
        height_px = mm_to_px(height_mm)
    else:
        raise ValueError("Phải cung cấp size_px hoặc size_mm")

    # Xác định màu nền
    background_colors = {
        'white': (255, 255, 255),
        'light_blue': (173, 216, 230),
        'blue': (0, 0, 255),
        'red': (255, 0, 0),
        'green': (0, 255, 0)
    }
    bg_color = background_colors.get(background_color, (255, 255, 255))

    # Đọc ảnh gốc
    img = cv2.imread(input_image_path)
    h_img, w_img = img.shape[:2]

    # Xử lý bounding box của đầu
    if len(bbox) == 4:
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # [x1, y1, x2, y2]
            head_x1, head_y1, head_x2, head_y2 = bbox
        else:  # [x, y, w, h]
            head_x1, head_y1, w, h = bbox
            head_x2, head_y2 = head_x1 + w, head_y1 + h
    else:
        raise ValueError("bbox phải có 4 giá trị")

    # Tính kích thước vùng đầu
    head_width = head_x2 - head_x1
    head_height = head_y2 - head_y1
    
    # Validate kích thước đầu
    min_head_width = 20  # Tối thiểu 20 pixel
    min_head_height = 25  # Tối thiểu 25 pixel
    
    if head_width < min_head_width or head_height < min_head_height:
        raise ValueError(
            f"Kích thước đầu quá nhỏ: {head_width}x{head_height} pixel. "
            f"Tối thiểu yêu cầu: {min_head_width}x{min_head_height} pixel. "
            f"Vui lòng kiểm tra lại ảnh đầu vào hoặc điều chỉnh ngưỡng confidence."
        )
    
    # Kiểm tra tỷ lệ khuôn mặt so với ảnh
    head_area = head_width * head_height
    image_area = w_img * h_img
    head_ratio_to_image = (head_area / image_area) * 100
    
    min_head_ratio = 7.0  # Tối thiểu 10%
    
    if head_ratio_to_image < min_head_ratio:
        raise ValueError(
            f"Tỷ lệ khuôn mặt so với ảnh quá nhỏ: {head_ratio_to_image:.2f}%. "
            f"Tối thiểu yêu cầu: {min_head_ratio}%. "
            f"Kích thước đầu: {head_width}x{head_height} pixel. "
            f"Kích thước ảnh: {w_img}x{h_img} pixel. "
            f"Vui lòng sử dụng ảnh có khuôn mặt lớn hơn hoặc điều chỉnh ngưỡng confidence."
        )
        
        # trả ra response với success = False
    
    # Kiểm tra tỷ lệ đầu có hợp lý không (chiều cao thường lớn hơn chiều rộng)
    head_ratio = head_height / head_width
    if head_ratio < 0.8 or head_ratio > 2.0:
        print(f"Cảnh báo: Tỷ lệ đầu không chuẩn ({head_ratio:.2f}). Kết quả có thể không chính xác.")
    
    # Kiểm tra vùng đầu có nằm trong ảnh không
    if (head_x1 < 0 or head_y1 < 0 or 
        head_x2 > w_img or head_y2 > h_img):
        raise ValueError(
            f"Vùng đầu ({head_x1},{head_y1},{head_x2},{head_y2}) "
            f"vượt quá kích thước ảnh ({w_img}x{h_img}). "
            f"Vui lòng kiểm tra lại bbox."
        )
    
    # Chuyển đổi margins từ mm sang pixel
    def mm_to_px_margin(mm_val):
        inch = mm_val / 25.4
        return int(round(inch * dpi))
    
    top_margin_px = mm_to_px_margin(top_margin_mm)
    bottom_margin_px = mm_to_px_margin(bottom_margin_mm)
    left_margin_px = mm_to_px_margin(left_margin_mm)
    right_margin_px = mm_to_px_margin(right_margin_mm)
    
    # Tính toán vùng cắt dựa trên vùng đầu và margins
    # Vùng cắt sẽ có kích thước đúng như yêu cầu (width_px x height_px)
    # với margins được tính từ vùng đầu
    
    # Tính vị trí đỉnh đầu và cằm trong ảnh kết quả
    head_top_in_result = top_margin_px  # Đỉnh đầu cách lề trên top_margin_px pixel
    head_bottom_in_result = height_px - bottom_margin_px  # Cằm cách lề dưới bottom_margin_px pixel
    
    # Tính chiều cao đầu trong ảnh kết quả
    head_height_in_result = head_bottom_in_result - head_top_in_result
    
    # Tính tỷ lệ scale để đạt được chiều cao đầu mong muốn
    scale = head_height_in_result / head_height
    new_head_width = int(head_width * scale)
    new_head_height = int(head_height * scale)
    
    # Tính vị trí đầu trong ảnh kết quả (căn giữa theo chiều ngang)
    head_left_in_result = (width_px - new_head_width) // 2
    head_right_in_result = head_left_in_result + new_head_width
    
    # Tính vùng cắt từ ảnh gốc để đạt được kết quả mong muốn
    # Cần tính ngược lại từ ảnh kết quả về ảnh gốc
    crop_width = int(width_px / scale)
    crop_height = int(height_px / scale)
    
    # Tính vị trí cắt trong ảnh gốc
    # Đầu sẽ được đặt ở vị trí head_left_in_result, head_top_in_result trong ảnh kết quả
    # Tương ứng với vị trí head_x1, head_y1 trong ảnh gốc
    crop_x1 = head_x1 - int(head_left_in_result / scale)
    crop_y1 = head_y1 - int(head_top_in_result / scale)
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height
    
    # Đảm bảo vùng cắt không vượt quá biên ảnh
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w_img, crop_x2)
    crop_y2 = min(h_img, crop_y2)
    
    # Cắt vùng ảnh theo tính toán
    cropped_region = img[crop_y1:crop_y2, crop_x1:crop_x2]
    h_cropped, w_cropped = cropped_region.shape[:2]

    # Resize vùng đã cắt về kích thước đích
    resized_region = cv2.resize(cropped_region, (width_px, height_px))

    # Tạo nền chuẩn
    result = np.full((height_px, width_px, 3), bg_color, dtype=np.uint8)

    # Đặt ảnh đã resize vào nền
    result[0:height_px, 0:width_px] = resized_region

    # Lưu ảnh kết quả
    cv2.imwrite(output_image_path, result)
    print(f"Đã lưu ảnh chuẩn tại: {output_image_path}")
    print(f"Kích thước nền: {width_px}x{height_px} px")
    if size_mm:
        print(f"Kích thước mm: {size_mm[0]}x{size_mm[1]} mm")
    print(f"DPI: {dpi}")
    print(f"Màu nền: {background_color}")
    print(f"Margins (mm): top={top_margin_mm}, bottom={bottom_margin_mm}, left={left_margin_mm}, right={right_margin_mm}")
    print(f"Margins (px): top={top_margin_px}, bottom={bottom_margin_px}, left={left_margin_px}, right={right_margin_px}")
    print(f"Vùng đầu gốc: ({head_x1},{head_y1}) -> ({head_x2},{head_y2})")
    print(f"Kích thước đầu: {head_width}x{head_height} px")
    print(f"Tỷ lệ khuôn mặt so với ảnh: {head_ratio_to_image:.2f}%")
    print(f"Vùng cắt: ({crop_x1},{crop_y1}) -> ({crop_x2},{crop_y2})")
    print(f"Kích thước vùng cắt: {w_cropped}x{h_cropped} px")
    print(f"Kích thước resize: {width_px}x{height_px} px")
    print(f"Tỷ lệ scale: {scale:.3f}")
    print(f"Vị trí đầu trong ảnh kết quả: ({head_left_in_result}, {head_top_in_result}) -> ({head_right_in_result}, {head_bottom_in_result})")
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

def convert_visa_with_detection(
    input_image_path, output_image_path, model_path="models/best_re_final.onnx",
    size_px=None, size_mm=None, dpi=300,
    top_margin_mm=0, bottom_margin_mm=0, left_margin_mm=0, right_margin_mm=0,
    background_color='white', conf_threshold=0.8
):
    """
    Hàm tích hợp để nhận diện vùng đầu và chuyển đổi ảnh visa:
    
    input_image_path: ảnh gốc
    output_image_path: nơi lưu ảnh kết quả
    model_path: đường dẫn đến model ONNX
    size_px: tuple (width_px, height_px) - kích thước pixel đầu ra
    size_mm: tuple (width_mm, height_mm) - kích thước mm
    dpi: dots per inch (mặc định 300)
    top_margin_mm: khoảng cách từ đầu tới rìa ảnh phía trên (mm)
    bottom_margin_mm: khoảng cách từ cằm tới rìa ảnh phía dưới (mm)
    left_margin_mm: khoảng cách từ mặt tới rìa ảnh bên trái (mm)
    right_margin_mm: khoảng cách từ mặt tới rìa ảnh bên phải (mm)
    background_color: màu nền ('white', 'light_blue', etc.)
    conf_threshold: ngưỡng confidence cho detection (mặc định 0.8)
    """
    
    # Nhận diện vùng đầu
    head_bbox = detect_head_region(input_image_path, model_path, conf_threshold)
    
    if head_bbox is None:
        raise ValueError("Không thể nhận diện vùng đầu trong ảnh")
    
    # Validate kích thước đầu trước khi xử lý
    if len(head_bbox) == 4:
        if head_bbox[2] > head_bbox[0] and head_bbox[3] > head_bbox[1]:  # [x1, y1, x2, y2]
            head_x1, head_y1, head_x2, head_y2 = head_bbox
        else:  # [x, y, w, h]
            head_x1, head_y1, w, h = head_bbox
            head_x2, head_y2 = head_x1 + w, head_y1 + h
        
        head_width = head_x2 - head_x1
        head_height = head_y2 - head_y1
        
        # Kiểm tra kích thước tối thiểu
        min_head_width = 20
        min_head_height = 25
        
        if head_width < min_head_width or head_height < min_head_height:
            raise ValueError(
                f"Kích thước đầu quá nhỏ: {head_width}x{head_height} pixel. "
                f"Tối thiểu yêu cầu: {min_head_width}x{min_head_height} pixel. "
                f"Vui lòng thử giảm ngưỡng confidence hoặc kiểm tra lại ảnh đầu vào."
            )
        
        # Kiểm tra tỷ lệ khuôn mặt so với ảnh
        img = cv2.imread(input_image_path)
        h_img, w_img = img.shape[:2]
        head_area = head_width * head_height
        image_area = w_img * h_img
        head_ratio_to_image = (head_area / image_area) * 100
        
        min_head_ratio = 20.0  # Tối thiểu 20%
        
        if head_ratio_to_image < min_head_ratio:
            raise ValueError(
                f"Tỷ lệ khuôn mặt so với ảnh quá nhỏ: {head_ratio_to_image:.2f}%. "
                f"Tối thiểu yêu cầu: {min_head_ratio}%. "
                f"Kích thước đầu: {head_width}x{head_height} pixel. "
                f"Kích thước ảnh: {w_img}x{h_img} pixel. "
                f"Vui lòng sử dụng ảnh có khuôn mặt lớn hơn hoặc điều chỉnh ngưỡng confidence."
            )
        
        # Kiểm tra tỷ lệ đầu
        head_ratio = head_height / head_width
        if head_ratio < 0.8 or head_ratio > 2.0:
            print(f"Cảnh báo: Tỷ lệ đầu không chuẩn ({head_ratio:.2f}). Kết quả có thể không chính xác.")
    
    # Chuyển đổi ảnh với vùng đầu đã nhận diện
    handle_convert_visa(
        input_image_path=input_image_path,
        output_image_path=output_image_path,
        bbox=head_bbox,
        size_px=size_px,
        size_mm=size_mm,
        dpi=dpi,
        top_margin_mm=top_margin_mm,
        bottom_margin_mm=bottom_margin_mm,
        left_margin_mm=left_margin_mm,
        right_margin_mm=right_margin_mm,
        background_color=background_color
    )
    
    return head_bbox

# Nếu muốn chạy thử:
if __name__ == "__main__":
    print_visa_photo_pixel_sizes()
    
    # Ví dụ sử dụng:
    print("\n=== Ví dụ sử dụng ===")
    
    # Ví dụ 1: Ảnh 3x4 cm với margins
    print("Ví dụ 1: Ảnh 3x4 cm với margins")
    """
    convert_visa_with_detection(
        input_image_path="input.jpg",
        output_image_path="output_3x4.jpg",
        size_mm=(30, 40),  # 3x4 cm
        dpi=300,
        top_margin_mm=2,    # 2mm từ đầu tới rìa trên
        bottom_margin_mm=3, # 3mm từ cằm tới rìa dưới
        left_margin_mm=2,   # 2mm từ mặt tới rìa trái
        right_margin_mm=2,  # 2mm từ mặt tới rìa phải
        background_color='white'
    )
    """
    
    # Ví dụ 2: Ảnh với kích thước pixel cụ thể
    print("Ví dụ 2: Ảnh với kích thước pixel cụ thể")
    """
    convert_visa_with_detection(
        input_image_path="input.jpg",
        output_image_path="output_custom.jpg",
        size_px=(600, 800),  # 600x800 pixel
        dpi=300,
        top_margin_mm=1.5,
        bottom_margin_mm=2.5,
        left_margin_mm=1.5,
        right_margin_mm=1.5,
        background_color='light_blue'
    )
    """
    
    # Ví dụ 3: Chỉ nhận diện vùng đầu
    print("Ví dụ 3: Chỉ nhận diện vùng đầu")
    """
    bbox = detect_head_region("input.jpg", conf_threshold=0.7)
    if bbox:
        print(f"Vùng đầu: {bbox}")
    """
    
    # Ví dụ 4: Chuyển đổi với bbox đã biết
    print("Ví dụ 4: Chuyển đổi với bbox đã biết")
    """
    bbox = [100, 150, 200, 250]  # [x1, y1, x2, y2]
    handle_convert_visa(
        input_image_path="input.jpg",
        output_image_path="output_known_bbox.jpg",
        bbox=bbox,
        size_mm=(20, 30),  # 2x3 cm
        dpi=300,
        top_margin_mm=1,
        bottom_margin_mm=2,
        left_margin_mm=1,
        right_margin_mm=1
    )
    """
