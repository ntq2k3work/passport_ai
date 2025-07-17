from prediction import prediction, load_image, predict, xywh2xyxy
import matplotlib.pyplot as plt
import fire 
import cv2
from handle_convert import handle_convert
import numpy as np
import os

def adjust_head_margin(image_path, output_path, output_size=(600, 800), head_height_ratio=0.7, top_margin_ratio=0.1):
    # Đọc ảnh gốc và lấy bounding box vùng đầu
    image, _, _ = load_image(image_path, [None, None, 640, 640])
    model = prediction.__globals__['load_model']('models/best_re_final.onnx')
    input_I = load_image(image_path, model[1])
    boxes, scores, class_ids = predict(input_I[0], model[0], input_I[1])
    if len(boxes) == 0:
        raise Exception('Không phát hiện vùng đầu nào!')
    # Lấy box có score cao nhất
    idx = np.argmax(scores)
    box = boxes[idx]
    # Chuyển box sang (x1, y1, x2, y2)
    box_xyxy = xywh2xyxy(np.array([box]))[0].astype(int)
    x1, y1, x2, y2 = box_xyxy
    head_h = y2 - y1
    head_w = x2 - x1
    # Tính toán vị trí head mới trên ảnh chuẩn
    out_w, out_h = output_size
    new_head_h = int(out_h * head_height_ratio)
    new_head_w = int(head_w * (new_head_h / head_h))
    # Resize vùng đầu
    head_crop = image[y1:y2, x1:x2]
    head_crop_resized = cv2.resize(head_crop, (new_head_w, new_head_h))
    # Tạo ảnh nền trắng
    result = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    # Tính vị trí dán head vào ảnh nền
    top = int(out_h * top_margin_ratio)
    left = (out_w - new_head_w) // 2
    result[top:top+new_head_h, left:left+new_head_w] = head_crop_resized
    cv2.imwrite(output_path, result)
    return output_path

def predict_from_teminal(image_path = "Reports/image/test.jpg", output_size_name="3x4"):
    # Map tên size sang pixel
    size_map = {"2x3": (400, 600), "3x4": (600, 800), "4x6": (800, 1200)}
    output_size = size_map.get(output_size_name, (600, 800))
    temp_path = "temp_head_margin.jpg"
    # Bước 1: Căn chỉnh margin vùng đầu chuẩn quốc tế
    adjusted_path = adjust_head_margin(image_path, temp_path, output_size=output_size)
    # Bước 2: Cắt ảnh theo tỉ lệ người dùng cần (nếu muốn crop lại)
    final_path = f"output_{output_size_name}.jpg"
    handle_convert(adjusted_path, final_path, output_size=output_size)
    # Hiển thị ảnh kết quả
    img = cv2.imread(final_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis('off')
    plt.show()
    # Xoá file tạm
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__=='__main__':
    print("Starting execution:")
    fire.Fire(predict_from_teminal)
