from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from handle_convert import handle_convert_visa, detect_head_region

app = FastAPI()

@app.post("/convert")
def convert_image(file: UploadFile = File(...), type: str = Form("3x4")):
    input_path = f"temp_{file.filename}"
    output_path = f"output_{file.filename}"
    
    # Lưu file upload tạm thời
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Nhận diện vùng đầu
    bbox = detect_head_region(input_path)
    if bbox is None:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="Không thể nhận diện vùng đầu trong ảnh")
    
    # Gọi hàm xử lý ảnh với bounding box
    handle_convert_visa(
        input_image_path=input_path,
        output_image_path=output_path,
        bbox=bbox,
        photo_type=type,  # '2x3' hoặc '3x4'
        dpi=300,
        head_ratio=0.75
    )
    
    # Xóa file input sau khi xử lý
    os.remove(input_path)
    
    # Trả về file output
    return FileResponse(output_path, media_type="image/jpeg", filename=output_path) 