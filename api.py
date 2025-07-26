from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from handle_convert import handle_convert_visa, detect_head_region
from typing import Optional

app = FastAPI()

@app.post("/convert")
def convert_image(
    file: UploadFile = File(...),
    size_mm: str = Form(...),
    size_px: str = Form(...),
    dpi: int = Form(...),
    background: str = Form(...),
    top_margin_mm: float = Form(...),
    bottom_margin_mm: float = Form(...),
    left_margin_mm: float = Form(...),
    right_margin_mm: float = Form(...)
):
    input_path = f"temp_{file.filename}"
    output_path = f"output_{file.filename}"
    
    # Lưu file upload tạm thời
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Nhận diện vùng đầu
    head_bbox = detect_head_region(input_path)
    if head_bbox is None:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="Không thể nhận diện vùng đầu trong ảnh")
    
    # Parse size_px để lấy width và height
    try:
        width_px, height_px = map(int, size_px.split('x'))
    except ValueError:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="Định dạng size_px không hợp lệ")
    
    # Parse size_mm để lấy width và height
    try:
        width_mm, height_mm = map(float, size_mm.split('x'))
    except ValueError:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="Định dạng size_mm không hợp lệ")
    
    # Gọi hàm xử lý ảnh với các tham số mới
    handle_convert_visa(
        input_image_path=input_path,
        output_image_path=output_path,
        bbox=head_bbox,
        size_px=(width_px, height_px),
        size_mm=(width_mm, height_mm),
        dpi=dpi,
        top_margin_mm=top_margin_mm,
        bottom_margin_mm=bottom_margin_mm,
        left_margin_mm=left_margin_mm,
        right_margin_mm=right_margin_mm,
        background_color=background
    )
    
    # Xóa file input sau khi xử lý
    os.remove(input_path)
    
    # Trả về file output
    return FileResponse(output_path, media_type="image/jpeg", filename=output_path) 