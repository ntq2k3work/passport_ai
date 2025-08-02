from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
from handle_convert import handle_convert_visa, detect_head_region, validate_photo_requirements, validate_smile,validate_shirt,validate_all
from typing import Optional
from fastapi.responses import JSONResponse as JsonResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    output_path = f"./output/output_{file.filename}"
    
    # Lưu file upload tạm thời
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Validate photo requirements (hat, glasses, etc.)
    try:
        validate_all(input_path)
    except ValueError as e:
        os.remove(input_path)
        raise HTTPException(
            status_code=422,
            detail={"error": "ValidationError", "message": str(e)}
        )
        
    # try:
    #     validate_shirt(input_path)
    # except ValueError as e:
    #     os.remove(input_path)
    #     raise HTTPException(
    #         status_code=422,
    #         detail={"error": "ValidationError", "message": str(e)}
    #     )
        
    # try:
    #     validate_smile(input_path)
    # except ValueError as e:
    #     os.remove(input_path)
    #     raise HTTPException(
    #         status_code=422,
    #         detail={"error": "SmileValidationError", "message": str(e)}
    #     )
    except Exception as e:
        os.remove(input_path)
        raise HTTPException(
            status_code=500,
            detail={"error": "UnknownError", "message": "Lỗi không xác định khi xử lý ảnh"}
        )
    
    # Nhận diện vùng đầu
    head_bbox = detect_head_region(input_path)
    if head_bbox is None:
        os.remove(input_path)
        raise HTTPException(
            status_code=422, 
            detail={"error": "HeadError", "message": "Không nhận diện được vùng đầu trong ảnh"}
        )
    
    # Parse size_px để lấy width và height
    try:
        width_px, height_px = map(int, size_px.split('x'))
    except ValueError:
        os.remove(input_path)
        raise HTTPException(
            status_code=400,
            detail={"error": "SizeError", "message": "Định dạng size_px không hợp lệ"}
        )
    
    # Parse size_mm để lấy width và height
    try:
        width_mm, height_mm = map(float, size_mm.split('x'))
    except ValueError:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="Định dạng size_mm không hợp lệ")
    
    # Kiểm tra tỷ lệ khuôn mặt so với ảnh
    head_ratio_to_image = (head_bbox[2] * head_bbox[3]) / (width_px * height_px)
    if head_ratio_to_image < 0.07:
        os.remove(input_path)
        raise HTTPException(
            status_code=422,
            detail={"error": "FaceMinimizeError", "message": "Tỉ lệ khuôn mặt so với ảnh quá nhỏ"}
        )
    
    # Gọi hàm xử lý ảnh với các tham số mới
    try:
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
    except Exception as e:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail=str(e))
    
    
    
    # Xóa file input sau khi xử lý
    os.remove(input_path)
    
    # Trả về file output
    return FileResponse(output_path, media_type="image/jpeg", filename=output_path) 