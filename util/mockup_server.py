# mockup_server.py
# 2025-05-11 00:15:50 FastAPI 기반 이미지 서버 초기 구현
# 2025-05-11 14:40:00 /result/create 엔드포인트 추가 (JSON POST, Pydantic 검증, 비동기 로깅)
import os
import sys
import json # 2025-05-11 14:40:00 JSON 처리를 위해 추가
from fastapi import FastAPI, HTTPException, APIRouter, Request, Body
from typing import Dict, Any
from fastapi.responses import FileResponse, JSONResponse # 2025-05-11 14:40:00 JSONResponse 추가
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import (
    request_validation_exception_handler as fastapi_validation_handler,
)
from loguru import logger
import uvicorn
from typing import List, Optional, Dict, Any # 2025-05-11 14:40:00 Typing 추가
from pathlib import Path # 2025-05-11 14:40:00 Path 추가
from datetime import datetime # 2025-05-11 14:40:00 datetime 추가
import pytz # 2025-05-11 14:40:00 pytz 추가
import asyncio # 2025-05-11 14:40:00 asyncio 추가
import re

# --- 상위 경로의 models.py 사용을 위한 sys.path 설정 ---
PARENT_DIR = Path(__file__).resolve().parent.parent  # ../
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from models import ChildCreateRequest, JsonResponseModel  # 변경된 모델 임포트

# --- 환경 설정 ---
IMAGE_DIR = "/Users/basaaja/Python/ocr_image/temp/server_image"  # 이미지가 저장된 기본 폴더 경로
LOG_DIR_STR = "/Users/basaaja/Python/openocr_log_mockup" # 2025-05-11 14:40:00 수신된 최종 결과 JSON 저장 경로 변수명 명확화 (기존 LOG_DIR 에서 LOG_DIR_STR로 변경)

LOG_FILE_PATH = "image_server.log" # 로그 파일 경로

# --- 로거 설정 ---
logger.add(LOG_FILE_PATH, rotation="10 MB", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# --- FastAPI 앱 생성 ---
app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    ChildCreateRequest 검증 오류 발생 시 422 원인을 필드별로 상세 로그로 기록합니다.
    """
    formatted_errors: List[str] = []
    for err in exc.errors():
        loc_path = " -> ".join(map(str, err.get("loc", [])))
        msg = err.get("msg", "")
        err_type = err.get("type", "")
        formatted_errors.append(f"[{loc_path}] ({err_type}) {msg}")

    logger.error(
        "422 Unprocessable Entity: ChildCreateRequest 검증 실패\n"
        f"URL        : {request.url.path}\n"
        f"Client Host: {request.client.host if request.client else 'N/A'}\n"
        f"Errors     :\n  - " + "\n  - ".join(formatted_errors) + "\n"
        f"Body       : {json.dumps(exc.body, ensure_ascii=False, indent=2)}"
    )

    # FastAPI 기본 422 응답을 그대로 반환
    return await fastapi_validation_handler(request, exc)


async def write_log_async(log_file_path: Path, data: Dict[str, Any]):
    """비동기적으로 로그 파일에 JSON 데이터를 기록합니다."""
    # 2025-05-11 14:40:00 비동기 파일 쓰기 작업
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, log_file_path.write_text, json.dumps(data, ensure_ascii=False, indent=4))
        logger.info(f"Successfully wrote log to: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to write log to {log_file_path}: {e}")
        # 필요시 여기서 추가적인 예외 처리를 할 수 있습니다. (예: 재시도 로직, 관리자 알림 등)
        # 현재는 에러 로깅만 수행합니다.

@app.post("/create")
async def result_server(request: Request):
    """
    JSON 데이터를 Post 형태로 수신하여 검증 없이 원본 그대로 로그 파일로 저장하고, 성공 응답을 반환합니다.
    """
    # 원본 JSON 데이터 추출
    try:
        data = await request.json()
    except Exception as e:
        logger.error(f"JSON 파싱 오류: {e}")
        return JSONResponse(status_code=400, content={"statusCd": "9001", "statusMessage": "JSON 데이터 형식 오류"})

    logger.info(f"Received /create with tran_id: {data.get('tran_id', 'N/A')}")
    logger.debug(data)
    logger.info(f"Request data: {json.dumps(data, ensure_ascii=False, indent=2)}")

    # 로그 디렉토리 및 파일명 생성
    kst_tz = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst_tz)

    log_dir_base = Path(LOG_DIR_STR)
    year_month_folder = log_dir_base / now_kst.strftime("%Y%m")

    try:
        year_month_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured log directory exists: {year_month_folder}")
    except Exception as e:
        logger.error(f"Could not create log directory {year_month_folder}: {e}")
        raise HTTPException(status_code=500, detail="Error creating log directory.")

    log_file_name_prefix = now_kst.strftime("%Y%m%d%H%M%S")
    # tran_id 추출
    string_value = data.get("StringINDTO", {}).get("stringValue", "")
    match = re.search(r'"tran_id"\s*:\s*"([^"]+)"', string_value)
    tran_id = match.group(1) if match else "N/A"
    log_file_name = f"{log_file_name_prefix}_{tran_id}_result.log"
    log_file_path = year_month_folder / log_file_name

    # 수신 데이터 로깅 (비동기)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, log_file_path.write_text, json.dumps(data, ensure_ascii=False, indent=4))
        logger.info(f"Successfully wrote log to: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to write log to {log_file_path}: {e}")

    # 최종 응답 생성 (간단하게 tran_id만 반환)
    response_data = {
        "tran_id": data.get("tran_id", "N/A"),
        "statusCd": "0000",
        "statusMessage": "정상적으로 등록 되었습니다."
    }
    logger.info(f"Successfully processed /create for tran_id: {data.get('tran_id', 'N/A')}")
    return JSONResponse(status_code=200, content=response_data)

# --- 기존 라우트 ---
@app.get("/")
async def read_root():
    """
    루트 경로 요청 시 간단한 환영 메시지를 반환합니다.
    """
    logger.info("Root path '/' accessed.")
    return {"message": "Image Server is running. Use /image/{filename} to get an image. Use POST /result/create to submit results."}

@app.get("/image/{filename}")
async def image_server(filename: str):
    """
    지정된 filename에 해당하는 이미지를 바이너리 형태로 제공합니다.
    이미지는 IMAGE_DIR에 정의된 폴더에서 찾습니다.
    """
    image_path = os.path.join(IMAGE_DIR, filename)
    logger.info(f"Request for image: {filename}")
    logger.debug(f"Full image path: {image_path}")

    if not os.path.exists(IMAGE_DIR):
        logger.error(f"Image directory not found: {IMAGE_DIR}")
        raise HTTPException(status_code=500, detail=f"Server configuration error: Image directory '{IMAGE_DIR}' not found.")

    if os.path.isfile(image_path):
        logger.info(f"Serving image: {filename}")
        return FileResponse(image_path)
    else:
        logger.warning(f"Image not found: {filename} at path: {image_path}")
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")


if __name__ == "__main__":
    logger.info("Starting image server with uvicorn.")
    # 서버 실행: host와 port는 필요에 따라 변경 가능합니다.
    # reload=True 옵션은 개발 중 코드 변경 시 서버를 자동으로 재시작해줍니다.
    # 운영 환경에서는 reload=False로 설정하는 것이 일반적입니다.
    uvicorn.run("mockup_server:app", host="0.0.0.0", port=5002, reload=True)
    # 위 uvicorn.run의 첫번째 인자는 "파이썬파일명:FastAPI인스턴스명" 입니다.
    # 이 파일명이 image_server.py 이므로 "image_server:app"으로 지정합니다.
    # 파일명이 mockup_server.py이므로 "mockup_server:app"으로 수정했습니다. (2025-05-11 14:40:00)