# openocr.py
# 2025-05-10 20:48:33 FastAPI 기반 JSON 처리 및 로깅 기능 구현
# 2025-05-10 23:32:10 ocr_id 타임스탬프 형식 변경 (%Y-%m-%d-%H-%M-%S -> %Y%m%d-%H%M%S)
# 2025-05-11 00:15:00 CommonUtil 클래스를 사용하여 load_settings, logCreate 기능 호출하도록 변경
# 2025-05-11 00:40:00 childCreate에 이미지 다운로드 기능 추가 및 httpx 라이브러리 사용
# 2025-05-11 16:53:00 FastAPI 앱 시작 시 Ocr 객체 미리 로드하여 성능 향상 (수정: 2025-05-11 16:57:00)

import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager # lifespan을 위해 추가 (선택적 고급 관리)

import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks 
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, ValidationError
from loguru import logger
import pytz
from Download import Download  # Download 클래스 임포트
# CommonUtil 클래스 임포트
from common import CommonUtil
import shutil # 파일 이동을 위해 추가
from models import  ChildCreateRequest # OpenOcr 클래스 import
from Ocr import Ocr  
# --- 설정 로드 ---
try:
    settings = CommonUtil.load_settings(Path(__file__).parent)
except FileNotFoundError as e:
    logger.critical(f"초기 설정 파일 로드 실패: {e}. 프로그램을 시작할 수 없습니다.")
    exit(1)
except Exception as e:
    logger.critical(f"설정 로드 중 예상치 못한 오류 발생: {e}. 프로그램을 시작할 수 없습니다.", exc_info=True)
    exit(1)


# --- 로깅 설정 (settings 로드 후 수행) ---
logger.remove()
log_level = settings.get("LOG_LEVEL", "INFO").upper()
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss,SSS} | {level: <5} | {module}.{function} | {message}",
    level=log_level,
    colorize=True
)
log_file_path_app = Path(settings["LOG_DIR"]) / "openocr_app.log"
logger.add(
    log_file_path_app,
    rotation="00:00",              # ① 매일 00:00에 새 로그 파일 생성
    # retention 파라미터 제거          # ② 7일 자동 삭제 기능 제거
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss,SSS} | {level: <5} | {module}.{function} | {message}",
    encoding="utf-8",
    enqueue=True                   # 멀티프로세스 환경에서 안전한 로깅을 위한 옵션
)

logger.info(f"프로그램 실행 환경: APP_ENV = {os.getenv('APP_ENV', 'basaaja')}")
logger.info(f"설정 정보: {settings}")
logger.info(f"애플리케이션 로그 파일 위치: {log_file_path_app}")
logger.info(f"이미지 다운로드 기본 위치: {settings.get('DOWNLOAD_IMAGE_DIR')}")


# --- OCR 프로세서 전역 변수 ---
ocr_processor: Optional[Ocr] = None

# --- FastAPI 앱 수명 주기 이벤트 (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시
    global ocr_processor
    logger.info("애플리케이션 시작: OCR 프로세서 초기화 중...")
    try:
        # Ocr 클래스 초기화 시 settings만 전달
        ocr_processor = Ocr(settings=settings)
        logger.info("OCR 프로세서 초기화 완료.")
    except Exception as e:
        logger.critical(f"OCR 프로세서 초기화 실패: {e}", exc_info=True)
        # OCR 프로세서 초기화 실패 시 앱 실행을 중단하거나,
        # ocr_processor를 None으로 두고 API에서 이를 확인하여 오류 응답 처리
        ocr_processor = None
    yield
    # 애플리케이션 종료 시 (필요한 경우 리소스 정리)
    logger.info("애플리케이션 종료.")

# --- FastAPI 앱 생성 (lifespan 적용) ---
app = FastAPI(lifespan=lifespan)


# --- 오류 핸들러 ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"예상치 못한 오류 발생: {exc}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content=jsonResult(tran_id="N/A", ocr_id="N/A", statusCd="9999")
    )

# --- 메서드 구현 ---
def jsonResult(tran_id: str, ocr_id: Optional[str], statusCd: str) -> Dict[str, str]:
    logger.debug(f"jsonResult 호출됨: tran_id={tran_id}, ocr_id={ocr_id}, statusCd={statusCd}")
    status_messages = {
        "0000": "정상적으로 등록 되었습니다.",
        "9001": "JSON 데이터 형식 오류",
        "9002": "이미 사용한 tran_id 입니다.",
        "9003": "이미지 다운로드 관련 설정 오류 또는 누락", 
        "9004": "이미지 다운로드 실패 (HTTP 오류 또는 요청 오류)", # 새로운 상태 코드 예시
        "9005": "이미지 파일 저장 실패", 
        "9006": "지원되지 않는 확장자", 
        "9007": "파일 이동·경로 조작 오류-임시 폴더 이동 시",
        "9999": "알수 없는 에러가 발생했습니다."
    }
    statusMessage = status_messages.get(statusCd, "알수 없는 상태 코드입니다.")
    
    response_data = {
        "tran_id": tran_id,
        "ocr_id": ocr_id if ocr_id else "N/A",
        "statusCd": statusCd,
        "statusMessage": statusMessage
    }
    logger.debug(f"jsonResult 응답: {response_data}")
    return response_data

# openocr.py 에 추가될 비동기 백그라운드 작업 함수

async def download_save_and_ocr_background(
    host_name : str,
    file_name: str,
    tran_id: str,
    ocr_id: str,
    host: str,
    type_: str,
    saved_file: str,
    channel: str,
    target: str
):
    """
    백그라운드에서 이미지를 다운로드, 저장 후 OCR 처리를 수행하고 결과를 콘솔에 출력합니다.
    """
    global ocr_processor # 전역 ocr_processor 사용
    global settings # 전역 settings 사용

    if ocr_processor is None:
        logger.error(f"[배경작업][{tran_id}] OCR 프로세서가 초기화되지 않았습니다. OCR 처리를 건너뜁니다.")
        return
    # 설정값 가져오기
    download_dir_str = settings.get("DOWNLOAD_IMAGE_DIR")
    ocr_input_dir_base_str = settings.get("OCR_INPUT_DIR")
    # ocr_json_path_str = settings.get("OCR_JSON_PATH") # Ocr 객체가 내부적으로 사용
    # ocr_output_dir_str = settings.get("OCR_OUTPUT_DIR") # Ocr 객체가 내부적으로 사용
    # result_web_server_dir = settings.get("RESULT_WEB_SERVER_DIR") # Ocr 객체가 내부적으로 사용


    if not all([download_dir_str, file_name, ocr_input_dir_base_str]): # 필수 경로 체크
        missing_info = []
        if not download_dir_str: missing_info.append("DOWNLOAD_IMAGE_DIR")
        if not file_name: missing_info.append("file_name")
        if not ocr_input_dir_base_str: missing_info.append("OCR_INPUT_DIR")
        logger.error(f"[배경작업][{tran_id}] OCR 처리 위한 정보 부족 ({', '.join(missing_info)}), 건너뜁니다.")
        return
    # 다운로드는 우선 DOWNLOAD_IMAGE_DIR에 저장
    download_image_dir = Path(download_dir_str)
    url_base_processed = host_name.rstrip('/')
    file_url = f"{url_base_processed}"

    # Download 클래스를 활용하여 이미지 다운로드 시도
    if channel == "nais":
        downloaded_file_path = await Download.save(
            file_url=file_url,
            download_dir=download_image_dir,
            file_name=file_name,
            tran_id=tran_id,
        )
    elif channel == "mms":
        mms_nas_dir = settings.get("MMS_NAS_DIR")
        downloaded_file_path = Path(mms_nas_dir) / saved_file
        if Download.checkMMS(saved_file, mms_nas_dir):  
            logger.info(f"[배경작업][{tran_id}] MMS 파일 확인 성공: {downloaded_file_path}")
        else:
            logger.error(f"[배경작업][{tran_id}] MMS 파일 확인 실패: {downloaded_file_path}")
            return
    else:
        logger.error(f"[배경작업][{tran_id}] 지원되지 않는 채널: {channel}")
        return
    # downloaded_file_path 출력
    logger.debug(f"[배경작업][{tran_id}] 다운로드 파일 경로: {downloaded_file_path}")

    if downloaded_file_path is None:
        logger.warning(f"[배경작업][{tran_id}] 이미지 다운로드 실패로 OCR 처리를 진행하지 않습니다.")
        return

    # --- OCR 처리 로직 ---
    # OpenOcr 클래스는 input_dir 내의 모든 이미지를 처리하므로,
    # 다운로드 받은 특정 파일 하나만 처리하기 위해 임시 input 디렉토리를 사용합니다.
    # tran_id 기반으로 고유한 임시 디렉토리 생성
    temp_ocr_input_dir = Path(ocr_input_dir_base_str) / tran_id
    temp_ocr_input_dir.mkdir(parents=True, exist_ok=True)
    
    # OCR 처리할 파일을 임시 입력 디렉토리로 이동 또는 복사 (여기서는 이동)
    # OpenOcr 클래스가 원본 파일을 변경할 수 있으므로, 복사가 더 안전할 수 있음.
    # 여기서는 요청에 따라 downloaded_file_path에 이미 저장된 것을 사용하므로
    # OpenOcr의 input_dir을 downloaded_file_path가 있는 디렉토리로 지정.
    # 단, OpenOcr가 해당 디렉토리의 모든 이미지를 처리하므로,
    # 이 파일 하나만 있는 임시 디렉토리를 만들고 그곳으로 파일을 옮긴다.
    
    ocr_target_file_path = temp_ocr_input_dir / file_name
    try:
        
        shutil.move(str(downloaded_file_path), str(ocr_target_file_path))
        logger.info(f"[배경작업][{tran_id}] OCR 처리를 위해 파일을 '{ocr_target_file_path}'로 이동했습니다.")
    except Exception as e:
        logger.error(f"[배경작업][{tran_id}] OCR 처리용 파일 이동 실패: {downloaded_file_path} -> {ocr_target_file_path}, 오류: {e}")
        # 이동 실패 시 임시 디렉토리 정리 시도
        try:
            shutil.rmtree(temp_ocr_input_dir)
        except Exception as rex:
            logger.error(f"[배경작업][{tran_id}] 임시 OCR 입력 디렉토리 삭제 실패: {temp_ocr_input_dir}, 오류: {rex}")
        jsonResult(tran_id, ocr_id, "9007")  # 파일 이동·경로 조작 오류
        return
        
    # OCR output 디렉토리도 tran_id별로 구분하면 좋을 수 있으나, OpenOcr.py의 현재 구조는 output_dir 하나를 사용.
    # OpenOcr 클래스에 output_dir도 tran_id에 따라 동적으로 설정하도록 수정하거나,
    # OpenOcr가 파일명에 tran_id를 포함시켜 저장하도록 수정이 필요할 수 있음.
    # 여기서는 설정된 OCR_OUTPUT_DIR을 그대로 사용.
    
    logger.info(f"[배경작업][{tran_id}] OCR 처리 시작. Input: {temp_ocr_input_dir}, Json: {temp_ocr_input_dir}")
    try:
        # 전역 ocr_processor 사용, process_images에 필요한 인자 전달
        result_json = ocr_processor.process_images(
            input_dir=str(temp_ocr_input_dir), # 처리할 이미지가 있는 특정 디렉토리
            log_dir=str(Path(settings["LOG_DIR"])), # 로그 저장 디렉토리
            tran_id=tran_id,
            ocr_id=ocr_id,
            host=host,
            type_=type_,
            saved_file=saved_file,
            target=target
        )

        logger.info(f"[배경작업][{tran_id}] OCR 처리 완료. ")

    except Exception as e:
        logger.error(f"[배경작업][{tran_id}] OCR 처리 중 오류 발생: {e}", exc_info=True)
    finally:
        try:
            shutil.rmtree(temp_ocr_input_dir)
            logger.info(f"[배경작업][{tran_id}] 임시 OCR 입력 디렉토리 삭제 완료: {temp_ocr_input_dir}")
        except Exception as e:
            logger.error(f"[배경작업][{tran_id}] 임시 OCR 입력 디렉토리 삭제 실패: {temp_ocr_input_dir}, 오류: {e}")

@app.post("/openocr/child/create")
async def childCreate(request: Request, background_tasks: BackgroundTasks):
    logger.info("childCreate 엔드포인트 호출됨")
    global settings # 전역 settings 사용
    raw_request_body = await request.body()
    logger.debug(f"수신 원시 데이터: {raw_request_body.decode('utf-8')}")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        logger.error("JSON 파싱 오류")
        return JSONResponse(status_code=200, content=jsonResult(tran_id="UNKNOWN_JSON_ERROR", ocr_id=None, statusCd="9001"))

    logger.debug(f"수신 JSON 데이터 (파싱 후): {data}")
    

    tran_id_from_request = data.get("tran_id")
    if not tran_id_from_request:
        logger.error("tran_id가 요청 데이터에 없습니다.")
        return JSONResponse(status_code=200, content=jsonResult(tran_id="MISSING_TRAN_ID", ocr_id=None, statusCd="9001"))

    current_status_cd: str | None = None       # UnboundLocalError 방지용 초기화

    try:
        validated_data = ChildCreateRequest(**data)
        logger.success(f"데이터 유효성 검사 통과: tran_id={validated_data.tran_id}")
    except ValidationError as e:
        logger.error(f"JSON 데이터 유효성 검사 실패: tran_id={tran_id_from_request}, 오류: {e.errors()}")
        return JSONResponse(status_code=200, content=jsonResult(tran_id=tran_id_from_request, ocr_id=None, statusCd="9001"))
    
    # ---------- [추가] saved_file 허용 확장자 검증 ----------
    allowed_extensions = settings.get(
        "VALID_IMAGE_EXTENSIONS"
    )
    
    kst_tz = pytz.timezone('Asia/Seoul')
    now_kst_for_ocr_id = datetime.now(kst_tz)
    ocr_id_timestamp = now_kst_for_ocr_id.strftime("%Y%m%d-%H%M%S")
    ocr_id_random_hex = uuid.uuid4().hex
    ocr_id = f"{ocr_id_timestamp}-{ocr_id_random_hex}"
    logger.debug(f"생성된 ocr_id: {ocr_id}")
    logger.debug(f"log type: {validated_data.type}")
    data_for_log = validated_data.model_dump()
    # --- 로그 저장 로직 ---
    log_status_cd = CommonUtil.logCreate(settings, data_for_log, validated_data.tran_id,validated_data.type, "create")

    current_status_cd = "0000" 

    saved_ext = Path(validated_data.saved_file).suffix.lower()
    if saved_ext not in [e.lower() for e in allowed_extensions]:
        logger.error(f"허용되지 않은 이미지 확장자: {saved_ext}")
        current_status_cd = "9006" 
        return JSONResponse(status_code=200, content=jsonResult(tran_id=validated_data.tran_id, ocr_id=ocr_id, statusCd=current_status_cd))
    if log_status_cd == "9002":
        logger.error(f"CommonUtil.logCreate에서 tran_id 중복 감지: {validated_data.tran_id}")
        current_status_cd = "9002"
        # 중복 감지 시 즉시 응답 (이미지 다운로드 X)
        return JSONResponse(status_code=200, content=jsonResult(tran_id=validated_data.tran_id, ocr_id=ocr_id, statusCd=current_status_cd))
    elif log_status_cd == "9999":
        logger.error(f"CommonUtil.logCreate에서 로그 저장 실패: {validated_data.tran_id}")
        current_status_cd = "9999"
        # 로그 저장 실패 시 즉시 응답 (이미지 다운로드 X)
        return JSONResponse(status_code=200, content=jsonResult(tran_id=validated_data.tran_id, ocr_id=ocr_id, statusCd=current_status_cd))
    
    # --- 이미지 다운로드 로직을 백그라운드 작업으로 추가 ---
    # 로그 저장이 성공했고 (current_status_cd가 "0000"으로 유지), tran_id 중복이 아닐 때만 이미지 다운로드 시도
    if current_status_cd == "0000":
        #image_web_server_url_base = settings.get("IMAGE_WEB_SERVER_DIR")
        # download_image_dir_str = settings.get("DOWNLOAD_IMAGE_DIR") # 이 변수는 download_save_and_ocr_background 내부에서 사용
        saved_file_name = validated_data.saved_file
        host_name  = validated_data.host

        if saved_file_name: # download_image_dir_str은 백그라운드 함수에서 직접 settings 참조
            background_tasks.add_task(
                download_save_and_ocr_background,
                # settings_dict 제거됨
                host_name,
                saved_file_name,
                validated_data.tran_id,
                ocr_id,
                validated_data.host,
                validated_data.type,
                validated_data.saved_file,
                validated_data.channel,
                validated_data.target
            )
            logger.info(f"이미지 다운로드 및 OCR 처리 작업 백그라운드로 예약됨: {saved_file_name} (tran_id: {validated_data.tran_id})")
        else:

            logger.warning(f"이미지 다운로드에 필요한 설정 부족 saved_file_name, 백그라운드 다운로드를 예약하지 않음. (tran_id: {validated_data.tran_id})")

    logger.info(f"childCreate 응답 반환: tran_id={validated_data.tran_id}, ocr_id={ocr_id}, statusCd={current_status_cd}. 이미지 다운로드는 백그라운드 처리.")
    return JSONResponse(status_code=200, content=jsonResult(tran_id=validated_data.tran_id, ocr_id=ocr_id, statusCd=current_status_cd))



if __name__ == "__main__":
    
    logger.success("FastAPI 애플리케이션 시작 준비 완료 (포트: 5001)")
    # APP_ENV가 basaaja인 경우에만 reload=True 추가
    app_env = os.getenv("APP_ENV")
    if app_env == "basaaja":
        # reload=True를 사용할 때는 import string 형태로 전달해야 함
        uvicorn.run("main:app", host="0.0.0.0", port=5001, log_config=None, reload=True)
    else:
        uvicorn.run(app, host="0.0.0.0", port=5001, log_config=None)