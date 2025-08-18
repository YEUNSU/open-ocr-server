# common.py
# 2025-05-11 00:15:00 공통 기능을 담당하는 CommonUtil 클래스 생성 및 메서드 이동 (load_settings, logCreate)
# 2025-05-11 00:40:00 load_settings에 DOWNLOAD_IMAGE_DIR 경로 처리 추가
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys

import yaml
import pytz
from loguru import logger

class CommonUtil:
    """
    공통 유틸리티 기능을 제공하는 클래스
    """

    @staticmethod
    def _initialize_directory_path(base_dir: Path, path_from_config: Optional[str], default_fallback_suffix: str) -> str:
        """
        설정 파일의 경로 문자열을 처리하여 절대 경로로 만들고 디렉토리를 생성합니다.
        """
        if not path_from_config:
            # 기본 폴백 경로는 base_dir 하위에 생성하거나, 홈 디렉토리 등 절대 경로로 지정 가능
            path_obj = base_dir / f"default_{default_fallback_suffix}_dir"
            logger.error(f"설정에 경로가 없어 프로그램을 종료합니다: {path_obj}")
            sys.exit(1)
        else:
            path_obj = Path(path_from_config)

        if not path_obj.is_absolute():
            path_obj = (base_dir / path_obj).resolve()
        
        path_obj.mkdir(parents=True, exist_ok=True)
        return str(path_obj)

    @staticmethod
    def load_settings(base_dir: Path) -> Dict[str, Any]:
        """
        실행 환경에 따라 설정 파일을 로드하고 LOG_DIR 및 DOWNLOAD_IMAGE_DIR 경로를 절대 경로로 변환합니다.
        base_dir: 설정 파일 및 상대 경로 디렉토리의 기준이 되는 디렉토리 (보통 app 루트 디렉토리)
        """
        app_env = os.getenv("APP_ENV")
        if not app_env:
            logger.critical("APP_ENV 설정이 없어서 종료합니다. 설정 예시 ) export APP_ENV=prod")
            sys.exit(1)
        config_file_name = f"{app_env}.env.yaml"
        config_path = base_dir / "config" / config_file_name

        logger.debug(f"CommonUtil.load_settings 호출됨. APP_ENV: {app_env}, Config Path: {config_path}")

        if not config_path.exists():
            logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
            # prod가 아니어도 무조건 예외 발생시켜 종료
            raise FileNotFoundError(f"설정 파일 '{config_path}'을(를) 찾을 수 없습니다.")
            # 아래 코드는 더 이상 실행되지 않음
            # logger.warning(f"개발 환경 설정 파일 '{config_path}'을 찾을 수 없어 기본 설정을 사용합니다.")
            # log_dir_default = CommonUtil._initialize_directory_path(Path.home(), None, "openocr_log")
            # download_dir_default = CommonUtil._initialize_directory_path(Path.home(), None, "openocr_downloads")
            # default_settings = {
            #     "APP_NAME": f"OpenOCR-{app_env.capitalize()}",
            #     "LOG_DIR": log_dir_default,
            #     "DOWNLOAD_IMAGE_DIR": download_dir_default,
            #     "IMAGE_WEB_SERVER_DIR": "http://localhost:8000/images", # 기본 예시 웹 서버 주소
            #     "LOG_LEVEL": "DEBUG",
            # }
            # return default_settings

        with open(config_path, "r", encoding="utf-8") as f:
            settings_data = yaml.safe_load(f)

        # LOG_DIR 처리
        settings_data["LOG_DIR"] = CommonUtil._initialize_directory_path(
            base_dir, settings_data.get("LOG_DIR"), "logs"
        )

        settings_data["APP_ENV"] = os.getenv("APP_ENV")

        settings_data["MMS_NAS_DIR"] = settings_data.get("MMS_NAS_DIR")
        
        # DOWNLOAD_IMAGE_DIR 처리
        settings_data["DOWNLOAD_IMAGE_DIR"] = CommonUtil._initialize_directory_path(
            base_dir, settings_data.get("DOWNLOAD_IMAGE_DIR"), "downloads"
        )

         # IMAGE_WEB_SERVER_DIR은 URL이므로 별도 경로 변환 없음
        if "IMAGE_WEB_SERVER_DIR" not in settings_data:
            logger.warning("IMAGE_WEB_SERVER_DIR 설정이 YAML 파일에 없습니다. 기본값을 사용하거나 확인 필요.")
            settings_data["IMAGE_WEB_SERVER_DIR"] = "http://localhost:8000/images" 

        # --- [추가] OCR 관련 경로 설정 ---
        # 2025-05-11 01:30:00 OCR 관련 경로 설정 추가
        settings_data["OCR_INPUT_DIR"] = CommonUtil._initialize_directory_path(
            base_dir, settings_data.get("OCR_INPUT_DIR"), "ocr_input"
        )
        settings_data["OCR_OUTPUT_DIR"] = CommonUtil._initialize_directory_path(
            base_dir, settings_data.get("OCR_OUTPUT_DIR"), "ocr_output"
        )
        settings_data["OCR_CHAR_RESULT_DIR"] = CommonUtil._initialize_directory_path(
            base_dir, settings_data.get("OCR_CHAR_RESULT_DIR"), "ocr_char_result_dir"
        )
        # OCR_JSON_PATH는 파일이므로 디렉토리 생성 로직은 필요 없으나, 경로의 존재 유무는 OpenOcr 클래스에서 처리.
        # 여기서는 경로를 설정에서 가져오거나 기본값을 설정. 상대경로일 경우 base_dir 기준으로.
        ocr_json_path_str = settings_data.get("OCR_JSON_PATH", str(base_dir / "config" / "default_ocr_ju_data.json"))
        ocr_json_path_obj = Path(ocr_json_path_str)
        if not ocr_json_path_obj.is_absolute():
            ocr_json_path_obj = (base_dir / ocr_json_path_obj).resolve()
        settings_data["OCR_JSON_PATH"] = str(ocr_json_path_obj)
        # --- OCR 관련 경로 설정 끝 ---


        logger.info(f"설정 로드 완료. LOG_DIR: {settings_data['LOG_DIR']}, DOWNLOAD_IMAGE_DIR: {settings_data['DOWNLOAD_IMAGE_DIR']}")
        logger.info(f"OCR 설정: INPUT_DIR={settings_data['OCR_INPUT_DIR']}, JSON_PATH={settings_data['OCR_JSON_PATH']}, OUTPUT_DIR={settings_data['OCR_OUTPUT_DIR']}")

        return settings_data

    @staticmethod
    def getLogName( tran_id: str, log_mode: str) -> str:
        """
        로그 파일명 생성 규칙에 따라 파일명을 반환합니다.
        log_file_name_prefix = now_kst.strftime("%Y%m%d%H%M%S")
        로그 파일명 = {log_file_name_prefix}_{tran_id}_{log_mode}
        """
        kst_tz = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(kst_tz)

        log_file_name_prefix = now_kst.strftime("%Y%m%d%H%M%S")
        return f"{log_file_name_prefix}_{tran_id}_{log_mode}"

    @staticmethod
    def logCreate(settings: Dict[str, Any], data: Dict[str, Any], tran_id: str,log_mode: str, action:str,statusCd_override: Optional[str] = None) -> str:
        """
        수신한 JSON 데이터를 로그 파일로 저장합니다.
        statusCd_override가 있으면 해당 코드를 사용합니다.
        settings: LOG_DIR 이 포함된 설정 딕셔너리
        """
        logger.debug(f"CommonUtil.logCreate 호출됨: tran_id={tran_id}, statusCd_override={statusCd_override}")
        
        log_dir_str = settings.get("LOG_DIR")
        if not log_dir_str:
            logger.error("LOG_DIR 설정이 settings에 없습니다. 로그를 저장할 수 없습니다.")
            return "9999" 

        kst_tz = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(kst_tz)
        
        log_dir_base = Path(log_dir_str)
        
        # log_file_name_prefix 및 log_file_name 생성 규칙 적용
        log_file_name_prefix = CommonUtil.getLogName(tran_id, log_mode)
        log_file_name = f"{log_file_name_prefix}_{action}.json"
        log_file_path = log_dir_base / log_file_name

        for existing_file in log_dir_base.glob(f"*_{tran_id}_{log_mode}_{action}.json"):
            if existing_file.exists():
                logger.warning(f"이미 존재하는 tran_id 입니다: {tran_id}, 파일: {existing_file}")
                return "9002"

        try:
            with open(log_file_path, "w", encoding="utf-8") as f:
                log_data_to_write = data.copy()
                if statusCd_override:
                    log_data_to_write["statusCd"] = statusCd_override
                elif "statusCd" not in log_data_to_write :
                    log_data_to_write["statusCd"] = "0000"
                json.dump(log_data_to_write, f, ensure_ascii=False, indent=4)
            logger.info(f"로그 파일 저장 성공: {log_file_path}")
            return log_data_to_write.get("statusCd", "0000")
        except Exception as e:
            logger.error(f"로그 파일 저장 실패: {log_file_path}, 오류: {e}")
            return "9999"