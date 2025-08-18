#-*- coding: utf-8 -*-
# 2025-08-10 20:50:00: guid를 generate_guid()로 생성하도록 수정
# 2025-05-07 23:38:00: OpenOcr 클래스 생성 및 관련 함수 이동
# 2025-05-11 15:20:00: OCR 결과 RestAPI 전송 기능 추가
import easyocr
import cv2, os, time,  json
import numpy as np
import re
import random
from loguru import logger
from typing import Dict, Any, List

import onnxruntime as ort
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import demo_postprocess, multiclass_nms

import difflib
from jamo import h2j
from common import CommonUtil
import requests
import gc, os
from models import StringINDTO

from GeneRateUtils import GeneRateUtils,GeneRateUtilsFacade

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """이미지를 angle(도)만큼 회전시켜 잘리지 않도록 확장."""
    (h,  w)   = image.shape[:2]
    center    = (w // 2, h // 2)
    M         = cv2.getRotationMatrix2D(center, angle, 1.0)
    

    cos, sin  = abs(M[0, 0]), abs(M[0, 1])
    new_w     = int((h * sin) + (w * cos))
    new_h     = int((h * cos) + (w * sin))

    M[0, 2]  += (new_w / 2) - center[0]
    M[1, 2]  += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,   # 가장자리 보간 대신
                          borderValue=(255, 255, 255))       # → 흰색(배경색)으로 채움

def correct_document_skew(self,img_bgr: np.ndarray, tran_id: str) -> np.ndarray:
    """
    주민등록등본처럼 가로 줄이 많은 문서를 deskew.

    Parameters
    ----------
    img_bgr : np.ndarray
        원본 BGR 이미지

    Returns
    -------
    np.ndarray
        기울기 보정이 완료된 BGR 이미지
    """
    # 1) 전처리 ───────
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2) 가로선 강조 → 컨투어 찾기 ───────
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horiz_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    cnts, _      = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_pts  = None

    # 3) 가장 긴 가로선 탐색 ───────
    max_len   = 0.0
    max_angle = 0.0
    max_center  = None        # (cx, cy) 저장
    line_pts    = None

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < h * 10:          # '가로선'이 아니면 건너뜀
            continue

        rect   = cv2.minAreaRect(c)      # (center, (w, h), angle)
        angle  = rect[2]
        width, height = rect[1]

        # 세로가 더 길게 나왔을 때 각도 보정
        if width < height:
            angle += 90

        # 길이 계산 (boundingRect 폭 또는 실제 hypotenuse 중 택 1)
        if w > max_len:
            max_len, max_angle = w, angle
            max_center        = rect[0]     # (cx, cy) 실좌표
            # 엔드포인트는 중심에서 ±½·길이·(cosθ, sinθ)
            half_len = w / 2.0
            theta    = np.deg2rad(max_angle)
            dx, dy   = half_len * np.cos(theta), half_len * np.sin(theta)
            cx, cy   = max_center
            line_pts = ( (int(cx - dx), int(cy - dy)),
                         (int(cx + dx), int(cy + dy)) )

    # 4) 각도 보정 로직 ───────
    # 180° 근처 각도는 ±180°로 정규화
    if abs(max_angle) > 90:
        max_angle = max_angle - 180 if max_angle > 0 else max_angle + 180

    # 4-B) 라인 시각화 & 저장 ─────── (운영 환경이 아닌 경우에만 저장 )
    if line_pts is not None and self.app_env.upper() != "PROD":
        debug_img = img_bgr.copy()
        cv2.line(debug_img, line_pts[0], line_pts[1], (0, 0, 255), 2)

        try:
            fname = CommonUtil.getLogName(tran_id, "line_deskew")
            out_dir = os.path.join(self.output_dir, f"{fname}.jpg")
            if out_dir:
                cv2.imwrite(out_dir, debug_img)

        except Exception as e:
            logger.warning(f"[Deskew-v2] line image save failed: {e}")

    # 5) 회전 (±0.3° 이하면 무시) ───────
    if abs(max_angle) <= 0.3:
        logger.info(f"[Deskew-v2] max_angle: {max_angle} 보정 무시 대상")
        return img_bgr
    # 두 번째 구현은 +각도 그대로 사용했으므로 부호 변경 없이 회전
    return rotate_image(img_bgr, max_angle)

def is_hangul(char: str) -> bool:
    # "가(AC00)" ~ "힣(D7A3)" 범위 확인
    return '가' <= char <= '힣'

def jamo_split_syllables(s: str) -> str:
    """문자열 s의 한글 음절을 초성/중성/종성으로 분리하여 반환"""
    result = []
    for char in s:
        if is_hangul(char):
            # h2j를 써서 음절 -> 초/중/종(jamo)
            result.append(h2j(char))
        else:
            # 한글이 아니면 그대로 사용
            result.append(char)
    return ''.join(result)

def correct_family_relation(ocr_text: str, candidates: list, threshold: float = 0.5):
    """OCR 추정 텍스트(ocr_text)에 대해, candidates 중 가장 유사한 문구를 찾고
       유사도가 threshold 이상이면 반환, 그렇지 않다면 원본 ocr_text 반환"""
    best_match = None
    best_ratio = 0.0

    # OCR 결과 자모 분리
    ocr_jamo = jamo_split_syllables(ocr_text)

    for candidate in candidates:
        candidate_jamo = jamo_split_syllables(candidate)
        ratio = difflib.SequenceMatcher(None, ocr_jamo, candidate_jamo).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    # [수정 - 2025-04-08] 임계값 이상이면 best_match 반환, 아니면 원본 그대로
    if best_ratio >= threshold:
        return best_match
    else:
        return ocr_text

# =======================
# [추가 - 2025-12-26] 이름 첫 글자 교정 함수 수정
# =======================
def correct_name_first_char(name: str, name_correction_map: dict) -> str:
    """이름의 첫 글자를 교정용 사전을 사용하여 교정합니다.
    
    Args:
        name (str): 원본 이름
        name_correction_map (dict): 이름 교정용 사전
        
    Returns:
        str: 첫 글자가 교정된 이름 (교정할 필요가 없으면 원본 반환)
    """
    if not name or len(name) < 2:
        return name
    
    first_char = name[0]
    if first_char in name_correction_map:
        corrected_first_char = name_correction_map[first_char]
        corrected_name = corrected_first_char + name[1:]
        logger.debug(f"[이름 교정] '{name}' → '{corrected_name}' (첫 글자: '{first_char}' → '{corrected_first_char}')")
        return corrected_name
    
    return name

# -- [추가 기능] 프로그램 수행시간 측정을 위한 함수들
def measure_time_start():
    return time.time()

def measure_time_end(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    # 초 단위로 반올림하여 출력 (필요에 따라 형식 변경 가능)
    logger.debug(f"TotalTime : {int(elapsed_time)}s")

class YOLOXRowDetector:
    """family_list 행 검출을 위한 YOLOX ONNX 추론 래퍼 (OpenOcr용)"""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (640, 640)
        self.score_thr = 0.60
        self.nms_thr = 0.45

        logger.success(f"YOLOXRowDetector 모델 로드 완료: {model_path}")

    def detect(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        """BGR 이미지에서 가족 목록 행들을 탐지하여 크롭된 이미지 배열로 반환"""

        # grayscale이면 BGR로 변환 (임시로직 추후 전처리에서 OCR 처리 직전으로 옮겨야 함)
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        
        img, ratio = preprocess(img_bgr, self.input_size)
        
        ort_outs = self.session.run(None, {self.input_name: img[None]})[0]
        preds = demo_postprocess(ort_outs, self.input_size)[0]
    
        if preds is None or preds.size == 0:
            return []

        boxes = preds[:, :4]
        obj = preds[:, 4:5]
        cls_p = preds[:, 5:]
        scores = obj * cls_p

        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy /= ratio

        dets = multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.nms_thr,
            score_thr=self.score_thr,
            class_agnostic=True,
        )
    
        if dets is None or dets.shape[0] == 0:
            return []

        dets = dets.astype(np.float32)  # (n,6)

        # y1 기준으로 정렬하여 위에서 아래 순서대로 처리
        dets_sorted = dets[np.argsort(dets[:, 1])]
        cropped_images = []

        for det in dets_sorted:
            x1, y1, x2, y2, score, _ = det
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), img_bgr.shape[1] - 1), min(int(y2), img_bgr.shape[0] - 1)
    
            if x1 < x2 and y1 < y2:
                crop = img_bgr[y1:y2, x1:x2].copy()   # 2025-06-01 22:36:00 view → copy 로 변경 (메모리 이슈)
                if crop.size > 0:
                    cropped_images.append(crop)
        
        return cropped_images
    
class Ocr:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.app_env = self.settings.get("APP_ENV")
        # self.input_dir 등은 process_images에서 인자로 받거나, settings에서 가져옴
        self.json_path = self.settings.get("OCR_JSON_PATH")
        self.output_dir = self.settings.get("OCR_OUTPUT_DIR")
        self.result_web_server_dir = self.settings.get("RESULT_WEB_SERVER_DIR")
        self.ocr_char_result_dir = self.settings.get("OCR_CHAR_RESULT_DIR")
        
        self.family_relations = ["본인","배우자","자녀"]
        self.const_ocr_data = None
        self.g_ocr_data = [] # 요청 처리 시 초기화 필요할 수 있음
        self.global_total_count = 0 # 요청 처리 시 초기화 필요할 수 있음
        self.global_recognized_count = 0 # 요청 처리 시 초기화 필요할 수 있음

        self.reader = easyocr.Reader(['ko'], gpu=False, download_enabled=False, model_storage_directory=self.settings.get("EASYOCR_STORAGE_DIR")) # 온프레미스 환경: GPU 미사용, 모델 다운로드 비활성화
        logger.success("EasyOCR Reader 로딩 완료.")
        # --- CLS 모델 초기화 ---
        self.CLS_MODEL_PATH = self.settings.get("CLS_MODEL_PATH")
        if not self.CLS_MODEL_PATH:
            logger.critical("[CLS] CLS_MODEL_PATH가 설정 파일(env.yaml)에 정의되지 않았습니다.")
            raise ValueError("CLS_MODEL_PATH not found in settings. Check env.yaml.")
        
        try:
            sess_opts = ort.SessionOptions()
            # env에서 옵션 읽기, 없으면 기본값 사용
            sess_opts.intra_op_num_threads = int(self.settings.get("CLS_INTRA_OP_NUM_THREADS", 4))
            sess_opts.inter_op_num_threads = int(self.settings.get("CLS_INTER_OP_NUM_THREADS", 1))
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self._cls_sess   = ort.InferenceSession(self.CLS_MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])
            self._cls_inname = self._cls_sess.get_inputs()[0].name
            logger.success(f"[CLS] 문서 분류 ONNXRuntime 세션 로딩 완료.")
        except Exception as e:
            logger.critical(f"[CLS] ONNXRuntime 세션 로딩 실패: {self.CLS_MODEL_PATH}, 오류: {e}")
            raise
        # --- CLS 모델 초기화 끝 ---


        # YOLOX 모델 경로도 settings에서 가져옴
        self.YOLOX_MODEL_PATH_ROW = self.settings.get("YOLOX_MODEL_PATH_ROW")  # 추가

        if not self.YOLOX_MODEL_PATH_ROW :  # ROW 경로도 확인
            logger.critical("[YOLOX] YOLOX_MODEL_PATH_*가 설정 파일에 정의되지 않았습니다.")
            # raise ValueError("YOLOX model path not found in settings.") # 필요시 에러 발생

        self._yolox_session_cache = {}
        # 초기 YOLOX 세션 로드는 get_yolox_session 호출 시 수행되도록 변경 가능
        # 또는 기본 DOC 타입("GA")으로 미리 로드


        self.YOLOX_INPUT_WH = (640, 640)
        self.YOLOX_SCORE_THR = 0.30
        self.YOLOX_NMS_THR = 0.45

        # YOLOXRowDetector 초기화 추가
        self.row_detector = YOLOXRowDetector(self.YOLOX_MODEL_PATH_ROW)

        # --- 관계·이름 패턴을 한 번만 준비 ---
        self._base_relationship_list = [
            "본인", "배우자", "아들", "딸", "사위", "며느리", "손자", "손녀",
            "아버지", "어머니", "계부", "계모", "시부", "시모", "장인", "장모",
            "할아버지", "할머니", "형", "오빠", "남동생", "여동생", "누나",
            "언니", "형수", "제수", "매형", "매제", "형부", "제부", "올케",
            "백부", "백모", "숙부", "숙모", "삼촌", "외삼촌", "고모", "고모부",
            "이모", "이모부", "사촌", "사촌의 배우자", "조카", "조카의 배우자",
            "동거인", "입양자", "양부", "양모", "미혼부", "자녀", "외손","부","모"
        ]
        self._name_pattern = re.compile(r'([가-힣]{2,4})(?=[^가-힣]|$)')

        # JSON 사전 파일들 로드
        self._load_dictionary_files()

    def _load_dictionary_files(self):
        """JSON 파일에서 사전 데이터를 로드합니다."""
        try:
            # 프로그램 실행 디렉토리에서 JSON 파일들 로드
            current_dir = os.getcwd()
            
            # relationship.json 로드
            relationship_path = os.path.join(current_dir, "relationship.json")
            if os.path.exists(relationship_path):
                with open(relationship_path, "r", encoding="utf-8") as f:
                    self.fallback_map = json.load(f)
                logger.success(f"관계 교정 사전 로드 완료: {relationship_path}")
            else:
                logger.warning(f"관계 교정 사전 파일이 없습니다: {relationship_path}. 빈 사전으로 초기화합니다.")
                self.fallback_map = {}
            
            # name.json 로드
            name_path = os.path.join(current_dir, "name.json")
            if os.path.exists(name_path):
                with open(name_path, "r", encoding="utf-8") as f:
                    self.name_first_char_correction_map = json.load(f)
                logger.success(f"이름 교정 사전 로드 완료: {name_path}")
            else:
                logger.warning(f"이름 교정 사전 파일이 없습니다: {name_path}. 빈 사전으로 초기화합니다.")
                self.name_first_char_correction_map = {}
                
        except Exception as e:
            logger.error(f"사전 파일 로드 중 오류 발생: {e}")
            # 오류 발생 시 빈 사전으로 초기화
            self.fallback_map = {}
            self.name_first_char_correction_map = {}

    # 2025-05-11 16:50:00: cls_document를 클래스 메서드로 변경
    def cls_document(self, img_bgr: np.ndarray) -> str:
        """
        EfficientNet-B0(ONNX) 기반 GA/JU 분류
        GA → 0, JU → 1 로 학습되었다고 가정
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_res - mean) / std
        img_tensor = np.transpose(img_norm, (2, 0, 1))[None, :]

        probs = self._cls_sess.run(None, {self._cls_inname: img_tensor})[0]
        pred  = int(np.argmax(probs, 1)[0])
        doc = "GA" if pred == 0 else "JU"
        self.DOC = doc 
        logger.info(f"[CLS] 분류 결과: {doc}, pred: {pred}")
        return doc
    


    def image_preprocess(self, image):
        """
        주민등록등본,가족관계증명서(공문서) OCR 정확도를 높이기 위한 전처리 파이프라인.
        """
        
        img = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) # 해상도 확대

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab) # LAB 채널 분리
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # CLAHE (국소 대비 증폭) 지역 대비 ↑ → 도장·작은 글자도 또렷
        L = clahe.apply(L)
        lab = cv2.merge((L, a, b)) # LAB 채널 병합 (보정된 밝기를 색상과 합쳐 원래 공간으로)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        #gray_final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 최종 Grayscale 반환 
        return img


    def ocr_process(self, processed_image, current_doc_type, tran_id):
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        # 2025-05-07 23:38:00: self.reader 사용하도록 수정, self.DOC 사용하도록 수정
        detected_info_list = []
        bounding_box_coords = []

         # YOLOXRowDetector를 사용하여 가족 목록 행들을 탐지
        row_detect_array = self.row_detector.detect(processed_image)
        
        if not row_detect_array:
            logger.info("[ocr_process] YOLOXRowDetector에서 행을 탐지하지 못했습니다.")
            return detected_info_list, bounding_box_coords

        # processed_image 전체 크기를 bounding_box_coords로 사용
        height_p, width_p = processed_image.shape[:2]
        bounding_box_coords.append((0, 0, width_p, height_p))
        
        relationship_list = self._base_relationship_list.copy()

        if current_doc_type == "GA": # 클래스 멤버 DOC 사용
            relationship_list.extend(["부", "모"])

        relationship = None
        name = None
        jumin = None
        

        # 이름(한글 2~4자)이며, 숫자가 포함되지 않은 경우만 매칭 (2025-06-01 18:00:00 추가)
        NAME_PATTERN = re.compile(r'([가-힣]{2,4})(?=[^가-힣]|$)')
        # 탐지된 각 행에 대해 OCR 수행 ( 메모리 이슈 방지를 위해 20개만 처리 )
        for idx, cropped_img in enumerate(row_detect_array[:20]): 
            logger.info(f"[ocr_process] 행 {idx+1} 처리 중...")

            # processed_image의 상단, 하단 10%씩 잘라내기 (이미지 보정 시 위 아래 다른 글자 오탐 차단하기 위함 )
            height, width = cropped_img.shape[:2]
            top = int(height * 0.10)
            bottom = int(height * 0.90)
            cropped_img = cropped_img[top:bottom, :]
            
            # crop된 이미지들 저장
            save_path = None  # 초기화
            try:
                save_log_name = CommonUtil.getLogName(tran_id, "crop")
                save_path = os.path.join(self.output_dir, f"{save_log_name}_{idx}.jpg")
                cv2.imwrite(save_path, cropped_img)
            except Exception as e:
                logger.error(f"crop 이미지 저장 실패: {save_path}, 에러: {e}")

            cropped_results = self.reader.readtext(cropped_img)
            merged_text = " ".join([res[1] for res in cropped_results])
            
            logger.debug(f"[DEBUG-ocr_process] 행 {idx+1} OCR 결과: {merged_text}")

            relationship = None
            name = None
            jumin = None

            tokens = merged_text.split()
            for token in tokens:
                logger.debug(f"[DEBUG-ocr_process] token: {token}")
                # ――― 1) 현재 반복 시작 시점의 관계 보존
                rel_before_iter = relationship  
                    
                if relationship is None:
                    for rel in relationship_list:
                        if rel == token:
                            relationship = rel
                            break

                if relationship is None:
                    corrected_token = correct_family_relation(token, self.family_relations, threshold=0.5)
                    if corrected_token in self.family_relations:
                        relationship = corrected_token
                    else:
                        fallback_candidate = self.fallback_map.get(token)  # self.fallback_map 사용
                        if fallback_candidate and fallback_candidate in self.family_relations:
                            relationship = fallback_candidate
                    
                # ――― 3) 관계가 막 확정되었다면(= 이 토큰이 관계) → 이름·주민번호 검출 건너뜀
                if rel_before_iter is None and relationship is not None:
                    continue

                m = self._name_pattern.match(token.strip())
                if m:
                    cand_name = m.group(1)
                    if cand_name not in relationship_list and name is None:
                        # 이름의 첫 글자 교정 적용 - 인스턴스 변수와 함께 전달
                        corrected_name = correct_name_first_char(cand_name, self.name_first_char_correction_map)
                        name = corrected_name

                match = re.match(r'^(\d{6})', token)
                if match:
                    jumin = match.group(1)

            person_info = {
                "세대주 관계": relationship,
                "성명": name,
                "주민등록번호": jumin,
                "전체인식문": merged_text
            }

            logger.debug(f"[DEBUG-ocr_process] => relationship: {relationship}, name: {name}, jumin: {jumin}")

            none_count = sum(1 for val in [relationship, name, jumin] if val is None)


            if none_count <= 1:
                detected_info_list.append(person_info)
            else:
                logger.debug(f"[DEBUG-ocr_process] 정보 불충분으로 무시됨: {none_count}개의 None 값 발견")
        
        del row_detect_array        # 2025-06-01 22:36:00 메모리 즉시 해제
        gc.collect()

        return detected_info_list, bounding_box_coords

    def ocr_char_result(self, image_path, detected_info_list):
        # 2025-05-07 23:38:00: 클래스 멤버 변수(global_recognized_count, g_ocr_data, const_ocr_data, output_dir) 사용
        filename_only = os.path.basename(image_path)

        self.g_ocr_data.append({
            "filename": filename_only,
            "인식결과": [
                {
                    "탐지된 이름": info.get("성명"),
                    "생년월일": info.get("주민등록번호"),
                    "관계 정보": info.get("세대주 관계")
                }
                for info in detected_info_list
            ]
        })


    def process_images(
        self,
        input_dir: str,
        log_dir: str,
        tran_id: str,
        ocr_id: str,
        host: str,
        type_: str,
        saved_file: str,
        target: str
    ):
        try:
            # 요청별 데이터 인자로 받음, input_dir도 인자로 받음
            self.g_ocr_data = [] # 요청 처리 시작 시 g_ocr_data 초기화
            self.global_recognized_count = 0 # 각 process_images 호출 시 인식된 수 초기화

            start_time = measure_time_start()

            current_doc_type = "GA" # 기본값, 이미지 분석 후 변경됨

            valid_extensions = self.settings.get(
                "VALID_IMAGE_EXTENSIONS"
            )
            image_files = [
                f for f in os.listdir(input_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            if not image_files:
                logger.warning(f"처리할 이미지가 입력 디렉토리에 없습니다: {input_dir}")
                return {"tran_id": tran_id, "ocr_id": ocr_id, "error": "No images found in input directory", "detect": []}
            
            ocr_detected_info_list_for_final_json = []

            for file_name in image_files:
                image_path = os.path.join(input_dir, file_name)
                logger.info(f"이미지 처리 시작: {file_name} (tran_id: {tran_id})")

                orig = cv2.imread(image_path)
                if orig is None:
                    logger.warning(f"이미지 파일을 읽을 수 없습니다: {image_path}")
                    continue

                # ------------------------------------------------------
                # 1) 문서 분류 (GA / JU)
                # ------------------------------------------------------
                
                orig_deskewed = correct_document_skew(self, orig,tran_id)
                current_doc_type = self.cls_document(orig_deskewed)
                # crop = orig

                preprocessd_image = self.image_preprocess(orig_deskewed)

                ocr_detected_info_list, ocr_bbox_coords = self.ocr_process(
                    preprocessd_image, current_doc_type, tran_id
                )

                ocr_detected_info_list_for_final_json = ocr_detected_info_list
                self.ocr_char_result(image_path, ocr_detected_info_list)

                if self.output_dir:
                    base_name, ext = os.path.splitext(file_name)
                    save_log_name = CommonUtil.getLogName(tran_id, "deskew")
                    save_path = os.path.join(self.output_dir, f"{save_log_name}{ext}")
                    cv2.imwrite(save_path, orig_deskewed)

                break # 한 파일만 처리

            # self.global_total_count는 load_json_data에서 설정 (검증 데이터 기준)
            if self.const_ocr_data and self.global_total_count > 0:
                final_rate = int((self.global_recognized_count / self.global_total_count) * 100)
            else:
                final_rate = 0

            if self.ocr_char_result_dir:
                if not os.path.exists(self.ocr_char_result_dir):
                    logger.error(f"OCR 문자 인식 결과 저장 디렉터리({self.ocr_char_result_dir})가 존재하지 않습니다.")
                ocr_char_result_filename = CommonUtil.getLogName(tran_id, "char_result") + ".json"
                result_json_path = os.path.join(self.ocr_char_result_dir, ocr_char_result_filename)
                with open(result_json_path, "w", encoding="utf-8") as f:
                    json.dump(self.g_ocr_data, f, ensure_ascii=False, indent=4)
                logger.success(f"ocr_char_result json 저장: {result_json_path}")
            else:
                logger.warning(f"OCR 문자 인식 결과 저장 디렉터리({self.ocr_char_result_dir})가 설정되지 않았습니다.")

            detect_entries = [
                {"detect_relation": info.get("세대주 관계"), "detect_name": info.get("성명"), "detect_birth": (info.get("주민등록번호") or "")[:6]}
                for info in ocr_detected_info_list_for_final_json
            ]
            
            # detect_entries가 None이거나 빈 리스트일 수 있으니 조건 처리
            detect_str = ""
            if detect_entries:
                detect_str = f'"detect": {json.dumps(detect_entries, ensure_ascii=False)},\n'

            string_value = (
                f'"tran_id": "{tran_id}",\n'
                f'"ocr_id": "{ocr_id}",\n'
                f'"saved_file": "{saved_file}",\n'
                f'"host": "{host}",\n'
                f'"type": "{type_}",\n'
                f'"document_type": "{current_doc_type.lower()}",\n'
                f'{detect_str}'
                f'"target": {json.dumps([t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in target], ensure_ascii=False)}'
            )
            string_value = "{" + string_value + "}"

            # 1) 서버 IP를 지정하지 않고 사용 (내부 resolveHexServerIP 활용)
            gr = GeneRateUtilsFacade()
            guid = gr.generateGuid(self.__class__)  # 또는 type(self), 또는 특정 클래스
            sig  = gr.get16ByteKeyBySha256(guid, string_value)
                        
            # guid = generateGuid(self.__class__);
            # guid = generateGuid(string_value) # 이부분은 Java 의 Obj를 정확하게 파악해서 수정해야 함
            
            logger.debug(f"최종 GUID: {guid}")

            

           
            
            final_json_result = {
                "StringINDTO": StringINDTO(stringValue=string_value).model_dump() if hasattr(StringINDTO(stringValue=string_value), "model_dump") else dict(StringINDTO(stringValue=string_value))
            }
            # 2025-07-28 10:00:00 guid를 이용하여 원문에 대한 hash 생성
            try:
                logger.debug(f"Hash 생성 시도 - guid: '{guid}' (타입: {type(guid)}, 길이: {len(str(guid))})")
                logger.debug(f"final_json_result 타입: {type(final_json_result)}")
                
                # final_json_result를 JSON 문자열로 변환
                json_string = json.dumps(final_json_result, ensure_ascii=False)
                logger.debug(f"JSON 문자열 길이: {len(json_string)}")
                
                pfmGlobalNoHash = gr.get16ByteKeyBySha256(str(guid), json_string)
                logger.debug(f"pfmGlobalNoHash 생성 성공: {pfmGlobalNoHash}")
                
            except Exception as e:
                logger.error(f"pfmGlobalNoHash 생성 중 오류 발생: {e}")
                logger.error(f"guid 값: '{guid}', 타입: {type(guid)}")
                # 기본값으로 빈 문자열 또는 기본 해시 사용
                pfmGlobalNoHash = ""

            if self.result_web_server_dir and final_json_result.get("StringINDTO"):
                try:
                    headers = {'Content-Type': 'application/json', 'Authorization': pfmGlobalNoHash}
                    response = requests.post(self.result_web_server_dir, json=final_json_result, headers=headers, timeout=30)
                    response.raise_for_status()

                    # --- 로그 저장 로직 ---
                    CommonUtil.logCreate(self.settings, final_json_result, tran_id, type_, "result")

                    logger.success(f"OCR 결과 RestAPI 전송 성공 [{tran_id}]. 응답 코드: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"OCR 결과 RestAPI 전송 실패 [{tran_id}]: {e}")
                    if self.output_dir:
                        failed_api_data_path = os.path.join(self.output_dir, f"failed_api_submission_{tran_id}.json")
                        try:
                            with open(failed_api_data_path, "w", encoding="utf-8") as f_fail:
                                json.dump(final_json_result, f_fail, ensure_ascii=False, indent=4)
                            logger.error(f"전송 실패 OCR 데이터 저장 [{tran_id}]: {failed_api_data_path}")
                        except Exception as dump_e: logger.error(f"전송 실패 데이터 저장 중 오류 [{tran_id}]: {dump_e}")
            elif not self.result_web_server_dir:
                logger.critical(f"RESULT_WEB_SERVER_DIR 미설정, API 전송 건너뜀 [{tran_id}].")
            elif not final_json_result.get("detect"):
                logger.error(f"탐지된 OCR 결과 없음, API 전송 건너뜀 [{tran_id}].")

            measure_time_end(start_time)
            return final_json_result
        
        finally:                     # ← 무조건 실행
            self.g_ocr_data.clear()  # 메모리 즉시 해제
            gc.collect()             # 가비지 컬렉터 강제 실행