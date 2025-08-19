# Open OCR Server

이미지에서 텍스트를 추출하는 OCR(광학 문자 인식) 서버입니다. 웹 API를 통해 이미지를 받아서 텍스트를 추출하고 결과를 반환하는 시스템입니다.

## 🚀 빠른 시작 (Windows)

### ⚠️ Windows 환경 설정 (필수!)

**소스 코드 수정 없이 Windows에서 실행하려면 다음 경로들을 수정해야 합니다:**

#### 1. Mockup Server 경로 수정
**파일**: `util/mockup_server.py` (라인 30-31)
```python
# 기존 (Mac/Linux 경로)
IMAGE_DIR = "/Users/basaaja/Python/ocr_image/temp/server_image"
LOG_DIR_STR = "/Users/basaaja/Python/openocr_log_mockup"

# Windows용으로 수정
IMAGE_DIR = "../ocr_image/temp/server_image"
LOG_DIR_STR = "../openocr_log_mockup"
```

#### 2. 설정 파일 경로 수정
**파일**: `config/basaaja.env.yaml` (라인 20)
```yaml
# 기존 (Mac/Linux 경로)
MMS_NAS_DIR: /Users/basaaja/Python/ocr_image/temp/mms_nas

# Windows용으로 수정
MMS_NAS_DIR: ../ocr_image/temp/mms_nas
```

#### 3. OCR 입력/출력 디렉토리 수정
**파일**: `config/basaaja.env.yaml` (라인 25, 40)
```yaml
# 기존 (Mac/Linux 경로)
OCR_OUTPUT_DIR: /Users/basaaja/Python/ocr_image/temp/output_image
OCR_INPUT_DIR: "/Users/basaaja/Python/ocr_image/temp/ocr_input_temp"

# Windows용으로 수정
OCR_OUTPUT_DIR: ../ocr_image/temp/output_image
OCR_INPUT_DIR: "../ocr_image/temp/ocr_input_temp"
```

### 1️⃣ 환경 변수 설정
```powershell
$env:APP_ENV="basaaja"
```

### 2️⃣ 패키지 설치
```powershell
pip install -r requirements.txt
```

### 3️⃣ 서버 시작
```powershell
# Mockup Server 시작 (새 창)
cd util
python mockup_server.py

# Main Server 시작 (또 다른 새 창)
cd ..
python main.py
```

### 4️⃣ 서버 확인
- Mockup Server: http://localhost:5002
- Main Server: http://localhost:5001/docs

### 🛠️ 자동 시작 스크립트
```powershell
# start_servers.ps1 파일 생성 후 실행
.\start_servers.ps1
```

---

## 🏗️ 시스템 아키텍처

이 시스템은 **두 개의 서버**로 구성되어 있습니다:

### 1. **Mockup Server (포트 5002)**
- **역할**: 이미지 제공 + 결과 수신
- **기능**: 
  - OCR 처리할 이미지들을 웹에서 제공
  - OCR 처리 결과를 받아서 로그 파일로 저장

### 2. **Main Server (포트 5001)**
- **역할**: OCR 처리 + 비즈니스 로직
- **기능**:
  - 클라이언트 요청 처리
  - Mockup Server에서 이미지 다운로드
  - EasyOCR을 사용한 텍스트 인식
  - 결과를 Mockup Server로 전송

## 🔄 서버 간 통신 흐름

```
┌─────────────┐    HTTP POST    ┌─────────────┐    HTTP GET     ┌─────────────┐
│  클라이언트  │ ──────────────→ │ Main Server │ ──────────────→ │ Mockup Server│
│             │   OCR 요청      │   (5001)    │   이미지 요청   │   (5002)    │
└─────────────┘                 └─────────────┘                 └─────────────┘
                                         │                              │
                                         │                              │
                                         ▼                              │
                                ┌─────────────┐                         │
                                │ OCR 처리    │                         │
                                │ (EasyOCR)   │                         │
                                └─────────────┘                         │
                                         │                              │
                                         ▼                              │
                                ┌─────────────┐                         │
                                │ 결과 생성   │                         │
                                └─────────────┘                         │
                                         │                              │
                                         ▼                              │
┌─────────────┐    HTTP POST    ┌─────────────┐    HTTP POST    ┌─────────────┐
│  클라이언트  │ ←────────────── │ Main Server │ ──────────────→ │ Mockup Server│
│             │   즉시 응답     │   (5001)    │   결과 전송     │   (5002)    │
└─────────────┘                 └─────────────┘                 └─────────────┘
                                                                        │
                                                                        ▼
                                                               ┌─────────────┐
                                                               │ 로그 파일   │
                                                               │ 저장        │
                                                               └─────────────┘
```

## 📝 상세 처리 흐름

### 1. **클라이언트 요청**
클라이언트가 Main Server에 OCR 처리 요청을 보냅니다.

### 2. **Main Server 처리**
- 데이터 검증 및 OCR ID 생성
- 로그 저장
- 백그라운드에서 이미지 다운로드 및 OCR 처리 예약
- **즉시 응답 반환** (비동기 처리)

### 3. **백그라운드 이미지 다운로드**
Main Server가 Mockup Server에서 이미지를 다운로드합니다.

### 4. **OCR 처리**
EasyOCR을 사용하여 이미지에서 텍스트를 추출합니다.

### 5. **결과 전송**
처리된 결과를 Mockup Server로 전송합니다.

### 6. **로그 저장**
Mockup Server가 결과를 로그 파일로 저장합니다.

## 📋 상세 설정 및 실행 가이드

### 📋 사전 준비사항
1. Python이 설치되어 있어야 함
2. 필요한 패키지들이 설치되어 있어야 함
   ```powershell
   pip install -r requirements.txt
   ```
3. 프로젝트 폴더로 이동

### 📸 테스트 이미지 준비

1. **이미지 파일을 다음 폴더에 복사**:
   ```
   ..\ocr_image\temp\server_image\
   ```

2. **지원되는 이미지 형식**:
   - .jpg, .jpeg, .png, .bmp, .webp, .tif

3. **테스트 이미지 예시**:
   - 텍스트가 포함된 문서 이미지
   - 명확한 글씨체의 이미지
   - 적당한 해상도 (너무 크지 않게)

### 🗂️ 필요한 디렉토리 생성

```powershell
# 프로젝트 루트에서 실행
# 로그 디렉토리
New-Item -ItemType Directory -Path "..\openocr_log" -Force

# 이미지 다운로드 디렉토리
New-Item -ItemType Directory -Path "..\ocr_image\temp\download_image" -Force

# OCR 입력 디렉토리
New-Item -ItemType Directory -Path "..\ocr_image\temp\ocr_input_temp" -Force

# OCR 출력 디렉토리
New-Item -ItemType Directory -Path "..\ocr_image\temp\output_image" -Force

# OCR 문자 결과 디렉토리
New-Item -ItemType Directory -Path "..\ocr_image\temp\ocr_char_result" -Force

# EasyOCR 캐시 디렉토리
New-Item -ItemType Directory -Path "..\ocr_image\temp\easyocr_storage" -Force

# Mockup Server 이미지 디렉토리 (테스트용)
New-Item -ItemType Directory -Path "..\ocr_image\temp\server_image" -Force
```

### 🖥️ 서버 시작 (수동)

**Mockup Server 시작 (새 PowerShell 창)**:
```powershell
# 프로젝트 루트에서
cd util
python mockup_server.py
```

**예상 출력**:
```
INFO     | Starting image server with uvicorn.
INFO     | Uvicorn running on http://0.0.0.0:5002 (Press CTRL+C to quit)
```

**Main Server 시작 (또 다른 새 PowerShell 창)**:
```powershell
# 프로젝트 루트에서
python main.py
```

**예상 출력**:
```
INFO     | 프로그램 실행 환경: APP_ENV = basaaja
INFO     | 애플리케이션 시작: OCR 프로세서 초기화 중...
INFO     | OCR 프로세서 초기화 완료.
INFO     | FastAPI 애플리케이션 시작 준비 완료 (포트: 5001)
INFO     | Uvicorn running on http://0.0.0.0:5001 (Press CTRL+C to quit)
```



## 🛠️ 자동 시작 스크립트

### start_servers.ps1 (PowerShell 스크립트)

다음 내용으로 `start_servers.ps1` 파일을 생성하세요:

```powershell
# Open OCR Server 시작 스크립트
Write-Host "=== Open OCR Server 시작 ===" -ForegroundColor Cyan

# 1. 환경 변수 설정
$env:APP_ENV="basaaja"
Write-Host "환경 변수 설정 완료: APP_ENV = $env:APP_ENV" -ForegroundColor Green

# 2. 디렉토리 생성
Write-Host "필요한 디렉토리 생성 중..." -ForegroundColor Yellow
$directories = @(
    "..\openocr_log",
    "..\ocr_image\temp\download_image",
    "..\ocr_image\temp\ocr_input_temp",
    "..\ocr_image\temp\output_image",
    "..\ocr_image\temp\ocr_char_result",
    "..\ocr_image\temp\easyocr_storage",
    "..\ocr_image\temp\server_image"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "생성됨: $dir" -ForegroundColor Gray
}

# 3. Mockup Server 시작 (백그라운드)
Write-Host "Mockup Server 시작 중... (포트 5002)" -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd util; python mockup_server.py" -WindowStyle Normal

# 4. 잠시 대기
Write-Host "Mockup Server 시작 대기 중..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 5. Main Server 시작
Write-Host "Main Server 시작 중... (포트 5001)" -ForegroundColor Yellow
python main.py
```

**사용 방법**:
```powershell
# 스크립트 실행
.\start_servers.ps1
```

## 🔍 서버 상태 확인

### Mockup Server 확인:
```powershell
# 브라우저에서 확인
Start-Process "http://localhost:5002"

# 또는 PowerShell에서
Invoke-WebRequest -Uri "http://localhost:5002" -Method GET
```

### Main Server 확인:
```powershell
# 브라우저에서 확인 (API 문서)
Start-Process "http://localhost:5001/docs"

# 또는 PowerShell에서
Invoke-WebRequest -Uri "http://localhost:5001" -Method GET
```

## 📡 API 사용법

### OCR 요청 예시:

```powershell
# PowerShell에서 API 호출
$body = @{
    tran_id = "test_001"
    host = "http://127.0.0.1:5002"
    saved_file = "test_image.jpg"
    channel = "nais"
    target = "document"
    type = "ocr"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5001/openocr/child/create" -Method POST -Body $body -ContentType "application/json"
```

### 응답 형식:

**성공 응답**:
```json
{
  "tran_id": "test_001",
  "ocr_id": "20250115-143022-a1b2c3d4e5f6",
  "statusCd": "0000",
  "statusMessage": "정상적으로 등록 되었습니다."
}
```

**오류 응답**:
```json
{
  "tran_id": "test_001",
  "ocr_id": "N/A",
  "statusCd": "9001",
  "statusMessage": "JSON 데이터 형식 오류"
}
```

## 📊 상태 코드 설명

| 코드 | 의미 | 설명 |
|------|------|------|
| 0000 | 성공 | 정상 처리 완료 |
| 9001 | JSON 오류 | 데이터 형식이 잘못됨 |
| 9002 | 중복 요청 | 같은 tran_id로 이미 처리됨 |
| 9003 | 설정 오류 | 다운로드 설정 문제 |
| 9004 | 다운로드 실패 | 이미지 다운로드 실패 |
| 9006 | 확장자 오류 | 지원하지 않는 이미지 형식 |
| 9999 | 시스템 오류 | 예상치 못한 오류 |

## 🔧 문제 해결

### 환경 변수 문제:
```powershell
# 환경 변수 확인
Get-ChildItem Env: | Where-Object {$_.Name -eq "APP_ENV"}

# 다시 설정
$env:APP_ENV="basaaja"
```

### 포트 충돌 문제:
```powershell
# 포트 사용 확인
netstat -ano | findstr :5001
netstat -ano | findstr :5002

# 프로세스 종료 (PID는 위 명령어로 확인)
taskkill /PID [PID] /F
```

### 디렉토리 권한 문제:
```powershell
# 관리자 권한으로 PowerShell 실행 후 다시 시도
```

## 🛠️ 기술 스택

- **FastAPI**: 웹 API 프레임워크
- **EasyOCR**: 텍스트 인식 엔진
- **OpenCV**: 이미지 처리
- **Loguru**: 로깅 시스템
- **Pydantic**: 데이터 검증
- **Uvicorn**: ASGI 서버
- **Python**: 프로그래밍 언어

## 📁 프로젝트 구조

```
open-ocr-server/
├── main.py                 # 메인 서버 (포트 5001)
├── Ocr.py                  # OCR 처리 엔진
├── common.py               # 공통 유틸리티
├── models.py               # 데이터 모델
├── requirements.txt        # Python 패키지 목록
├── config/
│   └── basaaja.env.yaml    # 환경 설정 파일
├── util/
│   └── mockup_server.py    # Mockup 서버 (포트 5002)
└── README.md               # 이 파일
```

## 👨‍💻 개발자

Created By IT_개발파트

---

## 📝 초급 개발자를 위한 핵심 포인트

1. **환경 변수는 반드시 먼저 설정**: `APP_ENV=basaaja`
2. **서버 시작 순서**: Mockup Server → Main Server
3. **비동기 처리**: API는 즉시 응답하고, 실제 처리는 나중에 실행
4. **에러 처리**: 각 단계마다 오류를 잡아서 적절한 응답을 반환
5. **리소스 관리**: 임시 파일들은 사용 후 반드시 삭제
6. **로깅**: 모든 과정을 로그로 기록하여 문제 발생 시 추적 가능
7. **설정 관리**: 모든 설정은 외부 파일에서 관리하여 유연성을 제공

이 시스템은 **확장 가능하고 안정적인 OCR 서비스**를 제공하는 것이 목표입니다! 🚀 