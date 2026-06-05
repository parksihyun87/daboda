# Missing Person CCTV VOD MVP

저장된 CCTV VOD 구간에서 특정 인물과 유사한 후보를 탐지하고, 후보 이미지와 타임라인 리포트를 생성하는 로컬 MVP입니다.

이 프로젝트는 실시간 RTSP 관제보다 **녹화 영상 파일 분석**에 초점을 둡니다. 사용자는 대상자 사진, 선택적 보행 영상, 실종 당일 착장 설명을 입력하고, 시스템은 YOLO 기반 사람 추적과 part-based Re-ID를 이용해 후보 track을 점수화합니다.

## 프로젝트 개요

Missing Person CCTV VOD MVP는 실종자 또는 특정 인물의 기준 이미지를 등록한 뒤, CCTV 저장 영상에서 유사 인물 후보를 빠르게 좁히기 위한 분석 도구입니다.

분석 결과는 확정 판정이 아니라 **유사도 기반 후보 목록**입니다. 저해상도 CCTV, 가림, 군중, 유사 복장 상황에서는 오탐과 누락이 발생할 수 있으므로 결과는 사람이 최종 검토해야 합니다.

## 주요 기능

- 대상자 등록
  - 정면, 측면, 후면 등 여러 인물 이미지 등록
  - OSNet 기반 Re-ID embedding 생성
  - full / upper / torso / lower / legs 단위 part embedding bank 생성

- 착장 기준 이미지 생성
  - Gemini, Flux, FASHN, placeholder provider 지원
  - 실종 당일 착장 설명 기반 이미지 생성
  - front / side / back 3면 착장 reference 생성 및 저장
  - 생성 이미지 품질 비교 평가

- CCTV VOD 분석
  - MP4 등 저장 영상 구간 분석
  - YOLO11 Pose 기반 사람 탐지
  - ByteTrack 또는 BoT-SORT 기반 local tracking
  - visible body parts 기반 part Re-ID scoring
  - 착장 색상 설명 기반 score boost
  - 선택적 gait 보조 점수 결합
  - 여러 CCTV 파일 병렬 분석

- 결과 리포트
  - 유사도 상위 후보 표시
  - threshold 통과 hit 표시
  - 후보 crop 이미지 저장
  - 시간순 타임라인 생성
  - PDF 리포트 export
  - CVAT XML 기반 탐지 정확도 평가

## 분석 파이프라인

```text
대상자 이미지 등록
  -> full / upper / torso / lower / legs part embedding 생성

착장 이미지 생성
  -> 실종 당일 복장 reference embedding 생성

CCTV VOD 분석
  -> YOLO11 Pose 사람 탐지
  -> ByteTrack / BoT-SORT local track 생성
  -> track별 visible part crop 수집
  -> OSNet embedding 추출
  -> 등록 대상 part bank와 cosine similarity 비교
  -> 색상 보정 및 선택적 gait score 결합
  -> 후보 / hit / 타임라인 / 리포트 생성
```

## 기술 스택

- Python
- Streamlit
- Ultralytics YOLO
- ByteTrack / BoT-SORT
- torchreid / OSNet
- FAISS
- OpenCV
- ReportLab
- Gemini API
- Flux / Diffusers

## 설치 및 실행

```powershell
cd c:\workplace\code\python\speed_yulam

python -m pip install -r missing_person_mvp\requirements.txt

copy missing_person_mvp\.env.example missing_person_mvp\.env
# .env에 GEMINI_API_KEY 등 필요한 값을 입력

$env:KMP_DUPLICATE_LIB_OK="TRUE"

streamlit run missing_person_mvp/streamlit_app.py
```

브라우저에서 접속합니다.

```text
http://localhost:8501
```

## 주요 환경 변수

```env
GEMINI_API_KEY=
FASHN_API_KEY=
HF_TOKEN=

YOLO_MODEL=yolo11n-pose.pt
YOLO_FALLBACK_MODEL=yolo11n.pt
YOLO_TRACKER=bytetrack.yaml

REID_THRESHOLD=0.72
SAMPLE_EVERY=5
MAX_WORKERS=4

IMAGE_PROVIDER_ORDER=gemini,flux,fashn,placeholder
FLUX_ENABLED=true
FLUX_MODEL_ID=black-forest-labs/FLUX.1-schnell
FLUX_MODE=inpaint
```

`YOLO_TRACKER`는 `bytetrack.yaml` 또는 `botsort.yaml`로 설정할 수 있습니다.

## 디렉터리 구조

```text
missing_person_mvp/
  streamlit_app.py         # Streamlit UI
  config.py                # 환경 변수 및 경로 설정
  models.py                # dataclass 모델

  pipeline/
    register.py            # 대상자, 착장, gait embedding 등록
    analyze_vod.py         # CCTV VOD 분석
    generate_image.py      # 착장 이미지 생성
    report.py              # 타임라인 / PDF 리포트

  utils/
    track.py               # YOLO 탐지 및 tracking
    embedding.py           # OSNet embedding 추출
    parts.py               # part crop / visible part scoring
    gait.py                # skeleton gait embedding
    color_score.py         # 착장 색상 보정
    video.py               # 영상 frame / crop 유틸

  scripts/
    evaluate_detection.py  # CVAT/MOT 기반 탐지 평가
    evaluate_generation.py # 생성 이미지 품질 평가
    health_check.py        # 의존성 확인

  tests/                   # pytest 테스트
```

## 테스트 및 검증

```powershell
python -m pytest missing_person_mvp\tests -q
python -m compileall -q missing_person_mvp
python missing_person_mvp\scripts\health_check.py
```

## 평가 기능

CVAT XML 또는 MOT 형식의 정답 annotation을 사용해 탐지 결과를 평가할 수 있습니다.

평가 지표는 다음을 포함합니다.

- track recall
- first hit delay
- false hits per 10 minutes
- temporal IoU
- bbox IoU
- ID switch count
- fragmentation count
- runtime ratio
- total score

생성 이미지 평가는 Gemini / Flux 등의 provider 결과를 비교하고, 색상 일치도, 신원 보존도, 신체 완성도, artifact, Re-ID 활용도 기준으로 점수를 산출합니다.

## 현재 범위와 한계

- 현재 시스템은 저장된 CCTV VOD 파일 분석용 MVP입니다.
- RTSP 실시간 스트림 분석은 기본 사용 흐름이 아닙니다.
- tracker ID는 기본적으로 단일 영상/단일 카메라 내부 local ID입니다.
- 여러 카메라의 결과는 Re-ID 점수 기반으로 후보를 모으지만, 완전한 global multi-camera tracking 시스템은 아닙니다.
- 가림, 군중, 유사 복장, 저해상도 CCTV에서는 오탐과 누락이 발생할 수 있습니다.
- 생성 착장 이미지는 검색 reference로 쓰기 전 수동 검토가 필요합니다.
- 분석 결과는 수사나 현장 판단을 보조하는 후보 정보이며, 최종 식별 근거로 단독 사용해서는 안 됩니다.

## 배포 참고

RunPod A5000 + Streamlit + Cloudflare 배포 가이드는 [`deploy.md`](deploy.md)에 정리되어 있습니다.
