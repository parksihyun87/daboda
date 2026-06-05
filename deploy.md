# vitadigna.com 배포 가이드 (RunPod A5000 Secure + Cloudflare)

도메인: `vitadigna.com` (가비아 구매)  
서버: RunPod RTX A5000 Secure (GPU, Docker 기반)  
심사일: 2026-06-12 / 유지 기간: ~2026-06-26

---

## 1. 목표 구조

```
vitadigna.com (가비아)
  → 네임서버: Cloudflare
  → CNAME @  →  {pod-id}-8501.proxy.runpod.net  (Proxied 🟠)
  → RunPod 포드 (RTX A5000, Ubuntu)
       → docker run vitadigna ... --server.port 8501
       → missing_person_mvp/streamlit_app.py
```

HTTPS는 Cloudflare가 자동 처리 — certbot 불필요.

---

## 2. 예상 비용

| 항목 | 단가 | 15일 예상 |
|---|---:|---:|
| RunPod A5000 Secure 컴퓨트 | $0.74/hr | 켤 때만 과금, **~$14** |
| 포드 네트워크 볼륨 50GB | $0.07/GB/월 | ~$1.75 |
| Cloudflare DNS | 무료 | $0 |
| **합계** | | **~$16** |

> 포드는 쓸 때만 Start — 평소에는 Stop으로 컴퓨트 과금 없음.  
> 네트워크 볼륨은 Stop 상태에도 과금됨. Docker 재시작 시 데이터 유지.

---

## 3. 준비물

- [ ] RunPod 계정 (runpod.io)
- [ ] Cloudflare 계정 (cloudflare.com, 무료)
- [ ] `GEMINI_API_KEY`
- [ ] Git 저장소 URL (코드 클론용)
- [ ] 가비아 도메인 관리 권한

---

## 4. 가비아 → Cloudflare 네임서버 이전

### 4-1. Cloudflare에 도메인 추가

1. cloudflare.com 로그인 → **Add a Site**
2. `vitadigna.com` 입력 → **Free 플랜** 선택
3. Cloudflare가 기존 DNS 레코드 자동 스캔 → **Continue**
4. Cloudflare가 네임서버 2개를 알려줌 (예시):

```
arnold.ns.cloudflare.com
donna.ns.cloudflare.com
```

### 4-2. 가비아에서 네임서버 변경

1. 가비아 로그인 → **My가비아** → **도메인** → `vitadigna.com` 선택
2. **네임서버 변경** 메뉴 진입
3. 기존 가비아 네임서버 삭제 후 Cloudflare 네임서버 2개 입력
4. 저장

> DNS 전파: 보통 10분~수 시간, 최대 24시간

### 4-3. 전파 확인

```powershell
nslookup -type=ns vitadigna.com
# cloudflare 네임서버가 나오면 완료
```

---

## 5. RunPod 포드 생성

1. runpod.io 로그인 → **Pods** → **+ Deploy**
2. **Secure Cloud** 탭 선택
3. GPU 선택: **RTX A5000** (24GB VRAM, ~$0.74/hr)
4. 템플릿: **RunPod PyTorch 2.x** (Ubuntu 22.04 + CUDA 12.x)
5. 스토리지:
   - **Container Disk: 30GB** (이미지 + 모델 가중치)
   - **Network Volume: 20GB** (data/ 디렉토리 — Stop 후에도 유지)
6. 포트 설정:
   - **HTTP 8501** 노출 (Streamlit)
7. **Deploy** 클릭

포드 생성 후 **pod-id** 확인:

```
pod-id: abc123xyz
Streamlit URL: https://abc123xyz-8501.proxy.runpod.net
```

---

## 6. Cloudflare DNS 설정

Cloudflare 대시보드 → **DNS** → **Add record**:

```
Type:    CNAME
Name:    @
Target:  abc123xyz-8501.proxy.runpod.net
Proxy:   ON (주황 구름 🟠)  ← 반드시 ON
TTL:     Auto
```

저장 후 잠시 기다리면 `https://vitadigna.com` 이 포드로 연결됨.

> `www` 도 연결하려면 동일하게 Name: `www` 로 레코드 추가.

---

## 7. 환경 세팅 (RunPod 포드 내부)

> **RunPod 포드 자체가 Docker 컨테이너**이므로 내부에서 `docker build`를 하지 않습니다.
> 네트워크 볼륨(`/workspace`)에 한 번 설치하면 Stop/Start 후에도 유지됩니다.

포드 → **Connect** → **Start Web Terminal** 또는 SSH 접속.

### 7-1. 코드 클론

```bash
cd /workspace
git clone https://github.com/YOUR_ORG/YOUR_REPO.git app
cd app
```

### 7-2. 의존성 설치 (첫 1회, 약 15~20분)

```bash
# Cython은 torchreid 빌드 의존성이므로 먼저 설치
pip install Cython
pip install -r missing_person_mvp/requirements.txt
```

### 7-3. .env 작성

```bash
cat > missing_person_mvp/.env << 'EOF'
FLUX_ENABLED=false
CATVTON_ENABLED=false
IMAGE_PROVIDER_ORDER=gemini,fashn,placeholder
GEMINI_API_KEY=여기에_키_입력
FASHN_API_KEY=
DATA_DIR=/workspace/app/missing_person_mvp/data
RESULTS_DIR=/workspace/app/missing_person_mvp/data/results
REID_THRESHOLD=0.72
SAMPLE_EVERY=5
MAX_WORKERS=4
EOF
```

> ⚠️ `SAMPLE_EVERY=5` 고정 — 8로 올리면 짧게 등장하는 정답 인물을 놓침

### 7-4. 데이터 디렉토리 생성

```bash
mkdir -p missing_person_mvp/data/db
mkdir -p missing_person_mvp/data/results
mkdir -p missing_person_mvp/data/footage
mkdir -p missing_person_mvp/data/video/cvat_xml
```

### 7-5. Streamlit 실행 (screen 백그라운드)

```bash
screen -S vitadigna

cd /workspace/app
KMP_DUPLICATE_LIB_OK=TRUE python -m streamlit run missing_person_mvp/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.maxUploadSize 500

# Ctrl+A, D 로 screen 분리 (앱은 계속 실행)
```

### 7-6. 실행 확인

```bash
curl http://localhost:8501      # 200 OK 확인
screen -r vitadigna             # 로그 재확인
```

### 7-7. 포드 재시작 후 재기동 (Stop → Start 후)

설치는 네트워크 볼륨에 유지됨. 재기동만 하면 됩니다:

```bash
screen -S vitadigna
cd /workspace/app
KMP_DUPLICATE_LIB_OK=TRUE python -m streamlit run missing_person_mvp/streamlit_app.py \
  --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.maxUploadSize 500
```

---

## 8. 운영 중 관리

```bash
# screen 세션 목록
screen -ls

# 앱 로그 확인
screen -r vitadigna

# 앱 재시작 (screen 안에서 Ctrl+C 후 재실행)
screen -r vitadigna   # 접속
# Ctrl+C 로 중지 후 7-5 명령어 다시 실행

# 코드 업데이트 후 재시작
cd /workspace/app && git pull
# screen 안에서 재실행
```

---

## 9. HTTPS / 도메인 확인

브라우저에서:

```
https://vitadigna.com
```

또는 curl:

```bash
curl -I https://vitadigna.com
# HTTP/2 200 이 나오면 성공
```

---

## 10. 데모 플로우 확인

심사 전 반드시 순서대로 확인:

1. **인물 등록** (페이지 1) — 정면·측면·후면 사진 업로드
2. **착장 이미지 생성** (페이지 2) — Gemini로 3면 생성, 피쳐 자동 저장 확인
3. **CCTV 분석** (페이지 3)
   - 데모 영상(🎬) 선택
   - 색상 설명: `노란색 상의에 검은색 바지` 자동 채움 확인
   - `sample_every=5`, 종료시각 전체 확인
   - 분석 시작 → 약 4~5분 → 후보 8명 표시 확인
   - 정답 1위 점수 ~86% 확인 (gait OFF, threshold 0.80)
   - 정확도 평가 XML 자동 연결 확인

Flux 비활성화 확인:

```bash
grep FLUX_ENABLED /workspace/app/missing_person_mvp/.env
# FLUX_ENABLED=false 가 나와야 함
```

---

## 11. 포드 관리 (비용 절약)

| 상황 | 조치 | 과금 |
|---|---|---|
| 심사 당일 | **Start** | 컴퓨트 + 볼륨 |
| 평소 대기 | **Stop** | 볼륨만 ($0.05/일) |
| 심사 완전 종료 | **Terminate** | 없음 |

> Stop 후 재Start 시 7-7 절의 재기동 명령어 실행.  
> `/workspace` 는 네트워크 볼륨이라 Stop/Start 후에도 코드·설치·데이터 유지됨.  
> Terminate하면 볼륨도 삭제 — 중요 데이터 먼저 백업.

---

## 12. 심사 종료 후 정리

```bash
# 결과 데이터 백업이 필요하면 포드 Stop 전에 로컬로 다운로드:
# RunPod Web Terminal의 파일 다운로드 기능 또는 scp 사용
scp -P {포드_SSH_포트} root@{포드_IP}:/workspace/app/missing_person_mvp/data ./backup_data
```

RunPod 대시보드:
1. 포드 **Terminate** → 컴퓨트 + 볼륨 과금 완전 중단

Cloudflare:
- 도메인을 계속 쓸 예정이면 DNS 레코드만 수정
- 가비아 네임서버를 원래대로 돌리려면 가비아에서 네임서버 변경

---

## 13. 운영 중 주의

- 실제 개인정보, 실종자 원본 사진은 심사 후 `/workspace/data/db` 정리
- RunPod Budget Alert 설정 권장 (대시보드 → Billing)
- `.env`의 API 키를 git에 올리지 않는다 (`.gitignore` 확인)
- `SAMPLE_EVERY=5` 변경하지 말 것 — 8 이상은 정답 인물 recall 저하
