#!/usr/bin/env bash
# RunPod 재시작 후 Streamlit 앱 기동 스크립트
# 사용법: bash start.sh
# screen 세션 이름: vitadigna

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
SCREEN_NAME="vitadigna"

# 기존 screen 세션 정리
if screen -ls | grep -q "$SCREEN_NAME"; then
    echo "[start.sh] 기존 '$SCREEN_NAME' screen 세션 종료..."
    screen -S "$SCREEN_NAME" -X quit || true
    sleep 1
fi

echo "[start.sh] '$SCREEN_NAME' screen 세션 시작..."
screen -dmS "$SCREEN_NAME" bash -c "
    cd '$REPO_DIR'
    export KMP_DUPLICATE_LIB_OK=TRUE
    echo '[vitadigna] Streamlit 시작...'
    streamlit run missing_person_mvp/streamlit_app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --server.maxUploadSize 500
"

sleep 2
if screen -ls | grep -q "$SCREEN_NAME"; then
    echo "[start.sh] 앱 실행 중. 로그 보기: screen -r $SCREEN_NAME"
    echo "[start.sh] URL: http://localhost:8501"
else
    echo "[start.sh] ERROR: screen 세션 시작 실패. 직접 실행해주세요:"
    echo "  KMP_DUPLICATE_LIB_OK=TRUE streamlit run missing_person_mvp/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false"
    exit 1
fi
