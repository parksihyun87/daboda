"""실종자 CCTV 구간 분석 MVP — Streamlit UI.

실행:
    cd c:\\workplace\\code\\python\\speed_yulam
    streamlit run missing_person_mvp/streamlit_app.py
"""
from __future__ import annotations

import os
# OpenMP 런타임 중복(faiss + torch가 각자 libiomp5md.dll 링크) → Error #15 방지.
# torch/faiss import 이전에 반드시 설정해야 함.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import sys
import tempfile
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image as _PILImage

from pathlib import Path as _Path

from missing_person_mvp import config as _cfg
from missing_person_mvp.models import PersonProfile, SearchRequest
from missing_person_mvp.pipeline.analyze_vod import analyze_multiple_cameras
from missing_person_mvp.pipeline.generate_image import (
    compare_generate_outfit_image,
    generate_outfit_image,
    generate_outfit_image_three_views,
)
from missing_person_mvp.pipeline.register import (
    delete_outfit_preset,
    delete_person,
    embedding_states,
    get_embedding_source,
    get_profile,
    has_gait_embedding,
    list_outfit_presets,
    list_persons,
    load_outfit_preset,
    preset_label,
    register_gait_videos,
    register_person,
    update_outfit_embedding,
)
from missing_person_mvp.pipeline.report import build_timeline, export_pdf
from missing_person_mvp.utils.video import extract_frame_with_bbox, extract_video_clip, hms_to_seconds, seconds_to_hms
from missing_person_mvp.scripts.evaluate_generation import WEIGHTS as _EVAL_WEIGHTS
from missing_person_mvp.scripts.evaluate_generation import score_image as _score_image


@st.cache_resource(show_spinner=False)
def _engine_status() -> dict[str, tuple[bool, str]]:
    """탐지 엔진(OSNet Re-ID, YOLO) 실제 로드 상태. 1회만 로드해 캐시."""
    status: dict[str, tuple[bool, str]] = {}
    try:
        from missing_person_mvp.utils import embedding as _emb
        ok = _emb._load_osnet() is not None
        status["osnet"] = (ok, "OSNet 로드됨 (실제 Re-ID)" if ok else "더미 임베딩 — 매칭 불가")
    except Exception as exc:
        status["osnet"] = (False, f"OSNet 오류: {exc}")
    try:
        from missing_person_mvp.utils import track as _trk
        ok = _trk._load_yolo() is not None
        status["yolo"] = (ok, "YOLO 로드됨 (사람 탐지)" if ok else "YOLO 로드 실패 — 탐지 0건")
    except Exception as exc:
        status["yolo"] = (False, f"YOLO 오류: {exc}")
    return status


@st.cache_data(show_spinner=False)
def _video_meta(path: str) -> tuple[float, float]:
    """(fps, duration_sec) — 메타데이터만 빠르게 읽어 캐시."""
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(path)
        fps = cap.get(_cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        return (fps, (frames / fps) if fps > 0 else 0.0)
    except Exception:
        return (0.0, 0.0)


def _search_vector_status(person_id: str) -> tuple[str, str]:
    """대상자의 영상검색 벡터 준비 상태. (레벨, 설명) 반환. 레벨: ok|warn|none.

    평소(등록) 임베딩과 생성 착장(VTON) 임베딩을 구분해 표시한다.
    검색 시 생성 착장이 있으면 그게 우선 사용된다.
    """
    try:
        states = embedding_states(person_id)
    except Exception:
        return ("none", "임베딩 없음")
    if states["outfit"]:
        tag = "3면" if states["outfit_3view"] else "단면"
        return ("ok", f"생성 착장 사용 ({tag}) ← 검색에 적용됨")
    if states["base"]:
        return ("warn", "평소 사진만 — 착장 미생성 (생성 권장)")
    return ("none", "임베딩 없음 — 등록/생성 필요")


def _provider_status() -> dict[str, tuple[bool, str]]:
    return {
        "gemini": (
            bool(_cfg.GEMINI_API_KEY),
            "API KEY 설정됨" if _cfg.GEMINI_API_KEY else "GEMINI_API_KEY 미설정",
        ),
        "flux": (
            bool(_cfg.FLUX_ENABLED),
            f"{_cfg.FLUX_MODE} / {_cfg.FLUX_MODEL_ID}" if _cfg.FLUX_ENABLED else "FLUX_ENABLED=false",
        ),
        "fashn": (
            bool(_cfg.FASHN_API_KEY),
            "API KEY 설정됨" if _cfg.FASHN_API_KEY else "FASHN_API_KEY 미설정",
        ),
    }


_PROVIDER_LABELS = {
    "🔄 자동 (설정 순서대로)": None,
    "✨ Gemini 2.0 Flash": ["gemini", "placeholder"],
    "👕 Flux Inpaint (로컬 GPU)": ["flux", "placeholder"],
    "🎨 FASHN.ai": ["fashn", "placeholder"],
}

_SCORE_LABELS: dict[str, tuple[str, str]] = {
    "prompt_color_score":          ("프롬프트 색상 일치", "생성 이미지의 주요 색상이 착장 설명과 얼마나 맞는지"),
    "identity_preservation_score": ("신원 보존도",        "원본 인물 상반신 특징이 얼마나 유지됐는지"),
    "body_integrity_score":        ("신체 완성도",        "전신 비율·크기가 자연스럽게 표현됐는지"),
    "artifact_score":              ("이미지 품질",        "선명도와 채널 다양성 — 아티팩트 없음 기준"),
    "reid_utility_score":          ("Re-ID 활용도",      "OSNet으로 파트 임베딩 추출 가능 여부"),
}


_EVAL_COMPARE_METRICS: list[tuple[str, str, bool]] = [
    # (metric_key, 한국어명, higher_is_better)
    ("track_recall",            "탐지 재현율",          True),
    ("false_hits_per_10min",    "오탐/10분",            False),
    ("id_switch_count",         "ID 전환 횟수",         False),
    ("fragmentation_count",     "단절 횟수",            False),
    ("mean_first_hit_delay_sec","첫 탐지 지연(초)",     False),
    ("temporal_iou",            "시간 겹침 IoU",        True),
    ("mean_bbox_iou",           "BBox IoU 평균",        True),
    ("runtime_ratio",           "처리 속도 비율",       False),
    ("total_score",             "종합 점수",            True),
]


def _fmt(val, metric_key: str) -> str:
    if val is None:
        return "N/A"
    if metric_key == "track_recall":
        return f"{val * 100:.1f}%"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _show_single_eval(result: dict) -> None:
    m = result["metrics"]
    score = m["total_score"]
    badge = "🟢" if score >= 0.8 else ("🟡" if score >= 0.6 else "🔴")
    st.markdown(f"### {badge} 종합 점수: **{score:.3f}** / 1.000")
    st.progress(float(score))
    r1 = st.columns(4)
    r1[0].metric("탐지 재현율", f"{m['track_recall'] * 100:.1f}%",
                 delta="✅ 합격" if m["recall_gate_pass"] else "❌ 미달")
    r1[1].metric("오탐/10분", f"{m['false_hits_per_10min']:.2f}건")
    bv = m["mean_bbox_iou"]
    r1[2].metric("BBox IoU", f"{bv:.3f}" if bv is not None else "N/A")
    r1[3].metric("시간 겹침", f"{m['temporal_iou']:.3f}")
    r2 = st.columns(3)
    dv = m["mean_first_hit_delay_sec"]
    r2[0].metric("첫 탐지 지연", f"{dv:.1f}초" if dv is not None else "N/A")
    r2[1].metric("ID 전환", str(m["id_switch_count"]))
    r2[2].metric("단절 횟수", str(m["fragmentation_count"]))


def _show_comparison_eval(base: dict, gait: dict) -> None:
    import pandas as pd

    bm = base["metrics"]
    gm = gait["metrics"]

    rows = []
    for key, label, higher_better in _EVAL_COMPARE_METRICS:
        bv = bm.get(key)
        gv = gm.get(key)
        if bv is None or gv is None:
            delta_str = "N/A"
            arrow = ""
        else:
            try:
                delta = float(gv) - float(bv)
                improved = (delta > 0) if higher_better else (delta < 0)
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
                color = "🟢" if improved else ("🔴" if abs(delta) > 1e-4 else "⬜")
                delta_str = f"{color} {arrow}{abs(delta):.3f}"
            except (TypeError, ValueError):
                delta_str = "—"
                arrow = ""
        rows.append({
            "지표": label,
            "Re-ID만": _fmt(bv, key),
            "Re-ID + Gait": _fmt(gv, key),
            "변화": delta_str,
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # 핵심 5개 지표를 나란히 st.metric으로도 표시
    st.markdown("#### 핵심 지표 비교")
    for key, label, higher_better in _EVAL_COMPARE_METRICS[:5]:
        bv = bm.get(key)
        gv = gm.get(key)
        if bv is None or gv is None:
            continue
        try:
            delta = float(gv) - float(bv)
            improved = (delta > 0) if higher_better else (delta < 0)
            delta_label = f"{'▲' if delta > 0 else '▼'}{abs(delta):.3f} ({'개선' if improved else '저하'})"
            delta_color = "normal" if improved else "inverse"
        except (TypeError, ValueError):
            delta_label = None
            delta_color = "normal"
        col_b, col_g = st.columns(2)
        col_b.metric(f"{label} (기준)", _fmt(bv, key))
        col_g.metric(f"{label} (Gait)", _fmt(gv, key), delta=delta_label, delta_color=delta_color)


def _build_md_report(
    person_id: str,
    person_name: str,
    outfit_desc: str,
    results: dict,
    providers_ok: list[str],
    anno_vals: dict[str, str],
    overall_comment: str,
    chosen_provider: str | None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# VTON 생성 비교 평가 리포트",
        "",
        f"- **인물 ID**: {person_id} ({person_name})",
        f"- **착장 설명**: {outfit_desc}",
        f"- **평가 일시**: {now}",
        f"- **최종 선택**: {(chosen_provider or '미결정').upper()}",
        "",
        "## 점수 비교표",
        "",
        "| 항목 | 가중치 |" + "".join(f" {p.upper()} |" for p in providers_ok),
        "|:-----|:------:|" + "".join(":-------:|" for _ in providers_ok),
    ]
    for metric, (label, _) in _SCORE_LABELS.items():
        w = _EVAL_WEIGHTS.get(metric, 0)
        row = f"| {label} | {w * 100:.0f}% |"
        for p in providers_ok:
            row += f" {results[p]['scores'].get(metric, 0):.3f} |"
        lines.append(row)
    total_row = "| **총점** | 100% |"
    for p in providers_ok:
        total_row += f" **{results[p]['scores'].get('total', 0):.3f}** |"
    lines.append(total_row)

    if len(providers_ok) > 1:
        totals = {p: results[p]["scores"].get("total", 0) for p in providers_ok}
        winner = max(totals, key=lambda x: totals[x])
        lines.append(f"\n**자동 선정 승자**: {winner.upper()} (총점 {totals[winner]:.3f})")

    lines += ["", "## 항목별 사용자 조언", ""]
    for metric, (label, desc) in _SCORE_LABELS.items():
        score_parts = " | ".join(
            f"{p.upper()}: {results[p]['scores'].get(metric, 0):.3f}" for p in providers_ok
        )
        anno = anno_vals.get(metric, "").strip()
        lines += [
            f"### {label}",
            f"*{desc}*",
            f"- 점수: {score_parts}  (가중치 {_EVAL_WEIGHTS.get(metric, 0) * 100:.0f}%)",
        ]
        if anno:
            lines.append(f"- **조언**: {anno}")
        lines.append("")

    if overall_comment.strip():
        lines += ["## 종합 의견", "", overall_comment.strip(), ""]

    return "\n".join(lines)

# ─────────────────────────── 페이지 설정 ────────────────────────────
st.set_page_config(
    page_title="실종자 CCTV 분석 MVP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stButton > button { width: 100%; }
    .hit-card { background:#fff5f5; border-left:4px solid #cc0000;
                padding:0.6rem 1rem; border-radius:4px; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── 사이드바 ────────────────────────────────
with st.sidebar:
    st.title("🔍 실종자 CCTV 분석")
    st.markdown("---")
    page = st.radio(
        "메뉴",
        ["📖 How to Use", "👤 인물 등록", "👕 착장 이미지 생성", "📹 CCTV 분석 & 정확도 평가", "🧑 Who Am I"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ── 엔진 상태 (탐지가 되려면 둘 다 ✅ 여야 함) ──
    eng = _engine_status()
    osnet_ok, osnet_msg = eng["osnet"]
    yolo_ok, yolo_msg = eng["yolo"]
    all_ok = osnet_ok and yolo_ok
    with st.expander(f"⚙️ 엔진 상태 {'✅ 정상' if all_ok else '⚠️ 점검필요'}", expanded=not all_ok):
        st.caption(("✅ " if osnet_ok else "❌ ") + osnet_msg)
        st.caption(("✅ " if yolo_ok else "❌ ") + yolo_msg)
        if not all_ok:
            st.caption("→ 엔진이 꺼져 있으면 영상 탐지가 0건이 됩니다.")
    st.markdown("---")

    persons = list_persons()
    if persons:
        st.caption(f"등록 인물: {len(persons)}명")
        _vec_badge = {"ok": "🟢", "warn": "🟡", "none": "⚪"}
        for p in persons:
            lvl, desc = _search_vector_status(p["person_id"])
            st.caption(f"{_vec_badge[lvl]} {p['name']} ({p['person_id']}) — {desc}")
    else:
        st.caption("등록된 인물 없음")


# ═══════════════════════════════════════════════════════════════════
# 화면 0: How to Use
# ═══════════════════════════════════════════════════════════════════
if page == "📖 How to Use":
    st.header("📖 How to Use")

    st.markdown("""
    ## 기능 소개
    DABODA 프로젝트는 실종 대상자의 평소 이미지 및 걸음 정보를 등록하여, 실종 당일 착장 이미지를 생성하고 CCTV 영상에서 탐지하는 시스템입니다.
        
    1. **인물 등록**: 평소 전신 사진과 걸음걸이 동영상을 등록하여 인물 프로필을 생성합니다.
    2. **착장 이미지 생성**: 실종 당일 착장 설명을 입력하면 Gemini 2.0 Flash, Flux Inpaint(로컬 모델로 미적용)등 다양한 생성 모델로 수배 이미지를 생성합니다.
    3. **CCTV 분석**: 생성된 착장 이미지로 CCTV 영상을 분석하여 탐지 구간과 정확도를 평가합니다.
                

    좌측의 해당 메뉴들을 통하여 각 기능을 이용할 수 있습니다.
                """)

    # 기술 시연 영상
    _tutorial_dir = _cfg.DATA_DIR / "video" / "tutorial"
    _tutorial_vids = sorted(_tutorial_dir.glob("*.mp4")) if _tutorial_dir.exists() else []
    if _tutorial_vids:
        st.subheader("🎬 기술 시연 영상")
        st.markdown("""
    실제 기술 시연 영상은 실제 대상자의 사진을 통해 착장을 생성 후 분석을 하는 과정을 담은 동영상 입니다.
                """)
        with open(str(_tutorial_vids[0]), "rb") as _vf:
            st.video(_vf.read())
    else:
        st.info("🎬 시연 영상을 추가하려면 `data/video/tutorial/` 폴더에 MP4를 넣으세요.")

    # 탐지용 원본 동영상-1
    if _cfg.DEMO_VIDEO_PATH.exists():
        st.subheader("📹 탐지용 원본 동영상")
        st.markdown("""
    탐지용 원본 동영상은 대상자를 찾기 위해서 탐색하는 영상입니다.
                    
    *4초~25초 : 정답(target label)인물
                    
    *5분 31초~6분 6초 : 비정답 인물. 검은 모자에 검은 상하의 복장. 안경 미착용
                    
    *14분 43초~ 15분 29초: 비정답 인물. 초록색 비니 모자에 검은색 가디건, 회색 상의, 짙은 카키색 7부 반바지. 붉은 테 안경 착용
                    
    *18분 37초~18분 51초 : 비정답 인물. 짧은 점퍼 상의에 아디다스 검은색 바탕에 옆면 삼색줄 트레이닝 바지. 붉은 뿔테 안경 착용
                """)
        st.caption(f"파일명: {_cfg.DEMO_VIDEO_PATH.name}")
        with open(str(_cfg.DEMO_VIDEO_PATH), "rb") as _df:
            st.video(_df.read())
    else:
        st.info("📹 탐지용 영상이 없습니다. `data/video/` 폴더에 데모 영상을 추가하세요.")


# ═══════════════════════════════════════════════════════════════════
# 화면 1: 인물 등록
# ═══════════════════════════════════════════════════════════════════
if page == "👤 인물 등록":
    st.header("👤 인물 등록")
    st.caption("평시 전신 사진 5~10장을 등록합니다. 다각도(정면·측면·후면)가 좋습니다.")

    with st.form("register_form"):
        c1, c2 = st.columns(2)
        person_id = c1.text_input("대상자 ID *", placeholder="예: p001")
        name      = c2.text_input("이름 *", placeholder="예: 홍길동")
        contact   = st.text_input("보호자 연락처", placeholder="예: 010-0000-0000")
        v1, v2, v3 = st.columns(3)
        front_upload = v1.file_uploader("정면 전신 사진 *", type=["jpg", "jpeg", "png"], key="front_view_upload")
        side_upload = v2.file_uploader("측면 전신 사진 *", type=["jpg", "jpeg", "png"], key="side_view_upload")
        back_upload = v3.file_uploader("후면 전신 사진 *", type=["jpg", "jpeg", "png"], key="back_view_upload")
        uploaded  = st.file_uploader(
            "추가 전신 사진 업로드 (선택)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        gait_uploads_form = st.file_uploader(
            "🚶 보행 동영상 (선택, 2개 이상) — 등록과 함께 걸음걸이 특징 저장",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            key="gait_in_register_form",
            help="화면 밖에서 안으로 걸어 들어오는 전신 영상이 적합합니다. 한 번 등록하면 영구 저장되어 CCTV 분석에 자동 사용됩니다.",
        )
        submitted = st.form_submit_button("등록", type="primary")

    if submitted:
        if not person_id.strip():
            st.error("대상자 ID를 입력하세요.")
        elif not (front_upload and side_upload and back_upload):
            st.error("Flux inpainting을 위해 정면/측면/후면 전신 사진을 모두 업로드하세요.")
        else:
            with st.spinner("Part embedding bank 생성 중… (첫 등록은 30초 내외)"):
                saved_paths: list[str] = []
                # 영구 경로: /data/db/{person_id}/photos/ — 앱 재시작·pod 재기동 후에도 유지
                photo_dir = _cfg.DB_DIR / person_id.strip() / "photos"
                photo_dir.mkdir(parents=True, exist_ok=True)
                view_uploads = {"front": front_upload, "side": side_upload, "back": back_upload}
                view_paths: dict[str, str] = {}
                for view, f in view_uploads.items():
                    p = photo_dir / f"{view}_{f.name}"
                    p.write_bytes(f.read())
                    view_paths[view] = str(p)
                    saved_paths.append(str(p))
                for f in uploaded or []:
                    p = photo_dir / f.name
                    p.write_bytes(f.read())
                    if str(p) not in saved_paths:
                        saved_paths.append(str(p))
                try:
                    result = register_person(PersonProfile(
                        person_id=person_id.strip(),
                        name=name.strip(),
                        contact=contact.strip(),
                        image_paths=saved_paths,
                        view_image_paths=view_paths,
                    ))
                    _msg = f"✅ **{result.person_id}** 등록 완료 ({result.image_count}장, 5개 파트 임베딩 생성)"
                    # 보행 동영상이 함께 올라오면 걸음걸이도 같이 등록 (영구 저장)
                    _gait_ups = gait_uploads_form or []
                    if len(_gait_ups) >= 2:
                        with st.spinner("YOLO-Pose로 걸음걸이 골격 추출 중… (영상 길이에 따라 수 분)"):
                            _gpaths = []
                            for _gv in _gait_ups:
                                _gp = photo_dir / _gv.name
                                _gp.write_bytes(_gv.read())
                                _gpaths.append(str(_gp))
                            _gait_ok = register_gait_videos(result.person_id, _gpaths)
                        _msg += ("  · 🚶 걸음걸이 등록됨" if _gait_ok
                                 else "  · ⚠️ 걸음걸이 등록 실패(유효 골격 부족)")
                    elif len(_gait_ups) == 1:
                        _msg += "  · ⚠️ 걸음걸이는 2개 이상 필요(이번엔 미등록)"
                    st.success(_msg)
                    st.rerun()
                except Exception as e:
                    st.error(f"등록 실패: {e}")

    # 등록 인물 목록 + 삭제
    if persons:
        st.markdown("---")
        st.subheader("등록 인물 목록")
        for p in persons:
            col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
            col1.write(f"**{p['name']}** (`{p['person_id']}`)")
            col2.caption(p.get("contact", ""))
            gait_badge = "🚶✅" if has_gait_embedding(p["person_id"]) else "🚶⬜"
            col3.caption(gait_badge)
            if col4.button("삭제", key=f"del_{p['person_id']}"):
                delete_person(p["person_id"])
                st.rerun()

    # ── 보행 특징 등록 ─────────────────────────────────────────────
    if persons:
        st.markdown("---")
        st.subheader("🚶 보행 특징 등록 (선택)")
        st.caption(
            "평소 걸음걸이 동영상 **최소 2개** 등록 시 Gait 보조 기능을 활성화할 수 있습니다. "
            "화면 밖에서 안으로 걸어 들어오는 전신 영상이 적합합니다."
        )
        person_options_g = {f"{p['name']} ({p['person_id']})": p["person_id"] for p in persons}
        g_label = st.selectbox("보행 등록 대상", list(person_options_g.keys()), key="gait_person_sel")
        g_pid   = person_options_g[g_label]
        g_status = "✅ 보행 특징 등록됨" if has_gait_embedding(g_pid) else "⬜ 미등록"
        st.caption(g_status)

        gait_uploads = st.file_uploader(
            "보행 동영상 (최소 2개)",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            key="gait_video_upload",
        )
        if st.button("🚶 보행 특징 등록", key="btn_gait_reg"):
            if len(gait_uploads or []) < 2:
                st.error("최소 2개 동영상이 필요합니다.")
            else:
                with st.spinner("YOLO-Pose로 골격 추출 중… (영상 길이에 따라 수 분 소요)"):
                    g_tmp = Path(tempfile.mkdtemp())
                    g_paths = []
                    for gv in gait_uploads:
                        gp = g_tmp / gv.name
                        gp.write_bytes(gv.read())
                        g_paths.append(str(gp))
                    ok = register_gait_videos(g_pid, g_paths)
                if ok:
                    st.success(f"✅ {g_label} 보행 특징 등록 완료!")
                    st.rerun()
                else:
                    st.error("유효한 골격 프레임 부족 (최소 15프레임 필요). 더 긴 동영상 또는 선명한 전신 영상을 사용하세요.")


# ═══════════════════════════════════════════════════════════════════
# 화면 2: 착장 이미지 생성
# ═══════════════════════════════════════════════════════════════════
elif page == "👕 착장 이미지 생성":
    st.header("👕 실종 당일 착장 이미지 생성")
    st.caption("단일 생성 또는 Gemini vs Flux 비교 평가를 선택하세요.")

    if not persons:
        st.warning("먼저 인물을 등록해주세요.")
    else:
        person_options = {f"{p['name']} ({p['person_id']})": p["person_id"] for p in persons}
        chosen_label   = st.selectbox("대상자 선택", list(person_options.keys()))
        chosen_id      = person_options[chosen_label]
        chosen_name    = chosen_label.split(" (")[0]

        src = get_embedding_source(chosen_id)
        st.info(f"현재 쿼리 임베딩: **{src}**  ← 생성 후 자동 전환됩니다.")
        st.caption("ℹ️ 생성한 착장은 날짜·프롬프트와 함께 **피쳐로 자동 저장**됩니다. "
                   "검색 페이지에서 어떤 피쳐로 검색할지 골라 쓸 수 있어요 (재생성 불필요).")

        # 착장 프리셋 + 직접 입력
        _OUTFIT_PRESETS = {
            "1) 흰 무지 반팔티 + 청바지":              "흰 무지 반팔티에 청바지",
            "2) 노란색 반팔티 + 검은색 바지":           "노란색 반팔티에 검은색 바지",
            "3) 검은색 가디건 + 흰 티셔츠 + 검은색 청바지": "검은색 가디건에 흰 티셔츠, 검은색 청바지",
            "4) 허리 패딩 상의 + 흰색 바지":           "허리까지오는 패딩 상의, 흰색 바지",
            "✏️ 직접 입력":                           None,
        }
        preset_label = st.selectbox("착장 선택", list(_OUTFIT_PRESETS.keys()))
        preset_value = _OUTFIT_PRESETS[preset_label]

        if preset_value is not None:
            outfit_desc = preset_value
            st.caption(f"착장: **{outfit_desc}**")
        else:
            outfit_desc = st.text_area(
                "착장 설명 직접 입력",
                placeholder="예: 회색 후드티, 검정 트레이닝 바지, 흰색 운동화",
                height=90,
            )

        st.markdown("---")
        tab_single, tab_three_views, tab_compare = st.tabs(["🖼️ 단일 생성", "📐 3면 생성", "⚖️ Gemini vs Flux 비교 평가"])

        # ── 탭 1: 단일 생성 ─────────────────────────────────────
        with tab_single:
            status = _provider_status()
            provider_label_options: list[str] = []
            for label in _PROVIDER_LABELS:
                if label == "🔄 자동 (설정 순서대로)":
                    provider_label_options.append(label)
                    continue
                key = "flux" if "flux" in label.lower() else (
                      "fashn"   if "fashn"   in label.lower() else "gemini")
                ok, hint = status.get(key, (True, ""))
                badge = "✅" if ok else "⚠️"
                provider_label_options.append(f"{label}  [{badge} {hint}]")

            chosen_provider_label = st.radio(
                "제공자",
                provider_label_options,
                label_visibility="collapsed",
            )
            base_label = chosen_provider_label.split("  [")[0]
            selected_providers = _PROVIDER_LABELS[base_label]

            if base_label == "👕 Flux Inpaint (로컬 GPU)" and not status["flux"][0]:
                st.warning("Flux is disabled. Set FLUX_ENABLED=true to use local inpainting.")

            spinner_msg = {
                "🔄 자동 (설정 순서대로)": "이미지 생성 중… (설정 순서대로 시도)",
                "✨ Gemini 2.0 Flash":    "Gemini API 호출 중… (최대 30초)",
                "👕 Flux Inpaint (로컬 GPU)":  "Flux inpainting 로컬 추론 중… (최대 5분)",
                "🎨 FASHN.ai":            "FASHN.ai API 호출 중… (최대 60초)",
            }.get(base_label, "이미지 생성 중…")

            if st.button("착장 이미지 생성", type="primary", key="btn_single"):
                if not outfit_desc.strip():
                    st.error("착장 설명을 입력하세요.")
                else:
                    with st.spinner(spinner_msg):
                        try:
                            image, path, gen_errors = generate_outfit_image(
                                chosen_id, outfit_desc, selected_providers
                            )
                            for err in gen_errors:
                                st.warning(f"⚠️ {err}")
                            is_placeholder = "Temporary outfit image" in getattr(image, "_info", {}).get("comment", "") or gen_errors
                            if not gen_errors or not any("placeholder" in e.lower() for e in gen_errors):
                                st.image(image, caption="생성된 수배 이미지", use_container_width=False, width=320)
                            else:
                                st.image(image, caption="⚠️ 플레이스홀더 (실제 생성 실패)", use_container_width=False, width=320)
                            with open(path, "rb") as f:
                                st.download_button(
                                    "⬇️  이미지 다운로드",
                                    data=f.read(),
                                    file_name=Path(path).name,
                                    mime="image/jpeg",
                                )
                            if not gen_errors:
                                st.success("착장 임베딩 추출 완료 → CCTV 분석에 자동 사용됩니다.")
                        except Exception as e:
                            st.error(f"이미지 생성 실패: {e}")

        # ── 탭 2: 3면 생성 (정면/측면/후면) ───────────────────────────────
        with tab_three_views:
            st.caption(
                "정면·측면·후면 3개 각도에서 동일 착장으로 이미지를 생성합니다. "
                "CCTV 분석 시 최적 각도 자동 선택을 통해 정확도를 높입니다."
            )

            status = _provider_status()
            provider_label_options_3v: list[str] = []
            for label in ["✨ Gemini 2.0 Flash", "👕 Flux Inpaint (로컬 GPU)", "🔄 자동 (설정 순서대로)"]:
                if label == "🔄 자동 (설정 순서대로)":
                    provider_label_options_3v.append(label)
                    continue
                key = "flux" if "flux" in label.lower() else "gemini"
                ok, hint = status.get(key, (True, ""))
                badge = "✅" if ok else "⚠️"
                provider_label_options_3v.append(f"{label}  [{badge} {hint}]")

            chosen_provider_label_3v = st.radio(
                "제공자",
                provider_label_options_3v,
                label_visibility="collapsed",
                key="radio_3v",
            )
            base_label_3v = chosen_provider_label_3v.split("  [")[0]
            selected_providers_3v = _PROVIDER_LABELS.get(base_label_3v, None)
            if selected_providers_3v is None:
                selected_providers_3v = [key.lower() for key in ["gemini", "flux", "placeholder"] if key.lower() in _PROVIDER_LABELS.get(base_label_3v, [])]

            if st.button("3면 이미지 생성", type="primary", key="btn_three_views"):
                if not outfit_desc.strip():
                    st.error("착장 설명을 입력하세요.")
                else:
                    with st.spinner("3면 이미지 생성 중… (최대 3분)"):
                        try:
                            images_dict, paths_dict, gen_errors = generate_outfit_image_three_views(
                                chosen_id, outfit_desc, selected_providers_3v
                            )
                            for err in gen_errors:
                                st.warning(f"⚠️ {err}")

                            # 3개 이미지 나란히 표시
                            cols = st.columns(3)
                            for idx, view in enumerate(["front", "side", "back"]):
                                with cols[idx]:
                                    if view in images_dict:
                                        st.image(images_dict[view], caption=f"{'정면' if view == 'front' else '측면' if view == 'side' else '후면'} ({view})")
                                        with open(paths_dict[view], "rb") as f:
                                            st.download_button(
                                                f"⬇️  {view.upper()} 다운로드",
                                                data=f.read(),
                                                file_name=Path(paths_dict[view]).name,
                                                mime="image/jpeg",
                                                key=f"dl_3v_{view}",
                                            )

                            if not gen_errors:
                                st.success("3면 임베딩 저장 완료 → CCTV 분석에서 다중각도 매칭이 자동으로 활성화됩니다.")
                        except Exception as e:
                            st.error(f"3면 생성 실패: {e}")

        # ── 탭 3: 비교 생성 & 평가 ───────────────────────────────
        with tab_compare:
            st.caption(
                "동일 착장 설명으로 Gemini와 Flux를 **병렬** 실행 후 5개 지표로 자동 평가합니다. "
                "항목별 점수에 조언을 달고 MD로 저장하세요."
            )

            status = _provider_status()
            c_g, c_c = st.columns(2)
            c_g.caption(("✅" if status["gemini"][0] else "⚠️") + f" Gemini: {status['gemini'][1]}")
            c_c.caption(("✅" if status["flux"][0] else "⚠️") + f" Flux: {status['flux'][1]}")

            run_cmp = st.button(
                "⚡ 두 제공자 동시 생성",
                type="primary",
                key="btn_compare",
                disabled=not outfit_desc.strip(),
            )

            if run_cmp:
                if not outfit_desc.strip():
                    st.error("착장 설명을 입력하세요.")
                else:
                    with st.spinner("Gemini + Flux 병렬 실행 중... (최대 5분)"):
                        try:
                            cmp_raw = compare_generate_outfit_image(chosen_id, outfit_desc)
                            profile = get_profile(chosen_id)
                            base_pil = _PILImage.open(profile.image_paths[0]).convert("RGB")
                            scored: dict = {}
                            for pname, res in cmp_raw.items():
                                if isinstance(res, tuple):
                                    img, path = res
                                    scores = _score_image(pname, base_pil, img, outfit_desc)
                                    scored[pname] = {"image": img, "path": path, "scores": scores}
                                else:
                                    scored[pname] = {"error": str(res)}
                            st.session_state[f"cmp_{chosen_id}"] = {
                                "results": scored,
                                "outfit_desc": outfit_desc,
                            }
                            st.session_state.pop(f"cmp_chosen_{chosen_id}", None)
                        except Exception as exc:
                            st.error(f"비교 생성 실패: {exc}")

            # 결과 표시 (세션에 있고 outfit_desc가 일치할 때)
            cmp_state = st.session_state.get(f"cmp_{chosen_id}")
            if cmp_state and cmp_state.get("outfit_desc") == outfit_desc:
                results = cmp_state["results"]
                providers_ok = [p for p, r in results.items() if "error" not in r]
                providers_err = [p for p, r in results.items() if "error" in r]

                for p in providers_err:
                    st.warning(f"{p.upper()} 실패: {results[p]['error']}")

                if not providers_ok:
                    st.error("생성 성공한 제공자가 없습니다.")
                else:
                    # ── 이미지 비교 ──────────────────────────────
                    st.markdown("---")
                    profile = get_profile(chosen_id)
                    img_cols = st.columns(1 + len(providers_ok))
                    img_cols[0].image(
                        profile.image_paths[0],
                        caption="원본 인물 사진",
                        use_container_width=True,
                    )
                    for i, pname in enumerate(providers_ok):
                        img_cols[i + 1].image(
                            results[pname]["image"],
                            caption=f"{pname.upper()} 생성",
                            use_container_width=True,
                        )

                    # ── 점수 표 ───────────────────────────────────
                    st.subheader("📊 자동 평가 점수")
                    import pandas as pd
                    score_rows = []
                    for metric, (label, _) in _SCORE_LABELS.items():
                        row: dict = {
                            "항목": label,
                            "가중치": f"{_EVAL_WEIGHTS.get(metric, 0) * 100:.0f}%",
                        }
                        for p in providers_ok:
                            row[p.upper()] = f"{results[p]['scores'].get(metric, 0):.3f}"
                        score_rows.append(row)
                    total_row: dict = {"항목": "총점", "가중치": "100%"}
                    for p in providers_ok:
                        total_row[p.upper()] = f"{results[p]['scores'].get('total', 0):.3f}"
                    score_rows.append(total_row)
                    st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)

                    if len(providers_ok) > 1:
                        totals = {p: results[p]["scores"].get("total", 0) for p in providers_ok}
                        winner = max(totals, key=lambda x: totals[x])
                        st.success(f"🏆 자동 선정 승자: **{winner.upper()}**  총점 {totals[winner]:.3f}")

                    # ── 항목별 진행 바 ────────────────────────────
                    st.subheader("📈 항목별 비교")
                    for metric, (label, desc) in _SCORE_LABELS.items():
                        st.markdown(f"**{label}** — *{desc}*")
                        for pname in providers_ok:
                            val = results[pname]["scores"].get(metric, 0)
                            st.progress(float(val), text=f"{pname.upper()}: {val:.3f}")

                    # ── 사용자 조언 텍스트 ────────────────────────
                    st.markdown("---")
                    st.subheader("📝 항목별 사용자 조언")
                    st.caption(
                        "각 지표의 점수가 실제와 맞는지 확인하고 조언을 입력하세요. "
                        "저장 버튼을 누르면 MD 파일로 내보낼 수 있습니다."
                    )
                    anno_vals: dict[str, str] = {}
                    for metric, (label, desc) in _SCORE_LABELS.items():
                        score_info = "  |  ".join(
                            f"{p.upper()}: {results[p]['scores'].get(metric, 0):.3f}"
                            for p in providers_ok
                        )
                        st.markdown(f"#### {label}")
                        st.caption(f"{score_info}  (가중치 {_EVAL_WEIGHTS.get(metric, 0) * 100:.0f}%)")
                        anno_vals[metric] = st.text_area(
                            label,
                            key=f"anno_{chosen_id}_{metric}",
                            placeholder="이 점수가 실제와 맞는지, 개선점 등을 입력하세요…",
                            height=80,
                            label_visibility="collapsed",
                        )

                    overall_comment = st.text_area(
                        "종합 의견",
                        key=f"anno_{chosen_id}_overall",
                        placeholder="전반적인 품질 평가, 최종 선택 이유 등을 입력하세요…",
                        height=100,
                    )

                    # ── 임베딩 저장 버튼 ──────────────────────────
                    st.markdown("---")
                    st.caption("선택한 이미지를 CCTV 분석 쿼리 임베딩으로 확정합니다.")
                    chosen_emb = st.session_state.get(f"cmp_chosen_{chosen_id}")
                    emb_cols = st.columns(len(providers_ok))
                    for i, pname in enumerate(providers_ok):
                        is_chosen = chosen_emb == pname
                        btn_txt = ("✅ " if is_chosen else "") + f"{pname.upper()} 이미지로 임베딩 저장"
                        if emb_cols[i].button(btn_txt, key=f"save_emb_{chosen_id}_{pname}"):
                            update_outfit_embedding(chosen_id, results[pname]["path"])
                            st.session_state[f"cmp_chosen_{chosen_id}"] = pname
                            st.success(f"{pname.upper()} 이미지로 임베딩이 업데이트됐습니다.")
                            st.rerun()

                    # ── MD 저장 ───────────────────────────────────
                    if st.button("💾 평가 리포트 MD 저장", type="primary"):
                        md_txt = _build_md_report(
                            person_id=chosen_id,
                            person_name=chosen_name,
                            outfit_desc=cmp_state["outfit_desc"],
                            results=results,
                            providers_ok=providers_ok,
                            anno_vals=anno_vals,
                            overall_comment=overall_comment,
                            chosen_provider=st.session_state.get(f"cmp_chosen_{chosen_id}"),
                        )
                        _cfg.ensure_dirs()
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        md_path = _cfg.RESULTS_DIR / f"vton_eval_{chosen_id}_{ts}.md"
                        md_path.write_text(md_txt, encoding="utf-8")
                        st.success(f"저장됨: {md_path}")
                        st.download_button(
                            "⬇️  MD 파일 다운로드",
                            data=md_txt.encode("utf-8"),
                            file_name=md_path.name,
                            mime="text/markdown",
                        )


# ═══════════════════════════════════════════════════════════════════
# 화면 3: CCTV 구간 분석
# ═══════════════════════════════════════════════════════════════════
elif page == "📹 CCTV 분석 & 정확도 평가":
    st.header("📹 CCTV 영상 분석 & 정확도 평가")
    st.caption("영상을 분석해 유사 인물을 찾습니다. CVAT 정답 XML을 함께 넣으면 정확도 지표도 같이 나옵니다.")

    if not persons:
        st.warning("먼저 인물을 등록해주세요.")
    else:
        person_options = {f"{p['name']} ({p['person_id']})": p["person_id"] for p in persons}

        col_l, col_r = st.columns([1, 1])
        with col_l:
            chosen_label = st.selectbox("분석 대상", list(person_options.keys()))
            chosen_id    = person_options[chosen_label]
            # 평소 vs 생성 착장 두 상태를 구분 표시
            _st = embedding_states(chosen_id)
            _base_txt   = "✅ 평소(등록) 임베딩" if _st["base"] else "❌ 평소(등록) 임베딩"
            if _st["outfit"]:
                _tag = "3면" if _st["outfit_3view"] else "단면"
                _outfit_txt = f"✅ 생성 착장({_tag}) 임베딩"
            else:
                _outfit_txt = "❌ 생성 착장 임베딩 (미생성)"
            st.caption(_base_txt)
            st.caption(_outfit_txt)
            _used = "생성 착장(VTON)" if _st["outfit"] else ("평소 등록 사진" if _st["base"] else "없음")
            st.caption(f"🔎 **검색에 사용:** {_used}")

        with col_r:
            _demo_exists = _cfg.DEMO_VIDEO_PATH.exists()
            _src_options = []
            if _demo_exists:
                _src_options.append(f"🎬 데모 영상 ({_cfg.DEMO_VIDEO_LABEL})")
            _src_options.append("📁 직접 업로드")
            video_source = st.radio(
                "영상 소스", _src_options, index=0,
                help="데모 영상은 업로드 없이 바로 사용(평가 XML·색상 자동 연결). '직접 업로드'를 고르면 업로드 란이 나타납니다.",
            )
            use_demo_video = _demo_exists and video_source.startswith("🎬")
            videos = None
            if use_demo_video:
                st.caption(f"✅ 데모 영상 사용: {_cfg.DEMO_VIDEO_PATH.name} (정답 4~25초 등장)")
            else:
                videos = st.file_uploader(
                    "📁 CCTV 저장 영상 (mp4, avi 등, 복수 선택 가능)",
                    type=["mp4", "avi", "mov", "mkv"],
                    accept_multiple_files=True,
                )

        # ── 검색에 사용할 생성 피쳐(벡터) 선택 — 재생성 없이 무료 ──
        _presets = list_outfit_presets(chosen_id)
        selected_prompt_hint = ""
        if _presets:
            _opt_labels = ["▶ 현재 활성 착장 그대로 사용"] + [preset_label(m) for m in _presets]
            # 기본값: 데모 정답 복장과 일치하는 3면 피쳐 → 없으면 최신 피쳐
            _default_idx = 1  # 최신 피쳐
            for _i, _m in enumerate(_presets):
                if _m.get("has_3view") and _m.get("prompt", "").strip() == _cfg.DEMO_VIDEO_OUTFIT:
                    _default_idx = _i + 1
                    break
            _pick = st.selectbox(
                "🧩 검색에 사용할 생성 피쳐 (언제·어떤 프롬프트로 만든 벡터)",
                _opt_labels,
                index=_default_idx,
                help="생성할 때마다 날짜·프롬프트가 붙어 자동 저장됩니다. 골라서 쓰면 Gemini 재호출 없이 그 벡터로 검색합니다.",
            )
            if _pick != "▶ 현재 활성 착장 그대로 사용":
                _meta = _presets[_opt_labels.index(_pick) - 1]
                selected_prompt_hint = _meta.get("prompt", "")
                # 선택 피쳐 자동 적용 (마지막 적용과 다를 때만 → 중복 NPZ 쓰기 방지)
                _sk = f"loaded_preset_{chosen_id}"
                if st.session_state.get(_sk) != _meta["id"]:
                    try:
                        load_outfit_preset(chosen_id, _meta["id"])
                        st.session_state[_sk] = _meta["id"]
                    except Exception as ex:
                        st.warning(f"피쳐 적용 실패: {ex}")
                colp1, colp2 = st.columns([3, 1])
                colp1.caption(f"✅ 적용됨: **{_pick}**")
                if colp2.button("🗑️ 삭제", key="del_feature"):
                    delete_outfit_preset(chosen_id, _meta["id"])
                    st.session_state.pop(_sk, None)
                    st.rerun()
        else:
            st.caption("🧩 저장된 생성 피쳐 없음 — '착장 이미지 생성'에서 만들면 자동 저장됩니다.")

        # 데모 영상 사용 시 정답 복장을 색상 설명 기본값으로
        if not selected_prompt_hint and use_demo_video:
            selected_prompt_hint = _cfg.DEMO_VIDEO_OUTFIT

        outfit_desc_color = st.text_input(
            "🎨 착장 색상 설명 (강력 권장 — 원거리 CCTV에선 사실상 필수)",
            value=selected_prompt_hint,
            placeholder="예: 노란색 상의에 검은색 바지  — 상의/하의 색을 명시하세요",
            help=(
                "원거리·부감 CCTV에서는 Re-ID(외형 임베딩)만으로 정답과 타인을 구분하기 어렵습니다. "
                "상의/하의 색을 입력하면 색상 보정이 적용돼 정답 인물이 상위로 올라옵니다. "
                "선택한 생성 피쳐의 프롬프트가 자동 입력됩니다. 비워두면 탐지가 부정확할 수 있습니다."
            ),
        )
        if not outfit_desc_color.strip():
            st.caption("⚠️ 색상 설명이 비어 있습니다 — 색상 보정 없이 Re-ID만으로는 정답/타인 구분이 어렵습니다.")

        # ── 정확도 평가 (선택): CVAT XML 넣으면 분석 결과로 지표 계산 ──
        with st.expander("📊 정확도 평가 (선택) — CVAT 정답 XML 업로드"):
            st.caption(
                "CVAT로 라벨링한 정답 어노테이션(XML)을 넣으면, 아래 분석 결과를 정답과 비교해 "
                "재현율·오탐·IoU 등 지표를 함께 보여줍니다. (FPS는 영상에서 자동 인식)"
            )
            eval_xml = st.file_uploader("CVAT XML 어노테이션", type=["xml"], key="cctv_eval_xml")
            ec1, ec2, ec3 = st.columns(3)
            cvat_label = ec1.text_input("CVAT 라벨명", value="target",
                                        help="XML 안에서 정답 인물에 붙인 라벨 이름")
            time_tol = ec2.slider("시간 허용 오차 (초)", 0.5, 10.0, 2.0, 0.5)
            iou_thr = ec3.slider("BBox IoU 임계값", 0.1, 0.9, 0.3, 0.05)

        # 트래커 선택
        _TRACKER_OPTIONS = {
            "ByteTrack (기본, 고속)":       "bytetrack.yaml",
            "BoT-SORT (실험적, 재등장 복구 우수)": "botsort.yaml",
        }
        tracker_label = st.radio(
            "트래커",
            list(_TRACKER_OPTIONS.keys()),
            horizontal=True,
            help=(
                "ByteTrack: 칼만 필터 기반, 빠르고 안정적. "
                "BoT-SORT: Re-ID 임베딩을 매칭에 활용해 가림막 후 재등장 추적 우수 (속도 약간 느림)."
            ),
        )
        selected_tracker = _TRACKER_OPTIONS[tracker_label]

        # Gait 보조
        gait_col1, gait_col2 = st.columns([1, 2])
        use_gait = gait_col1.toggle(
            "🚶 Gait 보조 활성화",
            value=False,
            help="등록된 보행 특징을 Re-ID와 결합하여 정확도를 향상시킵니다.",
        )
        gait_weight = 0.3
        if use_gait:
            if not has_gait_embedding(chosen_id):
                gait_col2.warning("보행 특징 미등록 — 페이지 1에서 보행 동영상을 먼저 등록하세요.")
                use_gait = False
            else:
                gait_weight = gait_col2.slider("Gait 기여 비율", 0.1, 0.5, 0.3, 0.05,
                                                help="0.1=Re-ID 90%+Gait 10% … 0.5=50:50")

        # 종료시각 기본값: 데모=영상 전체 길이(자동), 업로드=00:30:00.
        # iter_frames는 영상 끝에서 멈추므로 길게 잡아도 안전.
        if use_demo_video:
            _demo_fps, _demo_dur = _video_meta(str(_cfg.DEMO_VIDEO_PATH))
            _end_default = seconds_to_hms(int(_demo_dur)) if _demo_dur > 0 else "00:20:00"
        else:
            _end_default = "00:30:00"

        c1, c2, c3, c4 = st.columns(4)
        start_time  = c1.text_input("시작 시각", value="00:00:00", help="HH:MM:SS")
        end_time    = c2.text_input(
            "종료 시각", value=_end_default,
            help="HH:MM:SS · 기본=영상 전체. 빠른 확인은 00:00:30 등으로 줄이세요.",
        )
        threshold   = c3.slider(
            "유사도 임계값", 0.10, 0.99, 0.80, 0.01,
            help=(
                "색상 보정을 켜면 점수가 높아집니다(정답 ≈0.90, 타인 ≈0.85). "
                "기본 0.80은 정답을 놓치지 않으면서 후보를 내림차순으로 보여줍니다. "
                "오탐이 많으면 0.85~0.90으로, 정답이 안 잡히면 낮추세요."
            ),
        )
        sample_every = c4.slider("샘플 간격 (프레임)", 1, 30, 5, 1,
                                  help="N프레임마다 1장만 분석. 5=정확도·속도 균형(19분 ~4.5분). 짧게 등장하는 인물이 있으면 낮게.")

        # 예상 소요시간 안내 (약 25프레임/초 처리, fps는 데모면 실측·아니면 25 가정)
        try:
            _eta_fps = _demo_fps if (use_demo_video and _demo_fps > 0) else 25.0
            _s0 = hms_to_seconds(start_time) if start_time else 0
            _s1 = hms_to_seconds(end_time) if end_time else 0
            if use_demo_video and _demo_dur > 0:
                _s1 = min(_s1, int(_demo_dur))
            _span_sec = max(0, _s1 - _s0)
            if _span_sec > 0:
                _n_proc = int(_span_sec * _eta_fps / max(1, sample_every))
                _eta_min = (_n_proc / 25.0) / 60.0
                st.caption(
                    f"⏱️ 분석 범위 {_span_sec//60}분 {_span_sec%60}초 · sample_every={sample_every} "
                    f"→ 약 {_n_proc:,}프레임, 예상 ~{_eta_min:.1f}분"
                )
        except Exception:
            pass

        # ── 준비 상태 종합 (탐지 전 한눈에 확인) ──
        _eng = _engine_status()
        _osnet_ok = _eng["osnet"][0]
        _yolo_ok = _eng["yolo"][0]
        _vec_lvl, _vec_desc = _search_vector_status(chosen_id)
        _ready = _osnet_ok and _yolo_ok and _vec_lvl != "none"
        _checks = [
            ("OSNet Re-ID 엔진", _osnet_ok, _eng["osnet"][1]),
            ("YOLO 사람 탐지", _yolo_ok, _eng["yolo"][1]),
            ("검색 벡터", _vec_lvl != "none", _vec_desc),
        ]
        if _ready:
            _src_msg = "데모 영상으로 바로 시작하세요." if use_demo_video else "영상 업로드 후 시작하세요."
            st.success(f"✅ 분석 준비 완료 — {_src_msg}")
        else:
            lines = "\n".join(
                f"- {'✅' if ok else '❌'} {name}: {msg}" for name, ok, msg in _checks
            )
            st.warning(
                "⚠️ 아래 항목이 준비되어야 탐지가 됩니다:\n" + lines +
                ("\n\n→ 검색 벡터가 없으면 **착장 이미지 생성**(페이지 2)을 먼저 실행하세요."
                 if _vec_lvl == "none" else "")
            )

        run_btn = st.button("🚀 구간 분석 시작", type="primary",
                            disabled=not (videos or use_demo_video))

        if run_btn and (videos or use_demo_video):
            tmp_dir = Path(tempfile.mkdtemp())
            video_paths: list[str] = []
            if use_demo_video:
                # 번들 영상 직접 사용 (업로드/임시복사 없음 → 경로 문제·편차 제거)
                video_paths.append(str(_cfg.DEMO_VIDEO_PATH))
            else:
                for vf in videos:
                    vp = tmp_dir / vf.name
                    vp.write_bytes(vf.read())
                    video_paths.append(str(vp))

            # 평가용 XML: 업로드 우선, 없으면 데모 영상이면 번들 XML 자동 사용
            eval_xml_path = None
            if eval_xml is not None:
                eval_xml_path = tmp_dir / eval_xml.name
                eval_xml_path.write_bytes(eval_xml.read())
            elif use_demo_video and _cfg.DEMO_VIDEO_XML.exists():
                eval_xml_path = _cfg.DEMO_VIDEO_XML
                if not cvat_label or cvat_label == "target":
                    cvat_label = _cfg.DEMO_VIDEO_XML_LABEL

            progress_bar = st.progress(0.0, text="분석 준비 중…")
            status_box   = st.empty()
            _last_frac = [0.0]

            # analyze_segment는 워커 스레드에서 실행됨. Streamlit 위젯을 워커 스레드에서
            # 호출하면 NoSessionContext 예외로 분석이 통째로 죽으므로,
            # ① 메인 스크립트 컨텍스트를 워커 스레드에 붙여 진행률 갱신을 허용하고
            # ② 그래도 실패하면 조용히 무시(분석은 계속).
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
                _ctx = get_script_run_ctx()
            except Exception:
                add_script_run_ctx = None
                _ctx = None
            _ctx_attached: set = set()

            def _progress(msg: str, frac: float = -1.0):
                # frac: 0~1 시간 기반 진척률(-1이면 모름 → 직전 값 유지)
                import threading
                if frac is not None and frac >= 0:
                    _last_frac[0] = min(1.0, max(0.0, frac))
                try:
                    th = threading.current_thread()
                    if add_script_run_ctx is not None and _ctx is not None and id(th) not in _ctx_attached:
                        add_script_run_ctx(th, _ctx)
                        _ctx_attached.add(id(th))
                    pct = int(_last_frac[0] * 100)
                    progress_bar.progress(_last_frac[0], text=f"{pct}% · {msg}")
                    status_box.caption(msg)
                except Exception:
                    pass  # 워커 스레드 컨텍스트 문제 등 — 분석은 계속 진행

            try:
                request = SearchRequest(
                    person_id=chosen_id,
                    video_paths=video_paths,
                    start_time=start_time,
                    end_time=end_time,
                    threshold=threshold,
                    sample_every=int(sample_every),
                    outfit_desc=outfit_desc_color.strip(),
                    tracker=selected_tracker,
                    use_gait=use_gait,
                    gait_weight=gait_weight,
                )
                color_active = bool(outfit_desc_color.strip())
                tracker_short = tracker_label.split(" (")[0]
                spinner_suffix = " + 색상 보정" if color_active else ""
                with st.spinner(f"분석 중… YOLO-Pose [{tracker_short}] + Part-based Re-ID{spinner_suffix}"):
                    report = analyze_multiple_cameras(request, _progress)

                progress_bar.progress(1.0, text="분석 완료")
                status_box.empty()

                # 분석 결과를 session_state에 저장 — 클립 버튼 등으로 재렌더링돼도 유지
                st.session_state["analysis_report"] = report
                st.session_state["analysis_video_paths"] = video_paths
                st.session_state["analysis_threshold"] = threshold
                st.session_state["analysis_eval_xml_path"] = eval_xml_path
                st.session_state["analysis_cvat_label"] = cvat_label
                st.session_state["analysis_time_tol"] = time_tol
                st.session_state["analysis_iou_thr"] = iou_thr

            except Exception as e:
                st.error(f"분석 실패: {e}")
                progress_bar.empty()

        # ── 분석 결과 표시 (session_state에서 불러오기 — 클립 버튼 재렌더링에 안정적) ──
        _saved = st.session_state.get("analysis_report")
        if _saved is not None and st.session_state.get("analysis_video_paths"):
            report = _saved
            video_paths = st.session_state["analysis_video_paths"]
            threshold = st.session_state.get("analysis_threshold", threshold)
            eval_xml_path = st.session_state.get("analysis_eval_xml_path")
            cvat_label = st.session_state.get("analysis_cvat_label", "target")
            time_tol = st.session_state.get("analysis_time_tol", 2.0)
            iou_thr = st.session_state.get("analysis_iou_thr", 0.3)

            st.markdown("---")
            st.subheader("📊 분석 결과")
            st.info(report.summary)

            # 🔍 단계별 진단 로그 — 왜 이렇게 나왔는지 항상 확인 가능
            _diag = getattr(report, "diagnostics", []) or []
            _has_err = any("❌" in d for d in _diag)
            with st.expander("🔍 단계별 진단 로그 (어디서·왜 그렇게 나왔는지)",
                             expanded=(_has_err or not report.hits)):
                if _diag:
                    st.code("\n".join(_diag), language="text")
                else:
                    st.caption("진단 로그가 없습니다.")

            # camera_id → 영상 경로 매핑 (후보·탐지 공용)
            cam_to_path = {Path(p).stem: p for p in video_paths}

            # 🏅 유사도 상위 후보 — 임계값을 못 넘어도 항상 표시
            _cands = getattr(report, "candidates", []) or []
            if _cands:
                st.subheader(f"🏅 유사도 상위 후보 — {len(_cands)}명 (임계값 무관)")
                st.caption(f"임계값({threshold}) 통과 여부와 상관없이 가장 비슷한 순. ✅=임계값 통과 ▽=미통과")
                # 4열 × N행 그리드 (8명 → 2행 × 4열)
                _COLS = 4
                for _row_start in range(0, len(_cands), _COLS):
                    _row_cands = _cands[_row_start:_row_start + _COLS]
                    _cols = st.columns(_COLS)
                    for _ci, _cand in enumerate(_row_cands):
                        with _cols[_ci]:
                            _passed = _cand.similarity >= threshold
                            _rank = _row_start + _ci + 1
                            st.markdown(f"**#{_rank} · {_cand.similarity*100:.1f}%** {'✅' if _passed else '▽'}")
                            if Path(_cand.crop_image_path).exists():
                                st.image(_cand.crop_image_path, use_container_width=True)
                            _rec_short = (f" ({_cand.recording_datetime[-8:]})" if _cand.recording_datetime else "")
                            st.caption(f"{_cand.timestamp}{_rec_short} · Track {_cand.track_id}")
                st.markdown("---")

            # ── 🚶 2차 검색 (외형 독립 재검증 — 걸음걸이 + 체형) ──
            _tfeats = getattr(report, "track_features", []) or []
            try:
                _gait_ok = has_gait_embedding(chosen_id)
            except Exception:
                _gait_ok = False
            with st.expander("🚶 2차 검색 (외형 독립 재검증 — 걸음걸이 + 체형)", expanded=False):
                st.caption(
                    "색상·옷과 **무관하게** 걸음걸이(Gait)와 보이는 체형으로 1차 결과를 교차검증합니다. "
                    "색이 안 맞아도 걸음이 일치하면 후보로 올립니다. (영상 재처리 없음)"
                )
                st.caption(
                    "걸음걸이 가중치는 *깨끗한 걸음 데이터가 얼마나 잡혔는지*(유효 골격 프레임 수)로 정합니다. "
                    "프레임이 15장 미만이면 걸음걸이는 무시하고 부위 점수만 씁니다."
                )
                if not _gait_ok:
                    st.warning("이 대상자는 보행 특징(Gait)이 등록되지 않았습니다. "
                               "페이지 1에서 보행 동영상을 먼저 등록하세요.")
                elif not _tfeats:
                    st.info("재점수할 트랙 특징이 없습니다.")
                else:
                    if st.button("🔁 2차 검색 실행 (외형 독립 재검증)", key="btn_secondary"):
                        from missing_person_mvp.pipeline.analyze_vod import secondary_search_rescore
                        from missing_person_mvp.pipeline.register import (
                            get_part_bank as _gpb, get_part_bank_multi_view as _gpbmv,
                            get_gait_embedding as _gge,
                        )
                        _res = secondary_search_rescore(
                            _tfeats, _gpb(chosen_id), _gpbmv(chosen_id), _gge(chosen_id),
                        )
                        st.session_state["secondary_result"] = _res

                    _sec = st.session_state.get("secondary_result")
                    if _sec:
                        # 1차 순위 맵 (교차검증으로 순위 변동 가시화)
                        _prim_rank = {
                            (tf.camera_id, tf.track_id): i + 1
                            for i, tf in enumerate(
                                sorted(_tfeats, key=lambda t: t.primary_score, reverse=True)
                            )
                        }
                        st.markdown("**2차 재검증 결과** (걸음걸이+체형 내림차순)")
                        import pandas as _pd
                        _rows = []
                        for _si, _r in enumerate(_sec[:8]):
                            _tf = _r["feature"]
                            _p1 = _prim_rank.get((_tf.camera_id, _tf.track_id), "-")
                            _delta = ""
                            if isinstance(_p1, int):
                                _d = _p1 - (_si + 1)
                                _delta = f"▲{_d}" if _d > 0 else (f"▼{-_d}" if _d < 0 else "—")
                            _n = _r["gait_frames_n"]
                            _gait_cell = f"{_r['gait_score']*100:.0f}%" if _n >= 15 else "데이터부족"
                            _rows.append({
                                "2차순위": _si + 1,
                                "2차점수": f"{_r['secondary_score']*100:.1f}%",
                                "부위": f"{_r['part_score']*100:.0f}%",
                                "걸음걸이": _gait_cell,
                                "걸음프레임": _n,
                                "가림(참고)": f"{_tf.occlusion_ratio*100:.0f}%",
                                "1차순위": _p1,
                                "변동": _delta,
                                "시각": _tf.timestamp,
                                "Track": _tf.track_id,
                            })
                        st.dataframe(_pd.DataFrame(_rows), hide_index=True, use_container_width=True)
                        st.caption(
                            "부위=보이는 부위 Re-ID · 걸음걸이=Gait 유사도(외형 독립) · "
                            "걸음프레임=잡힌 유효 골격 수(15+ 라야 걸음걸이 반영) · "
                            "가림(참고)=가려진 정도 · 변동=1차→2차 순위 상승(▲)/하락(▼)"
                        )
                        # 상위 3명 크롭 미리보기
                        _pcols = st.columns(min(3, len(_sec)))
                        for _pi in range(min(3, len(_sec))):
                            _r = _sec[_pi]
                            _tf = _r["feature"]
                            with _pcols[_pi]:
                                st.markdown(f"**#{_pi+1} · {_r['secondary_score']*100:.1f}%**")
                                if Path(_tf.crop_image_path).exists():
                                    st.image(_tf.crop_image_path, use_container_width=True)
                                _gtag = f"걸음 {_r['gait_score']*100:.0f}%" if _r["gait_frames_n"] >= 15 else "걸음 데이터부족"
                                st.caption(f"{_tf.timestamp} · {_gtag}")
                st.markdown("---")

            if report.hits:
                # 유사도 내림차순 정렬
                hits_desc = sorted(report.hits, key=lambda h: h.similarity, reverse=True)

                st.subheader(f"🎯 탐지 결과 — {len(hits_desc)}건 (유사도 높은 순)")

                for i, hit in enumerate(hits_desc):
                    sim_pct = hit.similarity * 100
                    expander_label = (
                        f"#{i + 1}  {sim_pct:.1f}%  ·  {hit.camera_id}"
                        f"  ·  {hit.timestamp}  ·  Track {hit.track_id}"
                    )
                    with st.expander(expander_label, expanded=(i == 0)):
                        col_crop, col_frame, col_meta = st.columns([1, 2, 1])

                        with col_crop:
                            st.caption("감지 크롭")
                            if Path(hit.crop_image_path).exists():
                                st.image(hit.crop_image_path, use_container_width=True)
                            else:
                                st.caption("크롭 이미지 없음")

                        with col_frame:
                            st.caption("원본 프레임 + BBox")
                            vpath = cam_to_path.get(hit.camera_id)
                            if vpath and Path(vpath).exists():
                                try:
                                    frame_rgb = extract_frame_with_bbox(
                                        vpath, hit.timestamp,
                                        hit.bounding_box, hit.similarity,
                                    )
                                    st.image(frame_rgb, use_container_width=True)
                                except Exception as fe:
                                    st.caption(f"프레임 추출 실패: {fe}")
                                # 트래킹 구간 클립 버튼
                                _ck = f"clip_{hit.camera_id}_{hit.track_id}"
                                _has_end = bool(hit.tracking_end)
                                if st.button("▶ 구간 영상 보기", key=f"btn_{_ck}",
                                             disabled=not _has_end,
                                             help="클릭 시 트래킹 전체 구간을 동영상으로 추출합니다 (~2초)"):
                                    st.session_state[_ck] = True
                                if st.session_state.get(_ck) and _has_end:
                                    _cp = _cfg.RESULTS_DIR / f"{_ck}.mp4"
                                    if not _cp.exists():
                                        with st.spinner("클립 추출 중..."):
                                            try:
                                                extract_video_clip(
                                                    vpath,
                                                    hms_to_seconds(hit.timestamp),
                                                    hms_to_seconds(hit.tracking_end),
                                                    str(_cp),
                                                )
                                            except Exception as ce:
                                                st.error(f"클립 추출 실패: {ce}")
                                    if _cp.exists():
                                        st.video(str(_cp))
                                        st.caption(f"트래킹: {hit.timestamp} ~ {hit.tracking_end}")
                            else:
                                st.caption("원본 영상을 찾을 수 없습니다.")

                        with col_meta:
                            st.metric("유사도", f"{sim_pct:.1f}%")
                            st.caption(f"**카메라**: {hit.camera_id}")
                            st.caption(f"**영상 위치**: {hit.timestamp}")
                            if hit.recording_datetime:
                                st.caption(f"**실제 시각**: {hit.recording_datetime}")
                            st.caption(f"**Track ID**: {hit.track_id}")
                            x1, y1, x2, y2 = hit.bounding_box
                            st.caption(f"**BBox**: ({x1},{y1})→({x2},{y2})")
                            st.caption(f"**크기**: {x2 - x1}×{y2 - y1}px")

                # 타임라인 테이블 (시간순)
                st.markdown("---")
                st.subheader("🕐 타임라인 (시간순)")
                timeline = build_timeline(report)
                if timeline:
                    import pandas as pd
                    # 실제 시각 컬럼: hits에서 직접 뽑음
                    _hits_sorted = sorted(report.hits, key=lambda h: h.timestamp)
                    _rec_map = {(h.camera_id, h.timestamp): h.recording_datetime for h in _hits_sorted}
                    df = pd.DataFrame(timeline)[["timestamp", "camera_id", "track_id", "similarity"]]
                    df["recording_datetime"] = df.apply(
                        lambda r: _rec_map.get((r["camera_id"], r["timestamp"])), axis=1
                    )
                    _has_rec = df["recording_datetime"].notna().any()
                    if _has_rec:
                        df = df[["timestamp", "recording_datetime", "camera_id", "track_id", "similarity"]]
                        df.columns = ["영상 위치", "실제 시각", "카메라", "Track ID", "유사도"]
                    else:
                        df = df[["timestamp", "camera_id", "track_id", "similarity"]]
                        df.columns = ["영상 위치", "카메라", "Track ID", "유사도"]
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # PDF + 결과 저장
                dl_col, save_col = st.columns(2)
                pdf_path = export_pdf(report)
                with open(pdf_path, "rb") as f:
                    dl_col.download_button(
                        "⬇️  PDF 리포트 다운로드",
                        data=f.read(),
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                    )
                if save_col.button("💾 탐지 결과 저장 (JSON)", key="save_detection_result"):
                    import json as _json
                    save_payload = {
                        "saved_at": datetime.now().isoformat(),
                        "conditions": {
                            "person_id": chosen_id,
                            "threshold": threshold,
                            "tracker": selected_tracker,
                            "outfit_desc": outfit_desc_color.strip(),
                            "use_gait": use_gait,
                            "gait_weight": gait_weight if use_gait else None,
                        },
                        "summary": report.summary,
                        "hit_count": len(report.hits),
                        "hits": [
                            {
                                "timestamp": h.timestamp,
                                "similarity": h.similarity,
                                "track_id": h.track_id,
                                "camera_id": h.camera_id,
                                "bounding_box": list(h.bounding_box),
                            }
                            for h in report.hits
                        ],
                    }
                    _cfg.ensure_dirs()
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_fp = _cfg.RESULTS_DIR / f"detection_{chosen_id}_{ts_str}.json"
                    save_fp.write_text(
                        _json.dumps(save_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    st.success(f"저장됨: {save_fp.name}")
                    st.download_button(
                        "⬇️  JSON 다운로드",
                        data=save_fp.read_bytes(),
                        file_name=save_fp.name,
                        mime="application/json",
                        key="dl_detection_json",
                    )
            else:
                st.warning("해당 구간에서 유사 인물이 탐지되지 않았습니다. 색상 설명을 입력하거나 임계값을 낮춰보세요.")

            # ── 정확도 평가 (CVAT XML이 있으면 위 분석 결과로 지표 계산) ──
            if eval_xml_path is not None:
                st.markdown("---")
                st.subheader("📈 정확도 평가 (CVAT 정답 대비)")
                try:
                    import cv2 as _cv2
                    from missing_person_mvp.scripts.evaluate_detection import (
                        evaluate_detection as _eval_det,
                        hits_from_report as _hits_from_report,
                        load_cvat_xml as _load_cvat_xml,
                        write_outputs as _write_outputs,
                    )
                    # FPS·길이 자동 인식 (첫 영상 기준)
                    _cap = _cv2.VideoCapture(video_paths[0])
                    _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
                    _tot = _cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                    _cap.release()
                    _dur = _tot / _fps if _fps > 0 else None

                    gt_boxes = _load_cvat_xml(eval_xml_path, fps=_fps, label=cvat_label)
                    eval_hits = _hits_from_report(report)
                    eval_res = _eval_det(
                        gt_boxes, eval_hits,
                        time_tolerance_sec=time_tol, iou_threshold=iou_thr,
                        video_duration_sec=_dur,
                    )
                    st.caption(f"영상 FPS {_fps:.1f} 자동 인식 · GT 박스 {len(gt_boxes)}개 · 라벨 '{cvat_label}'")
                    _show_single_eval(eval_res)

                    import pandas as pd
                    for _lbl, _rows in [
                        ("✅ 매칭된 탐지", eval_res["matched_hits"]),
                        ("⚠️ 오탐 목록",   eval_res["false_hits"]),
                        ("❌ 미탐 GT 트랙", eval_res["missed_tracks"]),
                    ]:
                        if _rows:
                            with st.expander(f"{_lbl} ({len(_rows)}건)"):
                                st.dataframe(pd.DataFrame(_rows), hide_index=True,
                                             use_container_width=True)

                    _out_d = _cfg.RESULTS_DIR / f"eval_{chosen_id}"
                    _write_outputs(eval_res, _out_d)
                    _ec = st.columns(2)
                    _ec[0].download_button("⬇️ metrics.json",
                                           data=(_out_d / "metrics.json").read_bytes(),
                                           file_name="metrics.json", mime="application/json")
                    _ec[1].download_button("⬇️ summary.md",
                                           data=(_out_d / "summary.md").read_bytes(),
                                           file_name="summary.md", mime="text/markdown")
                except Exception as _ee:
                    st.error(f"평가 실패: {_ee}")


# ═══════════════════════════════════════════════════════════════════
# 화면 4: Who Am I
# ═══════════════════════════════════════════════════════════════════
if page == "🧑 Who Am I":
    st.header("🧑 Who Am I")
    _resume_path = _Path(__file__).resolve().parent / "assets" / "resume.pdf"
    if _resume_path.exists():
        _pdf_b64 = base64.b64encode(_resume_path.read_bytes()).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{_pdf_b64}" '
            f'width="100%" height="900px" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.info("📄 이력서 파일을 `missing_person_mvp/assets/resume.pdf` 에 넣어주세요.")

