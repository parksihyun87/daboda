from __future__ import annotations

import os
from pathlib import Path

from missing_person_mvp import config
from missing_person_mvp.models import AnalysisReport, DetectionHit


_KOREAN_FONT_CANDIDATES = [
    "malgun.ttf",                               # Windows 맑은 고딕
    "C:/Windows/Fonts/malgun.ttf",
    "NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def _find_korean_font() -> str | None:
    for path in _KOREAN_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def build_timeline(report: AnalysisReport) -> list[dict]:
    return [
        {
            "timestamp": hit.timestamp,
            "camera_id": hit.camera_id,
            "similarity": round(hit.similarity, 4),
            "track_id": hit.track_id,
            "crop_image_path": hit.crop_image_path,
        }
        for hit in report.sorted_hits()
    ]


def infer_movement(
    hits: list[DetectionHit],
    camera_locations: dict[str, str] | None = None,
) -> str:
    if not hits:
        return "탐지 결과가 없어 이동 동선을 추정할 수 없습니다."
    ordered = sorted(hits, key=lambda hit: hit.timestamp)
    names: list[str] = []
    for hit in ordered:
        label = camera_locations.get(hit.camera_id, hit.camera_id) if camera_locations else hit.camera_id
        if not names or names[-1] != label:
            names.append(label)
    return " -> ".join(names)


def export_pdf(report: AnalysisReport, output_path: str | None = None) -> str:
    config.ensure_dirs()
    path = Path(output_path) if output_path else config.RESULTS_DIR / "analysis_report.pdf"
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        kor_font = "Helvetica"
        font_path = _find_korean_font()
        if font_path:
            try:
                pdfmetrics.registerFont(TTFont("Korean", font_path))
                kor_font = "Korean"
            except Exception:
                pass

        pdf = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 50

        pdf.setFont(kor_font, 16)
        pdf.drawString(50, y, "실종자 CCTV 분석 리포트")
        y -= 30
        pdf.setFont(kor_font, 10)
        pdf.drawString(50, y, f"대상 ID: {report.request.person_id}")
        y -= 16
        pdf.drawString(50, y, f"분석 구간: {report.request.start_time} ~ {report.request.end_time}")
        y -= 16
        pdf.drawString(50, y, f"임계값: {report.request.threshold}")
        y -= 20

        movement = infer_movement(report.hits)
        pdf.setFont(kor_font, 9)
        pdf.drawString(50, y, f"이동 동선: {movement}")
        y -= 16
        summary_lines = [report.summary[i:i+80] for i in range(0, len(report.summary), 80)]
        for line in summary_lines:
            pdf.drawString(50, y, line)
            y -= 14
        y -= 10

        pdf.setFont(kor_font, 9)
        pdf.setFillColorRGB(0.75, 0, 0)
        pdf.drawString(50, y, "시각        카메라          Track  유사도")
        pdf.setFillColorRGB(0, 0, 0)
        y -= 14

        for hit in report.sorted_hits():
            if y < 60:
                pdf.showPage()
                y = height - 50
                pdf.setFont(kor_font, 9)
            bar = "█" * int(hit.similarity * 10) + "░" * (10 - int(hit.similarity * 10))
            line = f"{hit.timestamp}  {hit.camera_id:<16} {hit.track_id:<6} {hit.similarity:.3f} {bar}"
            pdf.drawString(50, y, line)
            y -= 14

        pdf.save()
        return str(path)
    except Exception:
        txt_path = path.with_suffix(".txt")
        movement = infer_movement(report.hits)
        lines = [
            "=== 실종자 CCTV 분석 리포트 ===",
            f"대상 ID: {report.request.person_id}",
            f"분석 구간: {report.request.start_time} ~ {report.request.end_time}",
            f"임계값: {report.request.threshold}",
            f"이동 동선: {movement}",
            f"요약: {report.summary}",
            "",
            "시각        | 카메라           | Track | 유사도",
            "-" * 56,
        ]
        lines.extend(
            f"{hit.timestamp} | {hit.camera_id:<16} | {hit.track_id:<5} | {hit.similarity:.3f}"
            for hit in report.sorted_hits()
        )
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        return str(txt_path)

