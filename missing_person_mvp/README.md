# Missing Person CCTV VOD MVP

Local MVP for batch analysis of stored CCTV video segments. The app registers a
person, generates outfit reference images, analyzes uploaded VOD files, and
exports a timeline/report.

## Run

```powershell
cd c:\workplace\code\python\speed_yulam
python -m pip install -r missing_person_mvp\requirements.txt
copy missing_person_mvp\.env.example missing_person_mvp\.env
# .env 에 GEMINI_API_KEY 입력 후:
$env:KMP_DUPLICATE_LIB_OK="TRUE"
streamlit run missing_person_mvp/streamlit_app.py
```

Open `http://localhost:8501`.

## VIGI C330 Notes

This MVP analyzes saved files, not live RTSP streams. Use the VIGI camera/NVR
tool to export the relevant time window as MP4, then upload it in the VOD
Analysis tab.

## Flux Local Fallback

Image generation uses provider order from `.env`. Flux now runs as a local
inpainting provider and requires registered front/side/back full-body photos:

```env
IMAGE_PROVIDER_ORDER=gemini,flux,fashn,placeholder
FLUX_MODEL_ID=black-forest-labs/FLUX.1-schnell
FLUX_MODE=inpaint
FLUX_NUM_INFERENCE_STEPS=4
FLUX_INPAINT_STRENGTH=0.65
FLUX_REQUIRE_THREE_VIEWS=true
```

The Flux provider edits the clothing mask on each registered view, then writes
front/side/back outputs and a contact sheet. Face/head regions are protected by
YOLO-Pose/OpenCV-assisted masking with a ratio fallback. This improves identity
preservation compared with text-to-image generation, but it still requires
manual review before using the output as a search reference.

## Generation Evaluation

Compare Gemini and Flux outputs:

```powershell
python missing_person_mvp\scripts\evaluate_generation.py `
  --person-image path\to\person.jpg `
  --outfit-desc "파란 점퍼 검은 바지" `
  --gemini-image path\to\gemini.jpg `
  --flux-image path\to\flux_sheet.jpg `
  --person-id p001 `
  --output-dir missing_person_mvp\data\results\vton_eval
```

The script writes `scores.json`, `scores.csv`, and `comparison.jpg`.

## Verify

```powershell
python -m pytest missing_person_mvp\tests -q
python -m compileall -q missing_person_mvp
python missing_person_mvp\scripts\health_check.py
```
