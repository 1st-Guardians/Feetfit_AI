# Feetfit AI API

FastAPI server for analyzing left/right foot images and forwarding the generated tinea
pedis report to the report backend.

## Structure

```text
app/
  main.py                 FastAPI entry point
  api/
    router.py             Root API router
    routes/
      reports.py          Tinea pedis report upload/proxy endpoint
  core/
    config.py             Environment and app settings
    security.py           Swagger Bearer token security scheme
    weights.py            Centralized model weight paths
  schemas/
    reports.py            Pydantic request schema
  services/
    tinea_analysis.py     Foot/tinea segmentation, scoring, suspicion map render
weights/                  Model weights, ignored by git
```

This is a common FastAPI layout: route definitions stay in `api`, shared
configuration and security live in `core`, request/response contracts live in
`schemas`, and business helpers live in `services`.

## Run

```powershell
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger:

```text
http://localhost:8000/docs
```

Click `Authorize` in Swagger and paste the JWT Bearer token. Then call:

```text
POST /api/reports/tina-pedis
```

Input fields:

```text
measurementSessionId  default: 1
leftFootImage         one original left foot image
rightFootImage        one original right foot image
```

The server analyzes both images, creates:

```text
request                 generated scores and temporary descriptions
suspiciousAreaMapImage  left + right circular suspicion maps combined side by side as PNG
originalFootImage       left + right cutout photo overlays combined side by side as PNG
```

Each side is cropped to the detected foot region with transparent PNG background,
then the left and right images are combined side by side for the tinea backend.

The safety scores use the more conservative value between the two feet.
When `TINEA_SLIDING_WINDOW_ENABLED=true`, the tinea model also runs on overlapping
tiles inside the detected foot bbox and merges tile probabilities with the
full-image probability map to improve small-region detection.
The tinea contrast/red enhancement options apply only to the tinea model input;
hallux valgus landmark inference keeps the original image preprocessing.

Then it forwards the multipart request to:

```text
http://35.94.253.151/api/reports/tina-pedis
```

The tinea description fields and hallux `scoreAnalysisText` are generated with
the OpenAI Responses API when `OPENAI_API_KEY` is configured. If the key is not
set or the API call fails, the server falls back to local rule-based Korean
summary text and still forwards the report request.

Model weights must be placed here:

```text
D:/Feetfit_AI/weights/foot_seg_yolo11n_best.pt
D:/Feetfit_AI/weights/tinea_pedis_best.pt
D:/Feetfit_AI/weights/sam_vit_b_01ec64.pth
```

Override it with `.env` if needed:

```text
TINEA_REPORT_ENDPOINT=http://35.94.253.151/api/reports/tina-pedis
HALLUX_VALGUS_REPORT_ENDPOINT=http://35.94.253.151/api/reports/hallux-valgus
REPORT_PROXY_TIMEOUT_SECONDS=60
OPENAI_API_KEY=paste-your-openai-api-key-here
OPENAI_REPORT_MODEL=gpt-4.1-mini
OPENAI_REPORT_TIMEOUT_SECONDS=20
OPENAI_REPORT_TEXT_ENABLED=true
OPENAI_REPORT_INCLUDE_IMAGES=true
COMBINED_IMAGE_MAX_HEIGHT=1600
COMBINED_IMAGE_GAP_PIXELS=16
FUNGAL_THRESHOLD=0.78
INFLAMMATION_THRESHOLD=0.77
TINEA_SLIDING_WINDOW_ENABLED=true
TINEA_SLIDING_WINDOW_TILE_SIZE=768
TINEA_SLIDING_WINDOW_OVERLAP=0.3
TINEA_SLIDING_WINDOW_PADDING=32
TINEA_SLIDING_WINDOW_MAX_TILES=12
TINEA_PREPROCESS_ENHANCE_ENABLED=true
TINEA_PREPROCESS_CONTRAST_GAIN=1.05
TINEA_PREPROCESS_CLAHE_CLIP_LIMIT=1.6
TINEA_PREPROCESS_RED_SATURATION_GAIN=1.10
TINEA_PREPROCESS_RED_VALUE_GAIN=1.02
MAX_DOT_AREA_FOR_SUSPICION_MAP=450
PHOTO_CUTOUT_BACKGROUND=true
PHOTO_CUTOUT_PADDING=28
```

## Curl Example

```bash
curl -X POST "http://localhost:8000/api/reports/tina-pedis" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "leftFootImage=@left-foot.png;type=image/png" \
  -F "rightFootImage=@right-foot.png;type=image/png"
```

## Single-photo integrated analysis

`POST /api/reports/integrated-foot-analysis` accepts one calibrated two-foot
photo and runs the complete pipeline:

```text
one 1280x720 photo
  -> lens undistortion
  -> ArUco board orientation and planar calibration
  -> one YOLO foot-boundary inference
  -> mask rectification and left/right split
  -> foot length and straight-line MTP-zone ball width
  -> tinea analysis with the same per-foot masks
  -> hallux-valgus analysis with the same per-foot masks
  -> existing tinea and hallux report backends
```

ArUco detects the board geometry before converting the YOLO masks to the
metric plane. The YOLO network itself runs only once on the lens-corrected
source frame. Both disease analyzers consume those shared masks and do not run
their own foot-boundary YOLO inference on this endpoint.

The validated ArUco implementation and camera calibration are loaded from the
separately cloned repository configured by `ARUCO_SOURCE_DIR` (default:
`D:/ArUco-marker-code`). Keep that clone at the matching validated revision and
make sure its `models/hybrid_best_1280x720.npz` file exists.

```bash
curl -X POST "http://localhost:8000/api/reports/integrated-foot-analysis" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "footImage=@two-feet-with-aruco.jpg;type=image/jpeg"
```

The response contains `analysis.feet.left/right` with `lengthMm`,
`ballWidthMm`, tinea safety scores, and hallux angle, plus the response from
each existing report backend. HTTP `200` means both reports were accepted;
`207` means analysis completed but at least one backend returned an error.

Important capture constraints:

- The default lens model is valid only for an unmodified 1280x720 full frame.
- Use `DICT_4X4_50` marker IDs 0 through 5 and the physical dimensions in
  `.env.example`.
- `image_left` and `image_right` in the ArUco code are board bays, not anatomy.
  Confirm the fixed capture protocol and set
  `ARUCO_IMAGE_LEFT_ANATOMICAL_SIDE` to `left` or `right` accordingly. The
  bundled example protocol maps the canonical image-left bay to the left foot.
- `ballWidthMm` is a straight line across the silhouette's MTP proxy zone, not
  foot girth.
- Always check `measurementValid` and `measurementInvalidReasons`; disease
  analysis and metric-measurement quality are reported separately.
