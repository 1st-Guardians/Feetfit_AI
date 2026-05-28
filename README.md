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
suspiciousAreaMapImage  left + right circular suspicion maps combined side by side
originalFootImage       left + right red/blue photo overlays combined side by side
```

The suspicion map is generated only from the segmentation mask used in the
photo overlay.

The safety scores use the more conservative value between the two feet.

Then it forwards the multipart request to:

```text
http://54.184.58.176/api/reports/tina-pedis
```

For now, the three description fields are filled with longer temporary Korean
comments before forwarding. Later, this can be replaced with GPT-generated text.

Model weights must be placed here:

```text
D:/Feetfit_AI/weights/foot_seg_yolo11n_best.pt
D:/Feetfit_AI/weights/tinea_pedis_best.pt
D:/Feetfit_AI/weights/sam_vit_b_01ec64.pth
```

Override it with `.env` if needed:

```text
TINEA_REPORT_ENDPOINT=http://54.184.58.176/api/reports/tina-pedis
REPORT_PROXY_TIMEOUT_SECONDS=60
COMBINED_IMAGE_MAX_HEIGHT=1600
COMBINED_IMAGE_GAP_PIXELS=16
FUNGAL_THRESHOLD=0.78
MAX_DOT_AREA_FOR_SUSPICION_MAP=450
# INFLAMMATION_THRESHOLD=0.88
```

## Curl Example

```bash
curl -X POST "http://localhost:8000/api/reports/tina-pedis" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "leftFootImage=@left-foot.jpeg;type=image/jpeg" \
  -F "rightFootImage=@right-foot.jpeg;type=image/jpeg"
```
