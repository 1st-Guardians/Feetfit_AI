# Feetfit AI API

FastAPI server for analyzing left/right foot images (tinea pedis, hallux valgus) and
computing shoe fit recommendations, forwarding all results to the report backend
(Feetfit_Server).

## Structure

```text
app/
  main.py                       FastAPI entry point
  api/
    router.py                   Root API router
    routes/
      reports.py                Tinea pedis / hallux valgus / shoe-recommendations endpoints
      shoes.py                  Shoe detail summary generation trigger endpoint
  core/
    config.py                   Environment and app settings
    security.py                 Swagger Bearer token security scheme
    weights.py                  Centralized model weight paths
  prompts/
    shoe_fit_comment_prompts.py SYSTEM_PROMPT / USER_PROMPT_TEMPLATE for Ollama
  schemas/
    reports.py                  Tinea pedis / hallux valgus request schemas
    shoes.py                    Shoe recommendation batch request/forward schemas
    shoe_fit_comment.py         Ollama-generated summary response schema
  services/
    tinea_analysis.py           Foot/tinea segmentation, scoring, suspicion map render
    hallux_valgus_analysis.py   Foot outline extraction, hallux valgus angle (HVA) scoring
    shoe_db.py                  Direct MySQL access (shared with Feetfit_Server) for
                                 shoes/reviews/foot analysis/saved recommendations
    shoe_embedding.py           BGE-M3 sentence embedding + disk cache + cosine ranking
    shoe_feature_rules.py       Foot-need thresholds, review keyword/polarity rules
    shoe_recommendation.py      fitScore/riskLevel/근거 리뷰 계산 (배치, LLM 미사용)
    shoe_fit_comment_service.py Ollama call: pointSummary + 부위별 reviewSummary 생성
weights/                        Model weights, ignored by git
```

This is a common FastAPI layout: route definitions stay in `api`, shared
configuration and security live in `core`, request/response contracts live in
`schemas`, LLM prompt text lives in `prompts`, and business helpers live in
`services`.

## Run

```powershell
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger:

```text
http://localhost:8000/docs
```

Click `Authorize` in Swagger and paste the JWT Bearer token before calling any
endpoint below (all of them forward the same token to Feetfit_Server).

The legacy tinea endpoint is:

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

The shoe-recommendation endpoints additionally require:

- A MySQL connection to the **same database Feetfit_Server uses** (`SHOE_DB_URL` /
  `SHOE_DB_USERNAME` / `SHOE_DB_PASSWORD`) — shoes, reviews, foot analysis results,
  and saved shoe recommendations are all read/written there directly.
- [Ollama](https://ollama.com) running locally with a pulled model:

  ```powershell
  ollama pull qwen2.5:7b-instruct
  ```

  (model name is configurable via `OLLAMA_MODEL`; defaults to `qwen2.5:7b-instruct`)

## Endpoints

### `POST /api/reports/tina-pedis`

Upload one left + one right foot image. The server analyzes both, creates:

```text
request                 generated scores and temporary descriptions
suspiciousAreaMapImage  left + right circular suspicion maps combined side by side
originalFootImage       left + right red/blue photo overlays combined side by side
```

The suspicion map is generated only from the segmentation mask used in the photo
overlay. The safety scores use the more conservative value between the two feet.
Then it forwards the multipart request to `TINEA_REPORT_ENDPOINT`.

The description fields are generated through the OpenAI Responses API when it
is configured. If generation is unavailable, local Korean rule-based fallback
text is used so report forwarding can still complete.

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
  the `.env` reference below.
- `image_left` and `image_right` in the ArUco code are board bays, not anatomy.
  Confirm the fixed capture protocol and set
  `ARUCO_IMAGE_LEFT_ANATOMICAL_SIDE` to `left` or `right` accordingly. The
  bundled example protocol maps the canonical image-left bay to the left foot.
- `ballWidthMm` is a straight line across the silhouette's MTP proxy zone, not
  foot girth.
- Always check `measurementValid` and `measurementInvalidReasons`; disease
  analysis and metric-measurement quality are reported separately.

### `POST /api/reports/hallux-valgus`

Upload one left + one right foot image. The server extracts the foot outline,
computes the hallux valgus angle (HVA) per foot, and renders an analysis image
with 3 keypoints and connecting lines drawn on the outline. Forwards to
`HALLUX_VALGUS_REPORT_ENDPOINT` the same way as the tinea pedis endpoint.

```bash
curl -X POST "http://localhost:8000/api/reports/hallux-valgus" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "leftFootImage=@left-foot.jpeg;type=image/jpeg" \
  -F "rightFootImage=@right-foot.jpeg;type=image/jpeg"
```

### `POST /api/reports/shoe-recommendations`

```json
{ "measurementSessionId": 30 }
```

Resolves the user behind that measurement session, reads their latest 종합 발
분석(자세 균형, 좌우 압력 분포, 발볼/발길이 수치, 평균 습도)/무지외반/무좀 분석
결과, then recomputes **fitScore + riskLevel + 근거 리뷰(FOREFOOT/HEEL/INSOLE)**
for every shoe in the DB from scratch every time it's called (nothing is read from
a cache). Review evidence is selected with BGE-M3 sentence embeddings (cosine
similarity against a need sentence built from the foot-state thresholds), one
sentence per review, up to 3 distinct reviews per body part.

**No LLM call happens here** — `pointSummary`/`reviewSummary` are intentionally
left out of the forwarded payload (see `shoe-summaries` below for why) so this
stays fast (~20s for ~180 shoes). Forwards the result to
`SHOE_RECOMMENDATION_ENDPOINT`.

```bash
curl -X POST "http://localhost:8000/api/reports/shoe-recommendations" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"measurementSessionId": 30}'
```

### `POST /api/shoes/summaries`

```json
{ "shoeId": 220, "userId": 3 }
```

Called by Feetfit_Server (fire-and-forget) when a shoe detail view finds
`pointSummary == null`. Returns `202` immediately; generation happens in a
background task:

1. Read the **already-computed** fitScore/riskLevel/title and evidence review
   texts for that (`userId`, `shoeId`) straight out of the DB — nothing is
   recalculated (no embeddings, no foot-state lookup).
2. Call Ollama once (`OLLAMA_MODEL`, temperature `OLLAMA_TEMPERATURE`) to turn
   that into natural Korean `pointSummary` + `forefootSummary`/`heelSummary`/
   `insoleSummary`. The prompt explicitly forbids inventing review content for
   body parts with zero evidence reviews, forbids medical-diagnosis language,
   and must not contradict the given riskLevel. Response is parsed as JSON with
   a regex-extraction fallback if the model wraps it in prose/markdown.
3. `POST` the generated summary to `{shoe_id}/summaries` on Feetfit_Server
   (`SHOE_SUMMARY_SAVE_ENDPOINT_TEMPLATE`), forwarding the same Bearer token.

Any failure (missing DB row, Ollama error, save callback failure) is logged and
silently dropped — Feetfit_Server will simply retry on the next detail view
since `pointSummary` stays `null`.

```bash
curl -X POST "http://localhost:8000/api/shoes/summaries" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"shoeId": 220, "userId": 3}'
```

## `.env` reference

```text
TINEA_REPORT_ENDPOINT=http://35.94.253.151/api/reports/tina-pedis
HALLUX_VALGUS_REPORT_ENDPOINT=http://35.94.253.151/api/reports/hallux-valgus
SHOE_RECOMMENDATION_ENDPOINT=http://54.184.58.176/api/shoes/recommendations
SHOE_SUMMARY_SAVE_ENDPOINT_TEMPLATE=http://54.184.58.176/api/shoes/{shoe_id}/summaries
REPORT_PROXY_TIMEOUT_SECONDS=60

# OpenAI report text generation
OPENAI_API_KEY=paste-your-openai-api-key-here
OPENAI_REPORT_MODEL=gpt-4.1-mini
OPENAI_REPORT_TIMEOUT_SECONDS=20
OPENAI_REPORT_TEXT_ENABLED=true
OPENAI_REPORT_INCLUDE_IMAGES=true

# Tinea analysis and visualization
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

# Single-photo ArUco pipeline
ARUCO_SOURCE_DIR=D:/ArUco-marker-code
ARUCO_DICTIONARY=DICT_4X4_50
ARUCO_EXPECTED_IMAGE_WIDTH=1280
ARUCO_EXPECTED_IMAGE_HEIGHT=720
ARUCO_MARKER_SIZE_MM=20
ARUCO_MARKER_ROW_SPACING_MM=171
ARUCO_MARKER_COLUMN_SPACING_MM=140
ARUCO_FIXED_OFFSET_MM=113
ARUCO_IMAGE_LEFT_ANATOMICAL_SIDE=left

# shared MySQL DB (same instance/schema Feetfit_Server uses)
SHOE_DB_URL=jdbc:mysql://<host>:3306/feetfit?serverTimezone=Asia/Seoul&characterEncoding=UTF-8&useSSL=false&allowPublicKeyRetrieval=true
SHOE_DB_USERNAME=feetfit
SHOE_DB_PASSWORD=<password>

# BGE-M3 sentence embedding (shoe-recommendations batch)
SHOE_EMBEDDING_DEVICE=auto        # auto | cpu | cuda
SHOE_MAX_CANDIDATE_REVIEWS_PER_REASON=40
SHOE_REVIEWS_PER_REASON=3
SHOE_RISK_LOW_MIN_SCORE=70
SHOE_RISK_MEDIUM_MIN_SCORE=40

# Ollama (shoe-summaries on-demand generation)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TEMPERATURE=0.3
```
