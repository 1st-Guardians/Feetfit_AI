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
    shoe_server.py              Feetfit_Server internal read contract schemas
    shoe_fit_comment.py         Ollama-generated summary response schema
  services/
    tinea_analysis.py           Foot/tinea segmentation, scoring, suspicion map render
    hallux_valgus_analysis.py   Foot outline extraction, hallux valgus angle (HVA) scoring
    shoe_db.py                  Legacy direct-DB adapter (not used by HTTP routes)
    shoe_server_client.py       Forwarded-JWT Feetfit_Server context/callback client
    shoe_embedding.py           BGE-M3 sentence embedding + disk cache + cosine ranking
    shoe_feature_rules.py       Session foot-need text used by semantic review search
    shoe_fit_policy.py          RunRepeat quantitative TEMPORARY_HEURISTIC policy
    shoe_recommendation.py      전체 신발 fitScore/riskLevel + BGE-M3 후보 계산
    shoe_fit_comment_service.py Ollama 후보 subset 검증 + point/review summaries
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

- Feetfit_Server's internal shoe-analysis endpoints. The incoming user Bearer token
  is forwarded unchanged together with `X-Internal-Api-Key`. Configure
  `FEETFIT_SERVER_INTERNAL_API_KEY` to the same service key as Feetfit_Server;
  missing or blank keys fail before any request is sent. The shared DB is not used
  as a fallback when an internal API request fails.
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

### `POST /api/reports/foot-type-text`

Feetfit_Server의 measurement session이 `COMPLETED`로 commit된 뒤 Server가
해당 세션의 DB 분석 결과를 다시 읽어 자동으로 호출하는 내부 전용
endpoint입니다. 클라이언트가 `typeText`나 `careTips`를 작성하여 보내지
않습니다. Server는 exact-session facts와 해당 facts의 SHA-256
`factsHash`를 함께 전달합니다.

```json
{
  "measurementSessionId": 30,
  "measurementStatus": "COMPLETED",
  "factsHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "analysis": {
    "archType": "LOW",
    "footWidthType": "WIDE",
    "pressureBalanceType": "BALANCED",
    "measuredLeftFootSizeMm": 253.0,
    "measuredRightFootSizeMm": 251.0,
    "leftFootWidthMm": 101.0,
    "rightFootWidthMm": 100.0,
    "leftPressurePercent": 49.0,
    "rightPressurePercent": 51.0,
    "plantarFootprintAnalysisText": "발바닥 중앙부와 뒤꿈치의 압력 분포 차이가 관찰됩니다."
  }
}
```

GPT는 `ARCH`, `WIDTH`, `PRESSURE_BALANCE`의 명시적 분류 중 신발 선택에
가장 유용한 evidence ID만 고릅니다. 최종 한국어는 해당 evidence에 고정된
검증 문구로 렌더링되므로 GPT가 평발이나 발볼 특성을 새로 추측할 수
없습니다. OpenAI 호출이 실패하거나 잘못된 ID를 반환하면 `ARCH → WIDTH →
PRESSURE_BALANCE` 순서의 로컬 fallback을 사용합니다. 원시 발 길이와 너비는
추적용일 뿐, 그 숫자만으로 아치나 발볼 유형을 분류하지 않습니다.
OpenAI에는 원시 측정값이 아닌 검증된 candidate evidence만 전달하며,
최종 문구는 `"이번 측정에서는"`으로 시작하지 않도록 응답 계약에서도
검증합니다.

AI는 DB를 수정하지 않고 다음 pure-generation 응답만 반환합니다.

```json
{
  "measurementSessionId": 30,
  "factsHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "typeText": "발의 아치가 낮아 발바닥이 넓게 닿는 편이에요. 오래 걷거나 서 있으면 피로가 커질 수 있어 아치를 잘 받쳐주는 신발이 더 편안할 수 있어요.",
  "evidenceId": "ARCH_LOW",
  "source": "OPENAI"
}
```

Server는 응답의 `measurementSessionId`와 `factsHash`를 현재 DB facts와 다시
검증하고, 동일 완료 세션의 `typeText`가 아직 없을 때만 `typeText`를
저장합니다. `careTips`는 읽거나 덮어쓰지 않습니다. 이후 신발 목록은
Server의 `GET /api/reports/foot-type-text`로 저장된 문구를 조회합니다.

```bash
# 정상 사용 시 Server가 자동 호출하며, 아래는 로컬 내부-계약 점검용 예시입니다.
curl -X POST "http://localhost:8000/api/reports/foot-type-text" \
  -H "Authorization: Bearer <SERVER_MINTED_TOKEN>" \
  -H "X-Internal-Api-Key: <FEETFIT_SERVER_INTERNAL_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"measurementSessionId":30,"measurementStatus":"COMPLETED","factsHash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","analysis":{"archType":"LOW"}}'
```

### `POST /api/reports/shoe-recommendations`

```json
{ "measurementSessionId": 30 }
```

Uses the forwarded Bearer token to page through Feetfit_Server's internal
recommendation-context endpoint. The response is fixed to the requested completed
measurement session and contains its foot analyses, MUSINSA-only reviews, and raw
RunRepeat lab measurements. The Phase-D scorer computes **fitScore + riskLevel
(FOREFOOT/HEEL/INSOLE)** from the relationship between the one requested session's
foot measurements and each shoe's real RunRepeat metrics. Review sentiment never
changes score or risk. Missing characteristics are not synthesized; present
components are reweighted, and a body area with zero real components fails closed.
Every catalog shoe is returned, including shoes with zero reviews (`reviewIds: []`).
BGE-M3 selects at most three shoe-local MUSINSA candidates per body part.

The numerical policy and semantic threshold are explicitly
`TEMPORARY_HEURISTIC` / `NOT_CLINICALLY_VALIDATED`; weights, allowances and risk/
pressure/humidity thresholds are environment-backed settings rather than clinical
claims. The shoe comparison feature is intentionally out of scope.

**No LLM call happens here.** The first-stage save explicitly sends
`pointSummary: null` and `reviewSummary: null`; null means summary generation is
pending. The result is forwarded to `FEETFIT_SERVER_RECOMMENDATION_ENDPOINT`.

```bash
curl -X POST "http://localhost:8000/api/reports/shoe-recommendations" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "X-Internal-Api-Key: <FEETFIT_SERVER_INTERNAL_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"measurementSessionId": 30}'
```

### `POST /api/shoes/summaries`

```json
{ "shoeId": 220, "measurementSessionId": 30 }
```

Must be called by Feetfit_Server (fire-and-forget) when a shoe detail view finds
`pointSummary == null`; this AI endpoint cannot initiate that hook itself. It
returns `202` immediately and generation happens in a background task:

1. Read the **already-computed** fitScore/riskLevel/title and evidence review
   texts from Feetfit_Server's JWT-scoped summary-context endpoint. Nothing is
   recalculated (no embeddings and no foot-state lookup). The exact completed
   `measurementSessionId` is required; no current/older-session fallback is used.
2. Call Ollama once (`OLLAMA_MODEL`, temperature `OLLAMA_TEMPERATURE`) to
   semantically remove unrelated BGE candidates (the returned IDs must be a unique
   subset of the exact shoe/reason candidates, maximum three), and turn the facts
   into Korean `pointSummary` + `forefootSummary`/`heelSummary`/`insoleSummary`.
   The prompt explicitly forbids inventing review content for
   body parts with zero evidence reviews, forbids medical-diagnosis language,
   and must not contradict the given riskLevel. Response is parsed as JSON with
   a regex-extraction fallback if the model wraps it in prose/markdown.
3. `POST` the generated summary to `{shoe_id}/summaries` on Feetfit_Server
   (`FEETFIT_SERVER_SUMMARY_SAVE_ENDPOINT_TEMPLATE`), forwarding the same Bearer
   token, exact measurementSessionId, summaries, and the validated final reviewIds.

Before generation, the AI also reads `GET /api/shoes/{shoe_id}/characteristics`
through `FEETFIT_SERVER_CHARACTERISTICS_ENDPOINT_TEMPLATE`. The returned RunRepeat
levels remain separate from subjective review evidence (for example cushioning,
shock absorption, and sole thickness are never treated as interchangeable).

Any failure (missing Server context, Ollama error, save callback failure) is logged and
silently dropped. Once the Server detail-view hook is installed, the next detail
view can retry because `pointSummary` stays `null`.

```bash
curl -X POST "http://localhost:8000/api/shoes/summaries" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "X-Internal-Api-Key: <FEETFIT_SERVER_INTERNAL_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"shoeId": 220, "measurementSessionId": 30}'
```

## `.env` reference

```text
TINEA_REPORT_ENDPOINT=http://35.94.253.151/api/reports/tina-pedis
HALLUX_VALGUS_REPORT_ENDPOINT=http://35.94.253.151/api/reports/hallux-valgus
FEETFIT_SERVER_RECOMMENDATION_ENDPOINT=http://127.0.0.1:8080/api/shoes/recommendations
FEETFIT_SERVER_SUMMARY_SAVE_ENDPOINT_TEMPLATE=http://127.0.0.1:8080/api/shoes/{shoe_id}/summaries
FEETFIT_SERVER_RECOMMENDATION_CONTEXT_ENDPOINT=http://127.0.0.1:8080/api/internal/shoe-analysis/recommendation-context
FEETFIT_SERVER_SUMMARY_CONTEXT_ENDPOINT_TEMPLATE=http://127.0.0.1:8080/api/internal/shoe-analysis/shoes/{shoe_id}/recommendation-summary-context
FEETFIT_SERVER_CHARACTERISTICS_ENDPOINT_TEMPLATE=http://127.0.0.1:8080/api/shoes/{shoe_id}/characteristics
SHOE_RECOMMENDATION_CONTEXT_PAGE_SIZE=100  # 1..200 (Feetfit_Server contract)
FEETFIT_SERVER_INTERNAL_API_KEY=<same-service-key-configured-on-Feetfit_Server>  # preferred
INTERNAL_API_KEY=<same-value>  # accepted shared-name fallback
REPORT_PROXY_TIMEOUT_SECONDS=60
FEETFIT_SERVER_CALLBACK_TIMEOUT_SECONDS=900  # callback floor; must be >= 900

# OpenAI report text generation
OPENAI_API_KEY=paste-your-openai-api-key-here
OPENAI_REPORT_MODEL=gpt-4.1-mini
OPENAI_REPORT_TIMEOUT_SECONDS=20
OPENAI_REPORT_TEXT_ENABLED=true
OPENAI_FOOT_TYPE_TEXT_ENABLED=true
FOOT_TYPE_PRESSURE_BALANCE_TOLERANCE_PERCENT=5
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

# BGE-M3 sentence embedding (shoe-recommendations batch)
SHOE_EMBEDDING_DEVICE=auto        # auto | cpu | cuda
SHOE_MAX_CANDIDATE_REVIEWS_PER_REASON=40
SHOE_REVIEWS_PER_REASON=3                  # 1..3 (save payload contract)
SHOE_REVIEW_SEMANTIC_MIN_SCORE=0.42        # TEMPORARY_HEURISTIC
SHOE_RELEASE_EMBEDDING_MODEL_AFTER_BATCH=true
SHOE_RISK_LOW_MIN_SCORE=75
SHOE_RISK_MEDIUM_MIN_SCORE=50

# Ollama (shoe-summaries on-demand generation)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TEMPERATURE=0.0
OLLAMA_NUM_GPU=-1                 # GPU preferred; 0 forces CPU
OLLAMA_CPU_FALLBACK_ENABLED=true
```

All `FEETFIT_SERVER_*_ENDPOINT*` values must be set to the intended deployment
environment. Defaults are loopback-only and never point at a public Server. The
internal API key is mandatory: a blank `FEETFIT_SERVER_INTERNAL_API_KEY` fails
before any Server request and is never logged.
