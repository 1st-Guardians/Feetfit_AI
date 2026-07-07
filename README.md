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

For now, the three description fields are filled with longer temporary Korean
comments before forwarding. Later, this can be replaced with GPT-generated text.

```bash
curl -X POST "http://localhost:8000/api/reports/tina-pedis" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "leftFootImage=@left-foot.jpeg;type=image/jpeg" \
  -F "rightFootImage=@right-foot.jpeg;type=image/jpeg"
```

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
TINEA_REPORT_ENDPOINT=http://54.184.58.176/api/reports/tina-pedis
HALLUX_VALGUS_REPORT_ENDPOINT=http://54.184.58.176/api/reports/hallux-valgus
SHOE_RECOMMENDATION_ENDPOINT=http://54.184.58.176/api/shoes/recommendations
SHOE_SUMMARY_SAVE_ENDPOINT_TEMPLATE=http://54.184.58.176/api/shoes/{shoe_id}/summaries
REPORT_PROXY_TIMEOUT_SECONDS=60
COMBINED_IMAGE_MAX_HEIGHT=1600
COMBINED_IMAGE_GAP_PIXELS=16
FUNGAL_THRESHOLD=0.78
MAX_DOT_AREA_FOR_SUSPICION_MAP=450
PHOTO_CUTOUT_BACKGROUND=true
# INFLAMMATION_THRESHOLD=0.88

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
