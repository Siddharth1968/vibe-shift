# VibeShift REST API Contract

**Base URL:** `http://127.0.0.1:8000`  
**CORS Default Allowed Origins:** `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:4173`, `http://127.0.0.1:4173` (extensible via `VIBESHIFT_CORS_ORIGINS`).

---

## Endpoints Overview

| Method | Endpoint | Purpose | Success Status |
|---|---|---|---|
| `GET` | `/api/health` | Server & AI runtime readiness (`CleanBTNVit` + `YuNet`) | `200 OK` |
| `GET` | `/api/emotion/model` | Inspect active single-model CleanBTNVit metadata | `200 OK` |
| `POST` | `/api/emotion` | Single-image facial emotion estimation (`multipart/form-data`, `<= 5 MB`) | `200 OK` |
| `POST` | `/api/emotion/live` | Real-time webcam tracking or photo-upload snapshot inference | `200 OK` |
| `POST` | `/api/sessions` | Create a new listening session in SQLite | `201 Created` |
| `POST` | `/api/sessions/{id}/mood` | Log a confirmed mood (`manual`, `camera_suggested`, `recheck`) | `200 OK` |
| `GET` | `/api/sessions/{id}/journey` | Retrieve chronological mood history for the session | `200 OK` |
| `POST` | `/api/recommendations` | Return top-10 ranked songs matching/shifting mood & preferences | `200 OK` |
| `POST` | `/api/feedback` | Record song feedback (`like`, `skip`, `replay`) | `200 OK` |

---

## Detailed Specifications

### `GET /api/health`
Returns server status and startup-loaded AI readiness:
```json
{
  "status": "ok",
  "ai": {
    "initialized": true,
    "model": "CleanBTNVit (Phase 12)",
    "model_loaded": true,
    "checkpoint_sha256": "0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58",
    "device": "cuda (NVIDIA GeForce RTX 3050 Laptop GPU)",
    "default_inference_mode": "single_view",
    "detector": "YuNetFaceDetector",
    "detector_loaded": true
  }
}
```

### `POST /api/emotion`
- **Content-Type:** `multipart/form-data`
- **Fields:** `file` (JPEG/PNG/WEBP/BMP, `<= 5 MB`), optional `horizontal_flip` (`true` | `false`, defaults to `false`).
- **Privacy:** Processed strictly in memory; never written to disk.
- **Response (`EmotionResponse`):**
```json
{
  "emotion": "happy",
  "confidence": 0.93,
  "note": "This is only an estimate of facial expression, not proof of feelings.",
  "status": "success",
  "expression": "happy",
  "expression_scores": {
    "angry": 0.01,
    "disgust": 0.0,
    "fear": 0.01,
    "happy": 0.93,
    "neutral": 0.03,
    "sad": 0.01,
    "surprise": 0.01
  },
  "model_name": "CleanBTNVit (Phase 12)",
  "inference_mode": "single_view",
  "inference_ms": 22.5,
  "latency_ms": 43.7
}
```
- **Unmapped expressions (`surprise`, `fear`, `disgust`)** return `"emotion": null`, `"expression": "<label>"`, and prompt manual mood selection (never silently defaulting to `"neutral"`).

### `POST /api/emotion/live`
- **Content-Type:** `multipart/form-data`
- **Fields:** `file` (`<= 5 MB` when `snapshot=true`, `<= 2 MB` for live frames), `session` (`16..64` chars), `selection` (`"null"` or `"[x, y]"`), `snapshot` (`bool`), `debug` (`bool`), optional `horizontal_flip` (`bool`).
- **Behavior:** Continuous live frames apply 2-frame EMA confirmation (`alpha = 0.5`) and multi-face tracking; `snapshot=true` returns immediate single-frame results.

### `POST /api/sessions`
- **Body:** `{"language": "English", "genre": "Pop", "context": "relax"}` (all optional).
- **Response (`201`):** `{"session_id": 1, "created_at": "2026-11-21 15:23:33"}`

### `POST /api/sessions/{id}/mood`
- **Body:** `{"mood": "happy", "source": "camera_suggested", "confidence": 0.93}`
- **Allowed moods:** `happy`, `sad`, `calm`, `energetic`, `angry`, `neutral`.
- **Allowed sources:** `manual`, `camera_suggested`, `recheck`.

### `GET /api/sessions/{id}/journey`
- **Response:** `{"session_id": 1, "entries": [{"mood": "happy", "source": "camera_suggested", "recorded_at": "2026-11-21 15:23:33"}]}`

### `POST /api/recommendations`
- **Body:**
```json
{
  "current_mood": "happy",
  "mode": "match",
  "target_mood": null,
  "language": "English",
  "genre": "Pop",
  "context": "relax"
}
```
- **Response (`200`):** Top-level JSON array of up to 10 songs (`id`, `title`, `artist`, `language`, `genre`, `mood_tags`, `audio_url`).

### `POST /api/feedback`
- **Body:** `{"session_id": 1, "song_id": 2, "action": "like"}` (`action` in `like`, `skip`, `replay`).
