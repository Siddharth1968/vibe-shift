# VibeShift — Architecture, Evaluation & Verification Reference

This document details the finalized single-model AI architecture (**CleanBTNVit Phase 12** + **YuNet**), exact preprocessing, evaluation metrics, confusion matrices, latency benchmarks, and database/recommendation architecture for VibeShift.

---

## 1. Final Single-Model AI Architecture

VibeShift deploys **one facial-expression recognition model (`CleanBTNVit — Phase 12`)** paired with **one face detector (`YuNet`)**. No secondary emotion classifiers, fallback emotion models, or ensembles exist in this codebase.

### A. Face Detector (`backend/ai/face_detector.py`)
- **Model:** OpenCV YuNet ONNX (`backend/ai/models/yunet-2023mar.onnx`, MIT License, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`).
- **Execution:** Thread-safe `cv2.FaceDetectorYN` with two-stage score thresholding (`0.9` primary, automatic `0.6` fallback), bounding-box clamping, and minimum-dimension filtering.
- **Live Multi-Face Tracking (`backend/ai/live_pipeline.py`):** Supports user click-to-select face targeting, IoU/normalized center-distance tracking across frames, a `0.8 s` reacquisition grace window, 2-frame EMA probability stabilization (`alpha = 0.5`) for continuous video, immediate raw prediction for `snapshot=True` photo uploads, and automatic TTL session pruning (`300 s`, max `128` sessions).

### B. Emotion Classifier (`backend/ai/btn_runtime/` + `backend/ai/emotion_model.py`)
- **Architecture:** `CleanBTNVit` wraps `BatchTransformerNetwork` and executes **strictly the batch-independent `output_VIT` pathway** at `112×112` resolution:
  1. Frozen `MobileFaceNet` (`112×112`) landmark branch generating 3-scale spatial queries (`28×28×64`, `14×14×128`, `7×7×256`).
  2. `IR-50` appearance backbone extracting 3-scale feature pyramids (`body1`, `body2`, `body3`).
  3. Three multi-scale landmark-guided cross-attention blocks (`window_size=[28, 14, 7]`, `num_heads=[2, 4, 8]`, `dims=[64, 128, 256]`).
  4. 2-layer `VisionTransformer` (`embed_dim=768`, 147 fused tokens + `[CLS]` token), Squeeze-and-Excitation (`se_block`), and 7-class linear head (`VIT.head`).
- **Expected Checkpoint:** `backend/ai/models/clean_btn_vit_phase12.pt` (SHA-256 `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58`, `790` tensors, loaded with `torch.load(..., weights_only=True)` strictly once during FastAPI `lifespan()` startup).
- **Inference Modes:**
  - **Single-view (default, `horizontal_flip=False`):** One FP16 CUDA (or FP32 CPU) forward pass.
  - **Horizontal-flip (optional accuracy mode, `horizontal_flip=True`):** Averages softmax probabilities across original and horizontally flipped tensors.

---

## 2. Exact Preprocessing & Canonical Seven-Class Mapping

Implemented in `backend/ai/preprocess.py :: preprocess_face_for_clean_btn`:
1. Crop YuNet bounding box (`margin_ratio=0.06`, clamped to image bounds).
2. Convert BGR crop to luminance uint8 grayscale (preserving pixel values identically if already grayscale).
3. Convert to `(1, 1, H, W)` float32 in `[0, 1]`, resize to `(112, 112)` via PyTorch bilinear interpolation (`align_corners=False`), and apply `.mul(255.0).round().div(255.0)`.
4. Broadcast-normalize across 3 channels with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

| Logit Index | Facial Expression | VibeShift Music Mood | Handling |
|---:|---|---|---|
| `0` | `angry` | `angry` | Suggested (user confirms) |
| `1` | `disgust` | `null` | Unmapped — prompts manual selection |
| `2` | `fear` | `null` | Unmapped — prompts manual selection |
| `3` | `happy` | `happy` | Suggested (user confirms) |
| `4` | `neutral` | `neutral` | Suggested (user confirms) |
| `5` | `sad` | `sad` | Suggested (user confirms) |
| `6` | `surprise` | `null` | Unmapped — prompts manual selection |

Predictions with top probability `< 0.40` return `status="low_confidence"` and are never elevated to `"success"`.

---

## 3. Evaluation Metrics & Confusion Matrices

> **Evaluation Disclosure:** All metrics below were measured on the **reused 4,015-image Training-derived FER-2013 development holdout** (`split.csv`), **not an independent test set**. `PublicTest` and `PrivateTest` were strictly untouched during Phase 12 training.
> **Licensing Disclosure:** Stage 1 knowledge distillation used 5,962 ExpW images (CUHK MMLab, non-commercial academic/research license) initialized from AffectNet-7 weights (University of Denver non-commercial academic EULA). See `docs/MODEL_SETUP.md` and `docs/THIRD_PARTY_LICENSES.md`.

### Summary & Backend Parity Proof (4,015 Reused Dev Holdout)

| Mode | Accuracy (Correct / 4,015) | Macro F1 | NLL | ECE | Backend Parity Delta |
|---|---:|---:|---:|---:|---:|
| **Single-View (Default)** | **74.10%** (`2,975 / 4,015`) | 72.37% | 0.7467 | 0.0390 | `0.000000` |
| **Horizontal-Flip (Accuracy Mode)** | **74.37%** (`2,986 / 4,015`) | 72.78% | 0.7338 | 0.0314 | `0.000000` |

### Per-Class Precision / Recall / F1

| Expression | Support | Single-View Prec / Rec / F1 | Horizontal-Flip Prec / Rec / F1 |
|---|---:|---:|---:|
| `angry` | 561 | 72.01% / 63.28% / 67.36% | 72.23% / 63.99% / 67.86% |
| `disgust` | 52 | 73.58% / 75.00% / 74.29% | 77.55% / 73.08% / 75.25% |
| `fear` | 564 | 62.90% / 55.32% / 58.87% | 63.65% / 56.21% / 59.70% |
| `happy` | 1,054 | 89.74% / 92.13% / 90.92% | 89.08% / 92.13% / 90.58% |
| `neutral` | 721 | 69.16% / 76.84% / 72.80% | 69.14% / 77.67% / 73.15% |
| `sad` | 698 | 63.39% / 66.48% / 64.90% | 64.48% / 66.05% / 65.25% |
| `surprise` | 365 | 78.21% / 76.71% / 77.46% | 78.65% / 76.71% / 77.67% |

### Confusion Matrix — Single-View Default (Rows = True, Cols = Pred)

| True \ Pred | `angry` | `disgust` | `fear` | `happy` | `neutral` | `sad` | `surprise` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`angry`** | **355** | 9 | 45 | 18 | 56 | 67 | 11 |
| **`disgust`** | 8 | **39** | 1 | 1 | 0 | 3 | 0 |
| **`fear`** | 43 | 3 | **312** | 14 | 45 | 105 | 42 |
| **`happy`** | 14 | 0 | 7 | **971** | 41 | 6 | 15 |
| **`neutral`** | 22 | 0 | 20 | 38 | **554** | 82 | 5 |
| **`sad`** | 44 | 2 | 73 | 17 | 93 | **464** | 5 |
| **`surprise`** | 7 | 0 | 38 | 23 | 12 | 5 | **280** |

### Confusion Matrix — Horizontal-Flip Accuracy Mode (Rows = True, Cols = Pred)

| True \ Pred | `angry` | `disgust` | `fear` | `happy` | `neutral` | `sad` | `surprise` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`angry`** | **359** | 7 | 45 | 21 | 56 | 62 | 11 |
| **`disgust`** | 8 | **38** | 1 | 1 | 0 | 4 | 0 |
| **`fear`** | 43 | 2 | **317** | 13 | 47 | 101 | 41 |
| **`happy`** | 16 | 0 | 9 | **971** | 41 | 4 | 13 |
| **`neutral`** | 21 | 0 | 17 | 40 | **560** | 78 | 5 |
| **`sad`** | 44 | 2 | 73 | 18 | 94 | **461** | 6 |
| **`surprise`** | 6 | 0 | 36 | 26 | 12 | 5 | **280** |

---

## 4. Latency & GPU Memory Benchmarks (RTX 3050 Laptop GPU, FP16)

| Benchmark | Single-View (Default) | Horizontal-Flip (Accuracy Mode) |
|---|---:|---:|
| **Model-Only Forward Pass** (batch=1, 100 iters) | **22.53 ms** median (`p95 31.63 ms`, **44.38 FPS**) | **42.63 ms** median (`p95 63.76 ms`, **23.46 FPS**) |
| **Fresh Standalone GPU VRAM** (batch=1 FP16) | `355.1 MiB` alloc / `374.0 MiB` reserved | `368.7 MiB` alloc / `396.0 MiB` reserved |
| **Full 640×480 Frame → Emotion Pipeline** | **43.70 ms** median | **67.24 ms** median |
| **SQLite Recommendation Query (137 songs)** | **1.96 ms** median | **2.02 ms** median |
| **Full Frame → Recommendations End-to-End** | **45.61 ms** median (`p95 55.51 ms`) | **69.25 ms** median (`p95 76.80 ms`) |

---

## 5. Database & Recommendation Engine

- **Auto-Initialized SQLite Schema (`backend/database/db.py`):** Four relational tables (`songs`, `sessions`, `mood_records`, `feedback`) with `PRAGMA foreign_keys = ON;`. Automatically creates `backend/database/vibeshift.db` on first startup and seeds all 137 tracks from `backend/data/songs.csv`.
- **Catalogue Distribution:** English (`43`), Tamil (`40`), Hindi (`35`), Malayalam (`19`); `137 / 137` validated Spotify track URLs.
- **Recommendation Scoring (`backend/services/recommendation.py`):** Mood match `+3`, Genre match `+2`, Context match `+1`, Feedback modifiers `like +2`, `replay +1`, `skip -2`, strict single-language isolation when a language is selected, and balanced round-robin interleaving in `"All Supported"` mode.
