# VibeShift — AI Model Asset Setup & Licensing

VibeShift uses a **single standalone emotion-recognition model (`CleanBTNVit — Phase 12`)** paired with **OpenCV YuNet (`yunet-2023mar.onnx`)** face detection. No emotion-model ensembles are used.

---

## 1. Included Model Asset (MIT Licensed)

| Asset | Path in Repository | Size | SHA-256 | License |
|---|---|---:|---|---|
| **YuNet Face Detector** | `backend/ai/models/yunet-2023mar.onnx` | ~228 KB | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT License (Shiqi Yu — see `docs/THIRD_PARTY_LICENSES.md`) |

This asset is included directly in the public repository.

---

## 2. Separately Installed Emotion Classifier Checkpoint (Not Redistributed Publicly)

| Property | Value |
|---|---|
| **Architecture** | `CleanBTNVit` (`BatchTransformerNetwork` strictly batch-independent `output_VIT` pathway at `112×112`) |
| **Expected Local Path** | `backend/ai/models/clean_btn_vit_phase12.pt` (or custom path via `VIBESHIFT_BTN_WEIGHTS`) |
| **Expected Size** | ~257.3 MB (`257,312,xxx` bytes, `790` state_dict tensors) |
| **Required SHA-256** | `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58` |
| **Canonical 7-Class Order** | `0: angry`, `1: disgust`, `2: fear`, `3: happy`, `4: neutral`, `5: sad`, `6: surprise` |
| **Verified Reused Dev Holdout Accuracy** | **74.10%** single-view (default) / **74.37%** horizontal-flip (optional accuracy mode) on the reused 4,015-image FER-2013 Training-derived development holdout *(not an independent test set)* |

### Why `clean_btn_vit_phase12.pt` Is Excluded From This Public Repository

Although the model architecture code in `backend/ai/btn_runtime/` is MIT-licensed (`SeoulTech-HCIRLab/BTN`), the Phase 12 checkpoint weights were initialized and distilled using multiple academic facial-expression datasets and pretrained checkpoints whose public redistribution rights **cannot be legally established**:

- **AffectNet** (University of Denver): Non-commercial academic research EULA; does not grant blanket public redistribution rights for derived weights.
- **ExpW** (CUHK MMLab): Restricted to non-commercial academic and educational research; no public redistribution grant.
- **RAF-DB** (used via cross-family teacher distillation): Restricted to non-commercial academic research.
- **MS-Celeb-1M** (upstream pretraining of the IR-50 backbone): Retired dataset with restricted redistribution status.

Accordingly, `backend/ai/models/clean_btn_vit_phase12.pt` is listed in `.gitignore` and is **not uploaded** to the public GitHub repository. A fresh clone of this repository is **not self-contained for camera AI inference** until authorized weights are placed locally.

### Installing Authorized Weights Locally

1. Place your authorized Phase 12 `CleanBTNVit` checkpoint file at:
   ```text
   backend/ai/models/clean_btn_vit_phase12.pt
   ```
   *(Or set `VIBESHIFT_BTN_WEIGHTS=/path/to/clean_btn_vit_phase12.pt` in `.env`.)*
2. Verify the SHA-256 checksum on Windows PowerShell:
   ```powershell
   Get-FileHash backend\ai\models\clean_btn_vit_phase12.pt -Algorithm SHA256
   ```
   Expected hash:
   ```text
   0316EA2A96E671F562A770ACE64B8665E8BC1996BE0046CBE94C4674BB0C5D58
   ```
3. Start the backend (`run_backend.bat`). `CleanBTNVitAdapter` verifies the SHA-256 hash automatically and loads the weights safely using `torch.load(..., weights_only=True)` once during application startup.

### Graceful Behavior When Weights Are Absent

If `clean_btn_vit_phase12.pt` is not installed yet:
- The FastAPI server still starts cleanly and `GET /api/health` returns `{"status": "ok", "ai": {"model_loaded": false, ...}}`.
- Camera/photo emotion endpoints safely block unvalidated inference and return `status: "model_weights_not_loaded"`, `emotion: null`, and a prompt to choose a mood manually.
- All manual mood selection, Match/Change mode, 4-language music recommendations, Spotify Embeds, feedback adaptation, and Mood Journey features work normally.
