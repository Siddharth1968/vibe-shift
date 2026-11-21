# VibeShift — AI Model Asset Setup & Licensing

VibeShift uses **one standalone facial-expression classifier (`CleanBTNVit — Phase 12`)** paired with **OpenCV YuNet (`yunet-2023mar.onnx`)** face detection.

---

## 1. Included Model Asset (MIT Licensed)

| Asset | Path in Repository | Size | SHA-256 | License |
|---|---|---:|---|---|
| **YuNet Face Detector** | `backend/ai/models/yunet-2023mar.onnx` | ~227 KB | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT License (Shiqi Yu — see `docs/THIRD_PARTY_LICENSES.md`) |

---

## 2. Separately Installed Emotion Classifier Checkpoint (Not Redistributed Publicly)

| Property | Value |
|---|---|
| **Architecture** | `CleanBTNVit` (`BatchTransformerNetwork` strictly batch-independent `output_VIT` pathway at `112×112`) |
| **Expected Local Path** | `backend/ai/models/clean_btn_vit_phase12.pt` (or custom path via `VIBESHIFT_BTN_WEIGHTS`) |
| **Exact File Size** | `257,223,791` bytes (~245.3 MiB, `790` tensors) |
| **Required SHA-256** | `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58` |
| **Canonical 7-Class Order** | `0: angry`, `1: disgust`, `2: fear`, `3: happy`, `4: neutral`, `5: sad`, `6: surprise` |
| **Verified Reused Dev Holdout Accuracy** | **74.10%** single-view (default) / **74.37%** horizontal-flip (optional accuracy mode) on the reused 4,015-image FER-2013 Training-derived development holdout *(not an independent test set)* |

---

## 3. Licensing Assessment: Explicit Prohibitions vs. Unestablished Redistribution Rights

Although `CleanBTNVit` model weights are learned parameters rather than raw dataset images, **public redistribution on GitHub Releases or public Git LFS is not established as legally authorized** across its upstream lineage:

| Upstream Component | Explicit Prohibition in Terms | Status of Derived Model Weights |
|---|---|---|
| **AffectNet** (University of Denver — initialized `AffectNet7_67.60.pth`) | Explicitly restricted to **non-commercial research and educational use**; prohibits providing the dataset to third parties; requires faculty Lead Researcher registration and a separate commercial license for non-academic/commercial use. | Does not explicitly grant public open redistribution rights for derived weights; unrestricted public GitHub hosting cannot enforce academic/non-commercial eligibility. |
| **ExpW** (CUHK MMLab — 5,962 images used in Stage 1 soft-KD) | Explicitly restricted to **non-commercial academic and educational research**; creators disclaim copyright over raw web-sourced face images. | No explicit prohibition naming model weights, and **no grant of public redistribution rights**. |
| **RAF-DB** (BUPT / PRIS — trained Stage 1 `raf_btn` teacher) | Explicitly restricts database to non-commercial research and prohibits commercial exploitation of *"any portion of the images and **any portion of derived data**"*. | Explicit derived-data commercial restriction; no public redistribution waiver granted. |
| **MS-Celeb-1M** (Microsoft Research — pretrained `ir50.pth` inside `btn.ir_back`) | Original license restricted use to non-commercial research and prohibited redistributing *"the corpus or **any derivative works of the corpus**"*; Microsoft **retracted and deleted** the dataset in June 2019 over privacy concerns. | Legally unsettled/disputed whether pretrained weights constitute a contractual derivative work; no public redistribution permission exists. |
| **`SeoulTech-HCIRLab/BTN` Dropbox Checkpoints** (`AffectNet7_67.60.pth`, `ir50.pth`, `mobilefacenet`) | Repository code is **MIT License**, but externally hosted Dropbox checkpoints carry **no explicit weights license**. | Redistribution permission for external weights is **not explicitly established**. |
| **FER-2013** (ICML 2013 / Kaggle) | Open academic benchmark; no active rule prohibiting fine-tuned weights. | Does not override restrictions from the components above. |

Because public redistribution rights remain uncertain/unestablished, `backend/ai/models/clean_btn_vit_phase12.pt` is excluded via `.gitignore` and is **not hosted publicly** in this repository.

---

## 4. Quickest Authorized Procedure for Academic Collaborators

To run the complete VibeShift application with real `CleanBTNVit` AI inference:

### Step 1 — Confirm Academic Collaborator Eligibility
Before sharing the checkpoint directly with classmates or academic teammates:
- Confirm all recipients are part of the same **non-commercial academic/educational project** (not for commercial deployment).
- Where institutional dataset agreements apply (e.g., AffectNet faculty Lead Researcher supervision / institutional academic-use scope), ensure your project supervisor/advisor authorizes direct non-commercial academic sharing within your project group rather than public distribution.
- Transfer `clean_btn_vit_phase12.pt` (`245.3 MiB`) via a **private collaborator-only channel** (e.g., direct USB transfer, university OneDrive/Google Drive link restricted to invited teammates, or private access-controlled institutional storage — never a public download link).

### Step 2 — Clone & Install Verified Checkpoint Automatically

**Option A — One-Step Setup with Local Transferred File (Windows CMD / PowerShell):**
```powershell
git clone https://github.com/Siddharth1968/vibe-shift.git
cd vibe-shift

# Point setup.bat to your privately transferred checkpoint file (optional CUDA GPU: set VIBESHIFT_INSTALL_CUDA=1)
$env:VIBESHIFT_MODEL_PATH = "D:\shared\clean_btn_vit_phase12.pt"
.\setup.bat
```

**Option B — Install Checkpoint After Running `setup.bat`:**
```powershell
.\.venv\Scripts\python.exe scripts\install_model.py --from-file "C:\path\to\clean_btn_vit_phase12.pt"
```

**Option C — Install from a Private Authorized Collaborator URL:**
```powershell
.\.venv\Scripts\python.exe scripts\install_model.py --url "<YOUR_PRIVATE_AUTHORIZED_DOWNLOAD_URL>"
```

**Option D — Manual Placement & Verification:**
1. Copy the file to `backend/ai/models/clean_btn_vit_phase12.pt`.
2. Run verification:
   ```powershell
   .\.venv\Scripts\python.exe scripts\install_model.py --verify-only
   ```
   Expected SHA-256: `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58`.

### Step 3 — Launch VibeShift
Run `run_backend.bat` and `run_frontend.bat` and open `http://localhost:5173`. Confirm `GET http://127.0.0.1:8000/api/health` reports `"model_loaded": true`.

---

## 5. Graceful Fallback Without Checkpoint

If `clean_btn_vit_phase12.pt` is not installed yet:
- FastAPI starts cleanly and `GET /api/health` reports `"model_loaded": false`.
- Emotion endpoints safely block unvalidated inference (`status: "model_weights_not_loaded"`, `emotion: null`) and prompt manual mood selection.
- Manual mood selection, Match/Change mode, 4-language recommendations, Spotify embeds, feedback, and Mood Journey remain fully functional.
