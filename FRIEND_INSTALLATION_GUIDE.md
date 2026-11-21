# VibeShift — Teammate Installation & Launch Guide (Windows)

Welcome! This guide walks you step-by-step through installing and running the complete **VibeShift** web application on a Windows 10 or Windows 11 laptop — including **YuNet face detection**, **CleanBTNVit Phase 12 facial-expression recognition**, manual mood selection, **Match My Mood / Change My Mood**, 4-language Spotify recommendations, feedback learning, and Mood Journey history.

---

## 1. System & Hardware Requirements (Measured)

| Requirement | Minimum | Recommended | Notes |
|---|---|---|---|
| **Operating System** | Windows 10 (64-bit) | Windows 11 (64-bit) | 64-bit Windows is required for PyTorch and OpenCV. |
| **Python** | **3.10, 3.11, or 3.12 (64-bit)** | **Python 3.12.10 (64-bit)** | **Do NOT use Python 3.13 or 3.14** (PyTorch wheels do not support them yet). Check **"Add python.exe to PATH"** during install. Download: https://www.python.org/downloads/release/python-31210/ |
| **Node.js & npm** | Node.js 18+ LTS | Node.js 20 or 22 LTS | Download: https://nodejs.org/ (includes `npm`). |
| **Git** | Not needed for Method A (ZIP) | Git for Windows | Needed **only** for Method B (GitHub clone): https://git-scm.com/ |
| **RAM** | **8 GB system RAM** | **16 GB system RAM** | **Measured** CPU backend process RAM is **~1.0 GiB** (`1,023.8 MiB` peak working set) when running CleanBTNVit; Chrome/Edge + Vite + OS require additional RAM. |
| **Free Disk Space** | **~1.5 GB** (CPU-only install) | **~5.0 GB** (NVIDIA CUDA GPU install) | PyTorch CUDA 12.6 wheel is `2.59 GB` (`.venv` ~4.4 GB total on GPU, ~1.0 GB on CPU) + `frontend/node_modules` (~220 MB) + checkpoint (`245.3 MiB`). |
| **Internet Connection** | Required for initial setup & Spotify playback | Broadband | Needed during `setup.bat` to download Python + npm packages and at runtime for Spotify Embeds. |
| **Webcam** | Built-in or USB camera | 720p+ webcam | Optional — you can also upload photos or use Manual Mood Mode without a webcam. |

### GPU vs. CPU Performance (Actual Measured Benchmarks)

VibeShift's `setup.bat` **automatically detects compatible NVIDIA GPUs (CUDA driver >= 12.1)** and installs GPU-accelerated PyTorch (`torch==2.10.0+cu126`), or installs CPU PyTorch automatically on laptops without an NVIDIA GPU.

| Runtime Mode | Model-Only Latency (Single-View) | Model-Only Latency (Horizontal-Flip) | Full 640×480 Frame-to-Emotion (YuNet + Preprocess + CleanBTNVit) | Memory Footprint | Expected Webcam Experience |
|---|---:|---:|---:|---|---|
| **NVIDIA GPU (FP16)** *(Measured on RTX 3050 Laptop GPU, CUDA 12.6)* | **22.53 ms** median (~44.4 FPS) | **42.63 ms** median (~23.5 FPS) | **45.61 ms** single / **69.25 ms** flip | **374–396 MiB VRAM** | Smooth real-time webcam scanning |
| **CPU Fallback (FP32)** *(Measured on 6 CPU threads via `VIBESHIFT_DEVICE=cpu`)* | **180.3 ms** median (~5.5 FPS) | **360.1 ms** median (~2.8 FPS) | **226.9 ms** single (~4.4 FPS) | **~1.0 GiB RAM** (`1,023.8 MiB` peak RSS) | **Usable for photo snapshots & ~3–4 FPS live polling** — slower than GPU, NOT 30 FPS real-time |

> **Tip:** To force CPU PyTorch installation even on an NVIDIA laptop (for smaller download/disk usage), run `set VIBESHIFT_FORCE_CPU=1` before `setup.bat`.

---

## 2. Why Dependencies Are Installed Online (Offline Bundle Feasibility)

You may wonder why the portable ZIP (`VibeShift-Team-Share.zip`, **under 1 MB**) does not bundle `.venv` and `node_modules` directly:
- A single Windows Python 3.12 CUDA PyTorch wheel alone is **`2.59 GB`**, plus **`~70 MB`** of runtime wheels (`opencv-python-headless`, `numpy`, `pandas`, `fastapi`, `uvicorn`) and **`~220 MB`** of `node_modules` (`289` packages).
- Virtual environments (`.venv`) contain absolute path launchers bound to the creator's Python directory and break when moved to another laptop.
- Supporting both CUDA and CPU across Python 3.10–3.12 offline would exceed **3.0+ GB** and still risk driver/ABI mismatches.
- Therefore, the portable ZIP ships clean, lightweight source code + the MIT-licensed YuNet ONNX model, while `setup.bat` builds a clean local `.venv` and `node_modules` natively on your laptop.

---

## 3. Important: Obtaining the AI Model Checkpoint (`clean_btn_vit_phase12.pt`)

VibeShift uses **exactly one facial-expression recognition model**: **CleanBTNVit Phase 12** (`245.3 MiB`, SHA-256 `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58`), paired with **YuNet** (`yunet-2023mar.onnx`, MIT-licensed, already included).

**Why `clean_btn_vit_phase12.pt` is NOT inside the public GitHub repo or ZIP:**
- Its upstream training lineage includes academic research datasets/backbones (**AffectNet**, **ExpW**, **RAF-DB**, and **MS-Celeb-1M**) whose license terms restrict use to **non-commercial academic research** and do not establish public or unapproved third-party redistribution rights.
- **Before transferring `clean_btn_vit_phase12.pt` directly to college teammates:**
  1. Confirm all teammates are working on the same **non-commercial academic/educational course project** (never for commercial use).
  2. Obtain confirmation from your **faculty advisor / Lead Researcher** that direct collaborator sharing within your institutional project team is permitted under your academic dataset scope (especially AffectNet's Lead Researcher / Permitted Researcher terms).
  3. Once authorized, share `clean_btn_vit_phase12.pt` privately (e.g., USB drive or restricted university OneDrive/Google Drive link accessible only to authorized team members — never a public URL).

> **Honest Note:** Without `clean_btn_vit_phase12.pt`, `setup.bat` completes as a **PARTIAL SETUP** (Manual Mood Mode, recommendations, Spotify playback, feedback, and Mood Journey work immediately, while camera AI emotion detection stays safely disabled until the verified checkpoint is installed).

---

## 4. Installation Instructions (Choose Method A OR Method B)

### METHOD A: Portable ZIP Installation (No Git Needed)

1. Copy **`VibeShift-Team-Share.zip`** onto your laptop (e.g., Desktop or `C:\Projects\`).
2. Right-click **`VibeShift-Team-Share.zip`** → **Extract All...**
   - Extract to a clean folder such as `C:\Projects\` (this creates `C:\Projects\VibeShift\`).
   - *Avoid extracting inside deeply nested OneDrive folders if you experience slow `npm` file syncing.*
3. Open the extracted **`VibeShift`** folder.
4. Double-click **`setup.bat`** (or open Command Prompt inside the folder and run `setup.bat`).
   - `setup.bat` automatically checks Windows 64-bit, Python 3.10–3.12, and Node.js 18+.
   - It creates `.venv`, auto-detects NVIDIA CUDA GPU (or installs CPU PyTorch), installs `requirements.txt`, and runs `npm ci` inside `frontend\`.
5. **When prompted for the CleanBTNVit checkpoint:**
   - **If you have the authorized `clean_btn_vit_phase12.pt` file:** drag and drop the file directly into the terminal window (or paste its full path or private authorized URL) and press **Enter**. `setup.bat` copies it into `backend\ai\models\clean_btn_vit_phase12.pt`, verifies its exact SHA-256, loads the PyTorch model, runs a 7-class inference test, and checks FastAPI `/api/health`.
   - **If you don't have the file yet:** press **Enter** on an empty line to skip for now. `setup.bat` will clearly display `PARTIAL SETUP: Core App Ready - AI MODEL NOT INSTALLED`.
6. *(Optional — Installing the checkpoint later)* Whenever you receive the authorized `clean_btn_vit_phase12.pt` file, run:
   ```cmd
   .venv\Scripts\python.exe scripts\install_model.py --from-file "C:\path\to\clean_btn_vit_phase12.pt" --check-load
   ```
   Or from a private authorized URL:
   ```cmd
   .venv\Scripts\python.exe scripts\install_model.py --url "https://your-private-authorized-link/clean_btn_vit_phase12.pt" --check-load
   ```

---

### METHOD B: Fresh GitHub Clone Installation

1. Open **Command Prompt** or **PowerShell** in the folder where you want the project (e.g., `C:\Projects`):
   ```powershell
   git clone https://github.com/Siddharth1968/vibe-shift.git
   cd vibe-shift
   ```
2. **One-step automated setup:**
   - **If you already have the authorized `clean_btn_vit_phase12.pt` locally (e.g., on USB `D:\clean_btn_vit_phase12.pt`):**
     ```powershell
     $env:VIBESHIFT_MODEL_PATH = "D:\clean_btn_vit_phase12.pt"
     .\setup.bat
     ```
   - **Or run interactive setup:**
     ```cmd
     setup.bat
     ```
     Drag and drop `clean_btn_vit_phase12.pt` when prompted (or press Enter to install it later with `scripts\install_model.py`).
3. Confirm the final banner prints:
   ```text
   SUCCESS: Full VibeShift AI installation verified!
   - YuNet face detector:     READY
   - CleanBTNVit Phase 12 AI: READY (SHA-256 verified)
   - FastAPI + SQLite + UI:   READY
   ```

---

## 5. Launching VibeShift (One Click)

1. Double-click **`start_vibeshift.bat`** inside the project folder.
2. What `start_vibeshift.bat` does automatically:
   - Verifies `.venv`, `frontend\node_modules`, `yunet-2023mar.onnx`, and `clean_btn_vit_phase12.pt` SHA-256.
   - Checks ports **`8000`** (FastAPI) and **`5173`** (Vite) — reuses already-running VibeShift servers or reports cleanly if another app occupies a port.
   - Opens two visible server log windows (**VibeShift Backend** and **VibeShift Frontend**).
   - Waits until `http://127.0.0.1:8000/api/health` confirms `detector_loaded: true` and `model_loaded: true`.
   - Waits until `http://localhost:5173` responds and opens VibeShift in your default browser.
3. **To stop VibeShift:** simply close the **VibeShift Backend** and **VibeShift Frontend** terminal windows (or press `Ctrl + C` in each window).

*(You can also launch servers manually using `run_backend.bat` and `run_frontend.bat`.)*

---

## 6. Enabling Webcam Permissions & Verifying Real AI Inference

### Allowing Browser Webcam Access
1. Open **`http://localhost:5173`** (always use `localhost` or `127.0.0.1` — browsers require a secure context for camera access).
2. Click **Start Listening** → choose **Camera Scan** → click **Start Live Camera**.
3. When Chrome or Microsoft Edge prompts **"localhost:5173 wants to use your camera"**, click **Allow**.
4. If the camera was previously blocked, click the **lock / camera icon** in the browser address bar → **Site settings** → set **Camera** to **Allow** → refresh the page.
5. Make sure no other app (Zoom, Teams, Windows Camera) is currently locking your webcam.

### How to Verify Real CleanBTNVit + YuNet AI Is Running
1. **Health Endpoint Check:** Open `http://127.0.0.1:8000/api/health` in your browser. You should see:
   ```json
   {
     "status": "ok",
     "ai": {
       "model": "CleanBTNVit (Phase 12)",
       "model_loaded": true,
       "detector_loaded": true,
       "default_inference_mode": "single_view"
     }
   }
   ```
   - `device` will read `"cuda"` on an NVIDIA GPU laptop or `"cpu"` on a CPU-only laptop.
2. **UI Verification:**
   - Confirm the **Backend Mode** pill in the navigation bar says **Live Mode** (not Demo Mode).
   - Inside the Camera panel, either start your webcam or click **Upload Photo** with a clear human face photo.
   - Toggle **Show Debug** inside the Camera panel to inspect:
     - Active model (`CleanBTNVit (Phase 12)`), inference mode (`single_view` or `horizontal_flip`), real inference latency in milliseconds, YuNet face detection box, and all 7 expression probabilities (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
   - Click **Confirm** on the detected mood, choose **Match My Mood** or **Change My Mood**, select Language/Genre/Activity, and verify Spotify recommendations, Like/Replay/Skip feedback, and your **Mood Journey** timeline.

---

## 7. Troubleshooting Common Issues

| Problem | Cause & Fix |
|---|---|
| **`Python 3.10, 3.11, or 3.12 is required`** | You either have no Python or only Python 3.13/3.14 installed. Install **Python 3.12.10 (64-bit)** from https://www.python.org/downloads/release/python-31210/ with **"Add python.exe to PATH"** checked, open a new terminal, and re-run `setup.bat`. |
| **`npm` blocked in PowerShell (`running scripts is disabled`)** | Use **Command Prompt** (`cmd.exe`) or double-click `setup.bat` / `start_vibeshift.bat` directly — `.bat` files bypass PowerShell's `npm.ps1` execution policy automatically. |
| **`Port 8000` or `Port 5173` already in use** | Another app is occupying that port. Find and close it (`netstat -ano \| findstr :8000`), or close any leftover VibeShift terminal windows, then run `start_vibeshift.bat` again. |
| **`SHA-256 mismatch` when installing checkpoint** | The file was corrupted during transfer or is the wrong checkpoint. Re-transfer `clean_btn_vit_phase12.pt` (`257,223,791` bytes) and verify SHA-256 equals `0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58`. |
| **Camera AI says `model_weights_not_loaded`** | `clean_btn_vit_phase12.pt` is not installed yet. Run `.venv\Scripts\python.exe scripts\install_model.py --from-file "PATH\TO\clean_btn_vit_phase12.pt" --check-load` and restart VibeShift. |
| **Webcam shows black video or permission denied** | Allow camera permission for `http://localhost:5173` in Chrome/Edge, check Windows **Settings → Privacy & security → Camera → Let desktop apps access your camera**, and close Teams/Zoom/Camera app. |
| **Slow webcam updates on CPU-only laptop** | Expected on CPU (`~227 ms` per 640×480 frame in single-view mode). Keep **Accuracy Mode (Horizontal Flip)** OFF (default single-view) or use **Upload Photo** for instant single-frame checks. |
| **Spotify only plays 30-second previews** | Expected behavior of Spotify Embeds when you are not signed into Spotify in the same browser. Log into `open.spotify.com` in your browser or click **Open in Spotify** on any track card. |
| **Windows Defender / SmartScreen warns about `.bat` file** | Click **More info → Run anyway** (standard Windows warning for downloaded batch scripts from ZIP archives). |
