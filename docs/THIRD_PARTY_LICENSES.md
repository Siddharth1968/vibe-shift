# Third-Party Licenses, Code Attribution, and Data Restrictions

> **Important Distinction:** This repository intentionally does **not** attach a blanket permissive `LICENSE` file to the entire project because VibeShift integrates third-party MIT-licensed components alongside an optional, non-redistributed AI checkpoint trained with academic-only datasets. Please respect each component's specific terms below.

---

## 1. Included Third-Party MIT-Licensed Components

### A. Batch Transformer Network (`backend/ai/btn_runtime/`)
Adapted from [SeoulTech-HCIRLab/BTN](https://github.com/SeoulTech-HCIRLab/BTN):

```text
MIT License

Copyright (c) 2025 SeoulTech-HCIRLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### B. YuNet ONNX Face Detector (`backend/ai/models/yunet-2023mar.onnx`)
From OpenCV Zoo / Shiqi Yu (`face_detection_yunet_2023mar.onnx`):

```text
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Academic Dataset & Pretrained Weight Acknowledgments (Not Included in Repo)

The optional Phase 12 `CleanBTNVit` checkpoint (`clean_btn_vit_phase12.pt`, excluded from public Git distribution) was trained and evaluated in an academic research setting using the following upstream datasets/pretrained artifacts:

1. **FER-2013:** Goodfellow et al., *Challenges in Representation Learning: A Report on Three Machine Learning Contests* (ICML 2013 Workshop).
2. **ExpW (Expression in-the-Wild):** Zhang, Z., Luo, P., Loy, C. C., & Tang, X., *From Facial Expression Recognition to Interpersonal Relation Prediction*, IJCV / CVPR 2017 (`arXiv:1609.06426`). **Restricted to non-commercial academic and educational research.**
3. **AffectNet:** Mollahosseini, A., Hasani, B., & Mahoor, M. H., *AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild*, IEEE Transactions on Affective Computing, 2017. **Restricted to non-commercial academic research under University of Denver EULA.**
4. **RAF-DB:** Li, S., Deng, W., & Du, J., *Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild*, CVPR 2017. **Restricted to non-commercial academic research.**
5. **MS-Celeb-1M / IR-50 & MobileFaceNet:** Upstream face representation backbones used inside the BTN architecture.

---

## 3. Spotify Embed Integration

VibeShift stores only public track metadata and canonical `https://open.spotify.com/track/<id>` URLs in `backend/data/songs.csv`. No audio files are hosted or redistributed. Track playback uses Spotify's official public Embed IFrame API and direct links to Spotify, subject to Spotify's Developer Terms of Service.
