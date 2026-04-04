# 🛡️ VoiceGuard Pro
**Real-Time AI Deepfake Voice Scam Detector** | v7.0 | Python + Streamlit

---

## What It Does
Detects AI-generated/cloned voices in audio calls using a 5-layer detection pipeline — ML model, acoustic forensics, heuristic scoring, reference comparison, and keyword analysis.

---

## Features
- 193-feature extraction (MFCCs, delta, delta-delta, HNR, spectral contrast, chroma)
- Segment-level ML voting (GradientBoosting / RandomForest)
- 20+ acoustic forensic metrics with per-indicator flagging
- 60+ fraud keyword scanner with danger scoring
- In-app one-click model training with heavy augmentation
- Supports WAV, MP3, FLAC, OGG, M4A, WebM

---

## Project Structure
VoiceGuardPro/
├── app.py
├── train_model.py
├── calibrate_demo.py
├── generate_dataset.py
├── download_real_voices.py
├── requirements.txt
├── voiceguard_model.pkl
└── demo_audio/
├── fake_scam.wav
└── real.wav

---

## Installation
```bash
git clone https://github.com/aryanmalhotraug25-hash/VoiceGuardPro.git
cd VoiceGuardPro
pip install -r requirements.txt
streamlit run app.py
```
> Optional: Install FFmpeg for broader audio format support.

---

## Usage
1. Add `fake_scam.wav` and `real.wav` to the `demo_audio/` folder
2. Click **⚡ Train Model from Demo Files** in the sidebar
3. Restart the app to load the new model
4. Use **🔴 Deepfake Call**, **🟢 Real Call**, or **🎤 Judge Test** to analyze audio

**Judge Test tip:** Upload AI audio files directly (ElevenLabs, Murf, etc.) — don't play and re-record.

---

## How Detection Works
AI voices are *too perfect* — they lack natural jitter, shimmer, breath pauses, and energy variation. VoiceGuard Pro detects the **absence of human imperfections**.

| Metric | Suspicious Range | Weight |
|---|---|---|
| Pitch Jitter | < 0.008 (too smooth) | 18 |
| Amplitude Shimmer | < 0.06 (flat) | 14 |
| HNR (Harmonic/Noise) | > 6.0 (too clean) | 14 |
| MFCC Variance | < 18 (uniform) | 12 |
| Silence Ratio | < 0.03 (no breathing) | 8 |

**Scoring:**

Danger Score = (AI_Probability × 0.30) + Keyword_Score
CRITICAL     → keyword_score ≥ 40
SUSPICIOUS   → keyword_score ≥ 16 or danger_score ≥ 40

---

## Dependencies
streamlit>=1.28.0
librosa>=0.10.0
scikit-learn>=1.3.0
joblib>=1.3.0
numpy>=1.24.0
soundfile>=0.12.0
SpeechRecognition>=3.10.0

---

## Limitations
- Model trained on 2 demo files — larger dataset improves accuracy
- Short clips (< 1s) may give unreliable results
- Heavy audio compression can reduce accuracy

---

*VoiceGuard Pro v7.0 — 🛡️ Protecting voices from deepfake scams.*

