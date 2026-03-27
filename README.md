# SIEVOX — Multimodal Emotion Recognition: Implementation Architecture

## System Overview

This document covers the complete upgrade from your existing audio-tracking gimbal system to a multimodal emotion recognition platform using Dempster-Shafer evidence fusion.

```
┌─────────────────────┐            ┌──────────────────────────────────────┐
│   K210 / MaixCAM    │   UART     │         ESP32-S3 Main Controller     │
│                     │  115200    │                                      │
│  Camera → FER       │ ─────────→│  uart_k210.cc (JSON parse + CRC)     │
│  fer_engine.py      │  JSON     │           │                           │
│  main.py            │  packets  │           ▼                           │
│  uart_comm.py       │           │  ┌─────────────────────┐             │
│                     │           │  │ DSFusionEngine       │             │
│  Emotion probs:     │ ←─────── │  │ (Dempster-Shafer)    │             │
│  [H, S, N, A]       │  CMD     │  │                      │             │
└─────────────────────┘  lines   │  └──────┬──────────────┘             │
                                  │         │                             │
┌─────────────────────┐           │  ┌──────┴──────────────┐             │
│   I2S Microphone    │  PCM     │  │ SpeechEmotionAnalyser│             │
│   (onboard)         │ ────────→│  │ (Pitch/RMS/MFCC)    │             │
│                     │  16kHz   │  │                      │             │
└─────────────────────┘           │  └─────────────────────┘             │
                                  │         │                             │
                                  │         ▼                             │
                                  │  BuildEmotionPayload()                │
                                  │         │                             │
                                  │    MQTT / WebSocket                   │
                                  └─────────┼─────────────────────────────┘
                                            │
                                            ▼
                                  ┌─────────────────────┐
                                  │   Raspberry Pi Host  │
                                  │   DeepSeek LLM       │
                                  │   → Empathetic Reply  │
                                  └─────────────────────┘
```

---

## File Manifest

| Task | File | Language | Description |
|------|------|----------|-------------|
| 1 | `task1_vision/fer_engine.py` | Python (MaixPy) | Face detection + expression classifier |
| 1 | `task1_vision/uart_comm.py` | Python (MaixPy) | JSON telemetry with CRC-8 + seq numbers |
| 1 | `task1_vision/main.py` | Python (MaixPy) | Upgraded main loop with FER integration |
| 2 | `task2_audio/speech_emotion.h` | C++ (ESP-IDF) | SER feature extractor header |
| 2 | `task2_audio/speech_emotion.cc` | C++ (ESP-IDF) | Pitch/RMS/MFCC extraction + heuristic mapping |
| 3 | `task3_fusion/ds_fusion_engine.h` | C++ (ESP-IDF) | D-S fusion engine header |
| 3 | `task3_fusion/ds_fusion_engine.cc` | C++ (ESP-IDF) | Dempster's Rule + conflict handling |
| 3 | `task3_fusion/uart_k210.h` | C++ (ESP-IDF) | Upgraded UART driver header |
| 3 | `task3_fusion/uart_k210.cc` | C++ (ESP-IDF) | JSON parsing + packet drop detection |
| 4 | `task4_upstream/emotion_upstream.h` | C++ (ESP-IDF) | JSON payload builder for Raspberry Pi |
| 4 | `task4_upstream/application_integration.cc` | C++ (ESP-IDF) | Wiring guide for application.cc |

---

## Task 1: Vision Module — Design Notes

**Model selection for K210/MaixCAM:**
The K210's KPU supports up to two concurrent INT8 models. We use a two-stage pipeline: a YOLO-based face detector (~200 KB) finds the bounding box, then a MobileNetV2-0.35 classifier (~400 KB) maps the cropped face to 4 emotion classes.

**FER cadence vs gimbal control:**
Running KPU inference every frame would starve the servo control loop of CPU time. The `fer_every_n` config parameter (default 3) throttles FER to every 3rd iteration, giving ~6–8 FPS for emotion updates while maintaining smooth gimbal tracking at ~20 FPS.

**UART protocol design:**
Each K210→ESP32 line is a self-contained JSON object with a monotonic sequence number and CRC-8. This design choice means that if the ESP32 drops packets (UART buffer overflow, busy processing), it can detect the gap from the sequence number without needing an ACK/NACK round-trip. The CRC guards against bit-flip corruption from EMI on the gimbal motor power rails.

---

## Task 2: Audio Module — Design Notes

**Why not a neural model?**
A full SER model (e.g., wav2vec2 → linear classifier) requires ~50 MB of weights and 100+ MFLOPS per frame — far beyond ESP32-S3's capabilities. Instead, we extract three well-established prosodic features that correlate strongly with basic emotions.

**Feature rationale:**

- **Pitch (F0):** Estimated via autocorrelation. Elderly speakers typically have lower baseline pitch. We set `pitch_min_hz=60` to accommodate age-related vocal changes.
- **Energy (RMS):** Simple but effective — depression and sadness correlate with reduced vocal energy.
- **MFCC (optional TFLite path):** 13 coefficients capture spectral envelope characteristics. When a trained INT8 TFLite Micro model is available (future work), it can replace the heuristic mapper.

**Memory budget:**
The SER module uses approximately 54 KB of heap memory (ring buffer + FFT + Mel filters), leaving ample room for WiFi, AudioService, and the fusion engine.

---

## Task 3: D-S Fusion Engine — Mathematical Foundation

### Why Dempster-Shafer over Bayesian Fusion?

Bayesian fusion requires known prior distributions and assumes independence between sources. D-S theory offers two critical advantages for our use case:

1. **Explicit uncertainty modelling.** When the K210 doesn't detect a face, we can assign m(Θ) = 0.95 (95% uncertainty) rather than fabricating a uniform probability. The fusion automatically lets the Audio modality dominate.

2. **Conflict quantification.** The K factor directly measures disagreement between face and voice — the exact signal we need for "hidden depression" detection.

### Dempster's Rule of Combination

Given two BPAs m₁ (Vision) and m₂ (Audio) over Θ = {H, S, N, A}:

```
m₁₂(A) = [Σ_{B∩C=A} m₁(B)·m₂(C)] / (1 - K)

where K = Σ_{B∩C=∅} m₁(B)·m₂(C)
```

For our restricted focal-element set (singletons + Θ):

```
raw(θᵢ) = m₁(θᵢ)·m₂(θᵢ) + m₁(θᵢ)·m₂(Θ) + m₁(Θ)·m₂(θᵢ)
raw(Θ)  = m₁(Θ)·m₂(Θ)
K       = Σ_{i≠j} m₁(θᵢ)·m₂(θⱼ)
```

### Conflict Handling (Zadeh's Paradox Protection)

When K exceeds 0.85, standard Dempster's Rule can produce counter-intuitive results. For example, if the face confidently says "happy" and the voice confidently says "sad", normalising by (1-K) when K≈0.9 amplifies tiny residual masses unpredictably.

Our solution: when K ≥ threshold, we fall back to a reliability-weighted average. More importantly, we flag `high_conflict = true` in the result — because in the elderly care context, this conflict IS the clinically meaningful signal. A person smiling while their voice trembles is the textbook presentation of masked depression.

### BPA Construction from Sensor Probabilities

Each sensor's raw probability array [p₁, p₂, p₃, p₄] is converted to a BPA using a "simple support function" construction:

```
m(θᵢ) = pᵢ × confidence
m(Θ)   = 1 - confidence
```

The confidence parameter (0.75 for vision, 0.70 for audio by default) controls how much influence each sensor has. When no face is detected, vision confidence drops to 0.05, effectively making the audio modality the sole information source.

---

## Task 4: Upstream Integration — JSON Schema

The ESP32 packages the fused result into this JSON structure for the Raspberry Pi:

```json
{
  "type":    "emotion_state",
  "ts":      1719500000000,
  "emotion": {
    "dominant":      "sad",
    "score":         0.72,
    "belief":        [0.08, 0.72, 0.15, 0.05],
    "conflict":      0.43,
    "high_conflict": false,
    "sources": {
      "vision": [0.60, 0.05, 0.10, 0.00],
      "audio":  [0.10, 0.56, 0.04, 0.00]
    }
  },
  "intent":  "我今天感觉还好",
  "device":  "SIEVOX-01"
}
```

The `sources` sub-object preserves both raw inputs for explainability — the Raspberry Pi can log these for caregiver review and model tuning.

---

## Integration Checklist

1. **K210 side:** Copy `fer_engine.py`, `uart_comm.py`, `main.py` to SD card. Place `face_detect.kmodel` and `fer_mobilenet.kmodel` in `/sd/models/`.

2. **ESP32 CMakeLists.txt:** Add `speech_emotion.cc`, `ds_fusion_engine.cc`, and the updated `uart_k210.cc` to your component's `SRCS` list.

3. **application.h:** Add member variables (SER, fusion engine, fusion timer) per `application_integration.cc` Step 1.

4. **application.cc Start():** Wire the vision callback and initialise modules per Step 2.

5. **AudioService:** Hook SER.FeedAudio() into the PCM processing pipeline per Step 3.

6. **MainEventLoop:** Handle MAIN_EVENT_FUSION_TICK per Step 4.

7. **VAD transitions:** Reset SER on speech onset; trigger fusion on speech end per Step 5.

---

## Future Improvements

- **TFLite Micro SER model:** Train a small 3-layer MLP on MFCC features using RAVDESS/IEMOCAP datasets, quantise to INT8, deploy via SPIFFS.
- **Temporal smoothing:** Apply exponential moving average to fusion results to avoid emotion "flickering."
- **Additional emotions:** Extend Θ to include {fear, surprise, disgust} by expanding the BPA arrays (change `kNumEmotions` and retrain models).
- **MaixCAM upgrade:** The MaixCAM's RISC-V core can run larger FER models (e.g., MobileFaceNet) for improved accuracy.
