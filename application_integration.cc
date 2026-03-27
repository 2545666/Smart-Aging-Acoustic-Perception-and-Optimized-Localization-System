/**
 * @file  application_integration.cc
 * @brief Integration guide — how to wire SER + D-S Fusion + Upstream
 *        into your existing Application class.
 * ========================================================================
 * This is NOT a standalone file — it shows the exact code changes needed
 * in your existing application.h and application.cc.
 * ========================================================================
 */

// =====================================================================
//  STEP 1: Add includes and members to application.h
// =====================================================================
//
// In application.h, add these includes near the top:
//
//   #include "speech_emotion.h"
//   #include "ds_fusion_engine.h"
//   #include "emotion_upstream.h"
//
// In the Application class private section, add these members:
//
//   SpeechEmotionAnalyser  ser_;               // Speech Emotion Recogniser
//   DSFusionEngine         fusion_engine_;      // D-S Fusion Engine
//   std::string            last_stt_text_;      // Last transcribed speech
//   std::array<float, kNumEmotions> last_vision_raw_ = {0};
//   esp_timer_handle_t     fusion_timer_ = nullptr;  // Periodic fusion timer
//
// Add a new event bit:
//
//   #define MAIN_EVENT_FUSION_TICK (1 << 7)
//


// =====================================================================
//  STEP 2: Initialise modules in Application::Start()
// =====================================================================
//
// Add this block AFTER uart_k210_.Init() and BEFORE StartReceiveTask():

void InitEmotionPipeline_Example() {
    // --- This code goes inside Application::Start() ---

    // ── 2a. Initialise Speech Emotion Recogniser ───────────────────
    SERConfig ser_cfg;
    ser_cfg.sample_rate = 16000;          // must match your AudioCodec
    ser_cfg.frame_size_ms = 30;
    // ser_cfg.tflite_model_path = "/spiffs/ser_model.tflite";  // future

    auto& ser_ = Application::GetInstance().ser_;   // pseudo-reference for illustration
    if (ser_.Init() != ESP_OK) {
        ESP_LOGE("App", "SER init failed!");
    }

    // ── 2b. Configure D-S Fusion Engine ────────────────────────────
    DSFusionConfig fusion_cfg;
    fusion_cfg.conflict_threshold    = 0.85f;
    fusion_cfg.vision_confidence     = 0.75f;
    fusion_cfg.audio_confidence      = 0.70f;
    fusion_cfg.vision_stale_timeout_ms = 2000;
    fusion_cfg.audio_stale_timeout_ms  = 1500;
    fusion_cfg.vision_reliability    = 0.6f;
    fusion_cfg.audio_reliability     = 0.4f;
    // fusion_engine_ = DSFusionEngine(fusion_cfg);  // re-initialise with config

    // ── 2c. Register Vision callback on UART ───────────────────────
    // This is the key wiring: when a JSON packet arrives from K210,
    // parse the emotion array and feed it into the fusion engine.

    auto& app = Application::GetInstance();
    app.GetUartK210().SetVisionCallback(
        [&app](const VisionEmotionPacket& pkt) {
            // Only feed valid, CRC-checked packets
            if (!pkt.crc_valid) {
                ESP_LOGW("App", "Dropping CRC-invalid vision packet seq=%u", pkt.seq);
                return;
            }

            // Store raw vision probs for upstream explainability
            // app.last_vision_raw_ = pkt.emo_probs;

            // Feed into fusion engine (thread-safe)
            app.GetDSFusionEngine().UpdateVision(
                pkt.emo_probs, pkt.face_detected);
        }
    );

    // ── 2d. Create a periodic fusion timer (1 Hz) ──────────────────
    // The fusion engine runs on a timer rather than being tied to either
    // sensor's update rate.  This decouples the two asynchronous streams.

    esp_timer_create_args_t fusion_timer_args = {
        .callback = [](void* arg) {
            Application* a = (Application*)arg;
            xEventGroupSetBits(a->event_group_, MAIN_EVENT_FUSION_TICK);
        },
        .arg = &app,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "fusion_timer",
        .skip_unhandled_events = true,
    };
    // esp_timer_create(&fusion_timer_args, &app.fusion_timer_);
    // esp_timer_start_periodic(app.fusion_timer_, 1000000);  // 1 second
    (void)fusion_timer_args;  // suppress unused warning in this example file

    ESP_LOGI("App", "Emotion pipeline initialised: SER + D-S Fusion + Upstream");
}


// =====================================================================
//  STEP 3: Feed audio into SER from the audio processing callback
// =====================================================================
//
// In your AudioService or wherever raw PCM frames are available,
// add a call to SER.  Example hook point:
//
//   void AudioService::OnAudioFrame(const int16_t* pcm, size_t samples) {
//       // Existing processing...
//
//       // Feed into Speech Emotion Recogniser
//       auto& app = Application::GetInstance();
//       app.ser_.FeedAudio(pcm, samples);
//
//       // Periodically update the fusion engine with audio emotion
//       if (app.ser_.IsReady()) {
//           auto audio_probs = app.ser_.GetEmotionProbs();
//           app.fusion_engine_.UpdateAudio(audio_probs);
//       }
//   }
//
// NOTE: SER.FeedAudio() is designed to be called from the audio task
// at whatever cadence frames arrive (typically 20-30 ms chunks).
// It internally accumulates and processes at its own frame rate.


// =====================================================================
//  STEP 4: Handle fusion ticks in MainEventLoop()
// =====================================================================
//
// Add this block inside Application::MainEventLoop(), alongside the
// existing event handlers:

void HandleFusionTick_Example() {
    // --- This code goes inside MainEventLoop's event handling ---

    auto& app = Application::GetInstance();

    // if (bits & MAIN_EVENT_FUSION_TICK) {

        // Run D-S fusion
        FusionResult result = app.GetDSFusionEngine().Fuse();

        // Log the result
        ESP_LOGI("App", "FUSED EMOTION: %s (%.1f%%) conflict=%.3f%s",
                 DSFusionEngine::EmotionLabel(result.dominant),
                 result.dominant_score * 100.0f,
                 result.conflict,
                 result.high_conflict ? " [HIGH CONFLICT — possible hidden depression]" : "");

        // ── Update the display with the fused emotion ──────────────
        // auto display = Board::GetInstance().GetDisplay();
        // display->SetEmotion(DSFusionEngine::EmotionLabel(result.dominant));

        // ── Build and send upstream payload to Raspberry Pi ────────
        auto vision_raw = app.last_vision_raw_;
        auto audio_raw  = app.ser_.GetEmotionProbs();

        std::string payload = BuildEmotionPayload(
            result,
            vision_raw,
            audio_raw,
            "",           // STT text — fill from protocol's STT callback
            "SIEVOX-01"
        );

        // Send via MQTT or WebSocket (use existing protocol infrastructure)
        // if (app.protocol_) {
        //     app.protocol_->SendCustomMessage(payload);
        // }

        ESP_LOGD("App", "Upstream payload: %s", payload.c_str());

        // ── Special handling: high-conflict alert ──────────────────
        // This is the "hidden depression" detection — our key innovation.
        // When face and voice disagree strongly, alert the caregiver.
        if (result.high_conflict) {
            ESP_LOGW("App", "⚠️  MULTIMODAL CONFLICT DETECTED — "
                     "possible masked emotional state. "
                     "Triggering empathetic LLM response.");
            // app.Alert("Emotion Alert",
            //           "Detected possible hidden distress — "
            //           "face and voice show conflicting emotions.",
            //           "heart_pulse", Lang::Sounds::OGG_VIBRATION);
        }
    // }
}


// =====================================================================
//  STEP 5: Reset SER on VAD transitions
// =====================================================================
//
// When VAD detects speech start, reset the SER accumulators:
//
//   // In the VAD callback or wherever speech onset is detected:
//   app.ser_.Reset();
//
// When VAD detects speech end, trigger a final fusion:
//
//   auto audio_probs = app.ser_.GetEmotionProbs();
//   app.fusion_engine_.UpdateAudio(audio_probs);
//   auto result = app.fusion_engine_.Fuse();
//   // Package and send result upstream


// =====================================================================
//  SUMMARY: Data flow
// =====================================================================
//
//  ┌─────────┐   JSON/UART    ┌──────────────────────────────────────┐
//  │  K210   │ ─────────────→ │  uart_k210.cc                       │
//  │  FER    │    emo probs   │  ParseEmotionPacket()                │
//  └─────────┘                │  → VisionCallback                   │
//                             │     → fusion_engine_.UpdateVision()  │
//                             └──────────┬───────────────────────────┘
//                                        │
//                                        ▼
//  ┌─────────┐   PCM frames   ┌──────────────────────────────────────┐
//  │  I2S    │ ─────────────→ │  speech_emotion.cc                   │
//  │  Mic    │                │  FeedAudio() → UpdateHeuristicProbs()│
//  └─────────┘                │  → fusion_engine_.UpdateAudio()      │
//                             └──────────┬───────────────────────────┘
//                                        │
//                                        ▼
//                             ┌──────────────────────────────────────┐
//                             │  ds_fusion_engine.cc                  │
//                             │  Fuse() → CombineBPAs()              │
//                             │  → FusionResult                      │
//                             └──────────┬───────────────────────────┘
//                                        │
//                                        ▼
//                             ┌──────────────────────────────────────┐
//                             │  emotion_upstream.h                   │
//                             │  BuildEmotionPayload()                │
//                             │  → JSON → MQTT/WS → Raspberry Pi     │
//                             │     → DeepSeek LLM                   │
//                             └──────────────────────────────────────┘
