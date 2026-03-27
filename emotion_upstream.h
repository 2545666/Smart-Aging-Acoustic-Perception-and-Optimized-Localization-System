/**
 * @file  emotion_upstream.h
 * @brief Package fused emotion results for upstream host (Raspberry Pi)
 * ========================================================================
 * Project : SIEVOX — Multimodal Elderly Care Robot
 * Target  : ESP32-S3 → MQTT/WebSocket → Raspberry Pi → DeepSeek LLM
 * ========================================================================
 *
 * This module serialises the D-S fusion result plus optional speech
 * intent text into a compact JSON payload suitable for transmission
 * over MQTT QoS 1 or WebSocket to the Raspberry Pi host, which will
 * inject it into the DeepSeek LLM prompt as emotional context.
 *
 * JSON schema sent to host:
 * {
 *   "type":    "emotion_state",
 *   "ts":      1719500000000,                     // epoch ms
 *   "emotion": {
 *     "dominant":  "sad",
 *     "score":     0.72,
 *     "belief":    [0.08, 0.72, 0.15, 0.05],     // [H, S, N, A]
 *     "conflict":  0.43,
 *     "high_conflict": false,
 *     "sources": {
 *       "vision": [0.60, 0.05, 0.10, 0.00],
 *       "audio":  [0.10, 0.56, 0.04, 0.00]
 *     }
 *   },
 *   "intent":  "我今天感觉还好",                  // STT text (if available)
 *   "device":  "SIEVOX-01"
 * }
 */

#ifndef EMOTION_UPSTREAM_H
#define EMOTION_UPSTREAM_H

#include <string>
#include <array>
#include <cJSON.h>
#include "ds_fusion_engine.h"

/**
 * @brief Build the upstream JSON payload string.
 *
 * Caller owns the returned string.  Uses cJSON (already in ESP-IDF).
 *
 * @param result       The fused emotion result from DSFusionEngine::Fuse()
 * @param vision_raw   Raw vision probabilities (for explainability)
 * @param audio_raw    Raw audio probabilities  (for explainability)
 * @param intent_text  User's transcribed speech (from STT), or "" if none
 * @param device_id    Unique device identifier string
 * @return             Heap-allocated JSON string; caller must free() it.
 */
inline std::string BuildEmotionPayload(
    const FusionResult& result,
    const std::array<float, kNumEmotions>& vision_raw,
    const std::array<float, kNumEmotions>& audio_raw,
    const std::string& intent_text = "",
    const std::string& device_id   = "SIEVOX-01")
{
    cJSON* root = cJSON_CreateObject();
    if (!root) return "{}";

    // ── Top-level metadata ─────────────────────────────────────────
    cJSON_AddStringToObject(root, "type", "emotion_state");
    cJSON_AddNumberToObject(root, "ts", (double)result.timestamp_ms);

    // ── Emotion sub-object ─────────────────────────────────────────
    cJSON* emo = cJSON_AddObjectToObject(root, "emotion");
    cJSON_AddStringToObject(emo, "dominant",
                            DSFusionEngine::EmotionLabel(result.dominant));
    cJSON_AddNumberToObject(emo, "score", result.dominant_score);

    // Belief array [happy, sad, neutral, anger]
    cJSON* belief_arr = cJSON_AddArrayToObject(emo, "belief");
    for (size_t i = 0; i < kNumEmotions; i++) {
        cJSON_AddItemToArray(belief_arr,
            cJSON_CreateNumber(result.belief[i]));
    }

    cJSON_AddNumberToObject(emo, "conflict", result.conflict);
    cJSON_AddBoolToObject(emo, "high_conflict", result.high_conflict);

    // Source-level probabilities (for explainability / logging)
    cJSON* sources = cJSON_AddObjectToObject(emo, "sources");

    cJSON* v_arr = cJSON_AddArrayToObject(sources, "vision");
    for (size_t i = 0; i < kNumEmotions; i++) {
        cJSON_AddItemToArray(v_arr, cJSON_CreateNumber(vision_raw[i]));
    }

    cJSON* a_arr = cJSON_AddArrayToObject(sources, "audio");
    for (size_t i = 0; i < kNumEmotions; i++) {
        cJSON_AddItemToArray(a_arr, cJSON_CreateNumber(audio_raw[i]));
    }

    // ── Intent text (STT output) ──────────────────────────────────
    if (!intent_text.empty()) {
        cJSON_AddStringToObject(root, "intent", intent_text.c_str());
    }

    // ── Device ID ─────────────────────────────────────────────────
    cJSON_AddStringToObject(root, "device", device_id.c_str());

    // ── Serialise ─────────────────────────────────────────────────
    char* json_str = cJSON_PrintUnformatted(root);
    std::string payload = json_str ? json_str : "{}";
    cJSON_free(json_str);
    cJSON_Delete(root);

    return payload;
}

#endif // EMOTION_UPSTREAM_H
