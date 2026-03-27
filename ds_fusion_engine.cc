/**
 * @file  ds_fusion_engine.cc
 * @brief Dempster-Shafer Evidence Theory — Implementation
 * ========================================================================
 * Project : SIEVOX — Multimodal Elderly Care Robot
 * ========================================================================
 *
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │                   D-S COMBINATION — WORKED EXAMPLE                  │
 * ├──────────────────────────────────────────────────────────────────────┤
 * │                                                                      │
 * │  Frame of discernment Θ = {H, S, N, A}  (Happy, Sad, Neutral, Anger)│
 * │                                                                      │
 * │  Vision BPA  m₁:                                                     │
 * │      m₁({H}) = 0.60   (face is smiling)                             │
 * │      m₁({S}) = 0.05                                                 │
 * │      m₁({N}) = 0.10                                                 │
 * │      m₁({A}) = 0.00                                                 │
 * │      m₁(Θ)   = 0.25   (25% uncertainty / low confidence)            │
 * │                                                                      │
 * │  Audio BPA  m₂:                                                      │
 * │      m₂({H}) = 0.10                                                 │
 * │      m₂({S}) = 0.56   (voice is trembling / low pitch)              │
 * │      m₂({N}) = 0.04                                                 │
 * │      m₂({A}) = 0.00                                                 │
 * │      m₂(Θ)   = 0.30                                                 │
 * │                                                                      │
 * │  Step 1 — Compute pairwise products m₁(B)·m₂(C) for all B,C:       │
 * │                                                                      │
 * │           m₂({H})  m₂({S})  m₂({N})  m₂({A})  m₂(Θ)               │
 * │  m₁({H})   0.060    0.336↯   0.024↯   0.000    0.180               │
 * │  m₁({S})   0.005↯   0.028    0.002↯   0.000    0.015               │
 * │  m₁({N})   0.010↯   0.056↯   0.004    0.000    0.030               │
 * │  m₁({A})   0.000    0.000    0.000    0.000    0.000               │
 * │  m₁(Θ)     0.025    0.070    0.010    0.000    0.075               │
 * │                                                                      │
 * │  ↯ = conflicting pair (B∩C = ∅)                                     │
 * │                                                                      │
 * │  K = sum of all ↯ cells = 0.336+0.024+0.005+0.002+0.010+0.056      │
 * │    = 0.433                                                           │
 * │                                                                      │
 * │  Step 2 — Sum agreeing masses for each singleton:                   │
 * │    raw({H}) = 0.060 + 0.180 + 0.025 = 0.265                        │
 * │    raw({S}) = 0.028 + 0.015 + 0.070 = 0.113                        │
 * │    raw({N}) = 0.004 + 0.030 + 0.010 = 0.044                        │
 * │    raw({A}) = 0.000                                                 │
 * │    raw(Θ)   = 0.075                                                 │
 * │                                                                      │
 * │  Step 3 — Normalise by (1 - K):                                     │
 * │    m₁₂({H}) = 0.265 / 0.567 = 0.467                                │
 * │    m₁₂({S}) = 0.113 / 0.567 = 0.199                                │
 * │    m₁₂({N}) = 0.044 / 0.567 = 0.078                                │
 * │    m₁₂({A}) = 0.000 / 0.567 = 0.000                                │
 * │    m₁₂(Θ)   = 0.075 / 0.567 = 0.132                                │
 * │                                                                      │
 * │  → Despite the face smiling (H=0.60) and voice suggesting sadness   │
 * │    (S=0.56), the fused belief in Happiness (0.467) outweighs Sad    │
 * │    (0.199) because the vision module was more confident overall.     │
 * │    K=0.433 is moderate — the system trusts the fusion.              │
 * │                                                                      │
 * │  → In the "hidden depression" scenario (elderly person masks with   │
 * │    a smile but voice trembles), a high K value (>0.85) triggers     │
 * │    our special handling: the system falls back to weighted average   │
 * │    and flags the *conflict itself* as clinically significant.       │
 * └──────────────────────────────────────────────────────────────────────┘
 */

#include "ds_fusion_engine.h"

#include <algorithm>
#include <cmath>
#include <esp_log.h>
#include <esp_timer.h>

#define TAG "DS_FUSION"

// Emotion label strings (matches EmotionIndex order)
static const char* const EMOTION_LABELS[] = {
    "happy", "sad", "neutral", "anger"
};


// =====================================================================
//  BPA factory
// =====================================================================

BPA BPA::FromProbArray(const std::array<float, kNumEmotions>& probs,
                       float confidence) {
    /**
     * Convert a probability array into a BPA by scaling each
     * singleton mass by a confidence factor ∈ [0, 1].
     *
     * The remaining mass (1 - confidence) is assigned to Θ,
     * representing "I'm not sure."
     *
     * This is the standard "simple support function" construction
     * commonly used in D-S applications.
     */
    BPA bpa;
    confidence = std::clamp(confidence, 0.0f, 1.0f);

    float total_prob = 0.0f;
    for (size_t i = 0; i < kNumEmotions; i++) {
        total_prob += probs[i];
    }

    // Guard against degenerate input
    if (total_prob < 1e-6f) {
        bpa.uncertainty = 1.0f;
        return bpa;
    }

    // Normalise probabilities (in case they don't sum to exactly 1)
    float scale = confidence / total_prob;
    for (size_t i = 0; i < kNumEmotions; i++) {
        bpa.singletons[i] = probs[i] * scale;
    }
    bpa.uncertainty = 1.0f - confidence;

    return bpa;
}


// =====================================================================
//  Constructor
// =====================================================================

DSFusionEngine::DSFusionEngine(const DSFusionConfig& cfg)
    : cfg_(cfg),
      vision_updated_ms_(0),
      audio_updated_ms_(0)
{
    // Start with maximum-uncertainty BPAs (no information)
    vision_bpa_.uncertainty = 1.0f;
    audio_bpa_.uncertainty  = 1.0f;

    // Default result: neutral with zero confidence
    last_result_.belief    = {0.0f, 0.0f, 1.0f, 0.0f};
    last_result_.conflict  = 0.0f;
    last_result_.dominant  = kEmotionNeutral;
    last_result_.dominant_score = 0.0f;
    last_result_.high_conflict  = false;
    last_result_.timestamp_ms   = 0;

    ESP_LOGI(TAG, "D-S Fusion Engine created (conflict_thresh=%.2f, "
             "vision_conf=%.2f, audio_conf=%.2f)",
             cfg_.conflict_threshold, cfg_.vision_confidence,
             cfg_.audio_confidence);
}


// =====================================================================
//  Update modalities
// =====================================================================

void DSFusionEngine::UpdateVision(
    const std::array<float, kNumEmotions>& probs,
    bool face_detected)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // If no face was detected, drastically reduce confidence
    // so that the Audio modality dominates the fusion.
    float conf = face_detected ? cfg_.vision_confidence : 0.05f;

    vision_bpa_ = BPA::FromProbArray(probs, conf);
    vision_updated_ms_ = NowMs();

    ESP_LOGD(TAG, "Vision BPA updated: H=%.3f S=%.3f N=%.3f A=%.3f Θ=%.3f (face=%d)",
             vision_bpa_.singletons[0], vision_bpa_.singletons[1],
             vision_bpa_.singletons[2], vision_bpa_.singletons[3],
             vision_bpa_.uncertainty, face_detected);
}

void DSFusionEngine::UpdateAudio(const std::array<float, kNumEmotions>& probs) {
    std::lock_guard<std::mutex> lock(mutex_);

    audio_bpa_ = BPA::FromProbArray(probs, cfg_.audio_confidence);
    audio_updated_ms_ = NowMs();

    ESP_LOGD(TAG, "Audio BPA updated: H=%.3f S=%.3f N=%.3f A=%.3f Θ=%.3f",
             audio_bpa_.singletons[0], audio_bpa_.singletons[1],
             audio_bpa_.singletons[2], audio_bpa_.singletons[3],
             audio_bpa_.uncertainty);
}


// =====================================================================
//  Fuse — main entry point
// =====================================================================

FusionResult DSFusionEngine::Fuse() {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t now = NowMs();

    // ── Handle stale data ──────────────────────────────────────────
    // If a modality hasn't been updated within its timeout window,
    // replace its BPA with total uncertainty (m(Θ) = 1).
    // This means a stale sensor effectively contributes nothing.

    BPA v_bpa = vision_bpa_;
    BPA a_bpa = audio_bpa_;

    if (vision_updated_ms_ == 0 ||
        (now - vision_updated_ms_) > cfg_.vision_stale_timeout_ms) {
        v_bpa = BPA{};   // m(Θ) = 1
        ESP_LOGD(TAG, "Vision BPA is stale — using max uncertainty");
    }
    if (audio_updated_ms_ == 0 ||
        (now - audio_updated_ms_) > cfg_.audio_stale_timeout_ms) {
        a_bpa = BPA{};
        ESP_LOGD(TAG, "Audio BPA is stale — using max uncertainty");
    }

    // ── Combine ────────────────────────────────────────────────────
    FusionResult result = CombineBPAs(v_bpa, a_bpa);
    result.timestamp_ms = now;

    last_result_ = result;
    return result;
}

FusionResult DSFusionEngine::GetLastResult() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_result_;
}


// =====================================================================
//  Core D-S combination (Dempster's Rule)
// =====================================================================

FusionResult DSFusionEngine::CombineBPAs(const BPA& m1, const BPA& m2) const {
    /**
     * Implements the orthogonal sum (Dempster's Rule of Combination)
     * for the restricted case where focal elements are either singletons
     * or the full frame Θ.
     *
     * For two BPAs m₁, m₂ with focal elements {θᵢ} and Θ:
     *
     *   raw(θᵢ) = m₁(θᵢ)·m₂(θᵢ)          // both agree on θᵢ
     *           + m₁(θᵢ)·m₂(Θ)            // m₁ specific, m₂ uncertain
     *           + m₁(Θ)·m₂(θᵢ)            // m₂ specific, m₁ uncertain
     *
     *   raw(Θ)  = m₁(Θ)·m₂(Θ)             // both uncertain
     *
     *   K       = Σ_{i≠j} m₁(θᵢ)·m₂(θⱼ)  // conflict: different singletons
     *
     *   m₁₂(A) = raw(A) / (1 - K)         // normalisation
     *
     * Complexity: O(N²) where N = |Θ| = 4 → 16 multiplications.
     */

    FusionResult result;

    // ── Step 1: Compute conflict K ─────────────────────────────────
    float K = 0.0f;
    for (size_t i = 0; i < kNumEmotions; i++) {
        for (size_t j = 0; j < kNumEmotions; j++) {
            if (i != j) {
                K += m1.singletons[i] * m2.singletons[j];
            }
        }
    }

    result.conflict = K;
    result.high_conflict = (K >= cfg_.conflict_threshold);

    // ── High-conflict guard (Zadeh's paradox protection) ───────────
    if (result.high_conflict) {
        ESP_LOGW(TAG, "High conflict detected: K=%.4f ≥ %.2f — "
                 "falling back to weighted average", K, cfg_.conflict_threshold);
        return FallbackAverage(m1, m2);
    }

    // ── Step 2: Compute raw combined masses ────────────────────────
    float norm = 1.0f - K;
    if (norm < 1e-8f) {
        // Nearly total conflict — should have been caught above
        ESP_LOGE(TAG, "Normalisation factor near zero (K=%.6f)", K);
        return FallbackAverage(m1, m2);
    }

    float inv_norm = 1.0f / norm;
    float belief_sum = 0.0f;

    for (size_t i = 0; i < kNumEmotions; i++) {
        float raw_i = m1.singletons[i] * m2.singletons[i]   // agree on θᵢ
                    + m1.singletons[i] * m2.uncertainty       // m₁ specific
                    + m1.uncertainty   * m2.singletons[i];    // m₂ specific
        result.belief[i] = raw_i * inv_norm;
        belief_sum += result.belief[i];
    }

    // Residual uncertainty = m₁(Θ)·m₂(Θ) / (1-K)
    float fused_uncertainty = (m1.uncertainty * m2.uncertainty) * inv_norm;

    // ── Step 3: Distribute residual uncertainty proportionally ──────
    // (Optional: converts Plausibility-like output to a cleaner distribution
    //  that sums to 1.0 for downstream consumers like the LLM prompt.)
    if (belief_sum > 1e-6f && fused_uncertainty > 1e-6f) {
        for (size_t i = 0; i < kNumEmotions; i++) {
            result.belief[i] += fused_uncertainty * (result.belief[i] / belief_sum);
        }
    } else if (fused_uncertainty > 0.5f) {
        // Both sensors are highly uncertain — distribute equally
        for (size_t i = 0; i < kNumEmotions; i++) {
            result.belief[i] = 1.0f / kNumEmotions;
        }
    }

    // ── Final normalisation to exactly 1.0 ─────────────────────────
    float total = 0.0f;
    for (size_t i = 0; i < kNumEmotions; i++) total += result.belief[i];
    if (total > 0) {
        for (size_t i = 0; i < kNumEmotions; i++) result.belief[i] /= total;
    }

    // ── Find dominant emotion ──────────────────────────────────────
    result.dominant = kEmotionNeutral;
    result.dominant_score = 0.0f;
    for (size_t i = 0; i < kNumEmotions; i++) {
        if (result.belief[i] > result.dominant_score) {
            result.dominant_score = result.belief[i];
            result.dominant = static_cast<EmotionIndex>(i);
        }
    }

    ESP_LOGI(TAG, "Fused: H=%.3f S=%.3f N=%.3f A=%.3f | K=%.3f → %s (%.1f%%)",
             result.belief[0], result.belief[1],
             result.belief[2], result.belief[3],
             K, EMOTION_LABELS[result.dominant],
             result.dominant_score * 100.0f);

    return result;
}


// =====================================================================
//  Fallback: weighted average for high-conflict scenarios
// =====================================================================

FusionResult DSFusionEngine::FallbackAverage(const BPA& vision,
                                              const BPA& audio) const {
    /**
     * When conflict is too high, Dempster's Rule can produce
     * counter-intuitive results (Zadeh's paradox).
     *
     * Instead, we compute a simple reliability-weighted average of
     * the two BPA singleton masses.  This is more conservative but
     * always produces a sensible output.
     *
     * Importantly, the high-conflict flag itself carries clinical
     * information: "face says happy, voice says sad" is a strong
     * indicator of masked/hidden depression — our core use case.
     */

    FusionResult result;
    result.high_conflict = true;

    float w_v = cfg_.vision_reliability;
    float w_a = cfg_.audio_reliability;
    float w_total = w_v + w_a;
    if (w_total < 1e-6f) w_total = 1.0f;

    float total = 0.0f;
    for (size_t i = 0; i < kNumEmotions; i++) {
        result.belief[i] = (vision.singletons[i] * w_v +
                            audio.singletons[i]  * w_a) / w_total;
        total += result.belief[i];
    }

    // Normalise to account for uncertainty mass that was discarded
    if (total > 0) {
        for (size_t i = 0; i < kNumEmotions; i++) {
            result.belief[i] /= total;
        }
    } else {
        // Both fully uncertain
        for (size_t i = 0; i < kNumEmotions; i++) {
            result.belief[i] = 1.0f / kNumEmotions;
        }
    }

    // Recompute conflict from original BPAs
    float K = 0.0f;
    for (size_t i = 0; i < kNumEmotions; i++) {
        for (size_t j = 0; j < kNumEmotions; j++) {
            if (i != j) K += vision.singletons[i] * audio.singletons[j];
        }
    }
    result.conflict = K;

    // Find dominant
    result.dominant = kEmotionNeutral;
    result.dominant_score = 0.0f;
    for (size_t i = 0; i < kNumEmotions; i++) {
        if (result.belief[i] > result.dominant_score) {
            result.dominant_score = result.belief[i];
            result.dominant = static_cast<EmotionIndex>(i);
        }
    }

    ESP_LOGW(TAG, "Fallback avg: H=%.3f S=%.3f N=%.3f A=%.3f | K=%.3f → %s",
             result.belief[0], result.belief[1],
             result.belief[2], result.belief[3],
             K, EMOTION_LABELS[result.dominant]);

    return result;
}


// =====================================================================
//  Utility
// =====================================================================

const char* DSFusionEngine::EmotionLabel(EmotionIndex idx) {
    if (idx < kNumEmotions) return EMOTION_LABELS[idx];
    return "unknown";
}

int64_t DSFusionEngine::NowMs() {
    return esp_timer_get_time() / 1000;    // µs → ms
}
