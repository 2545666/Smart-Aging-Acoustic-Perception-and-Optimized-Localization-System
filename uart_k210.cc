/**
 * @file  uart_k210.cc
 * @brief UART bridge to K210 — upgraded implementation
 * ========================================================================
 * Project : SIEVOX — ESP32-S3 side
 * ========================================================================
 */

#include "uart_k210.h"

#include <esp_log.h>
#include <cJSON.h>
#include <cstring>
#include <cstdlib>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#define TAG "UART_K210"

// CRC-8 lookup table (polynomial 0x07, matches K210 side)
static uint8_t crc8_table[256];
static bool crc8_table_init = false;

static void init_crc8_table() {
    if (crc8_table_init) return;
    for (int i = 0; i < 256; i++) {
        uint8_t crc = (uint8_t)i;
        for (int j = 0; j < 8; j++) {
            crc = (crc & 0x80) ? ((crc << 1) ^ 0x07) : (crc << 1);
        }
        crc8_table[i] = crc;
    }
    crc8_table_init = true;
}


// =====================================================================
//  Init — unchanged from original, with added CRC table setup
// =====================================================================

void UartK210::Init() {
    init_crc8_table();

    uart_config_t uart_config = {
        .baud_rate  = BAUD_RATE,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_param_config(UART_NUM_, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM_, TX_PIN, RX_PIN,
                                  UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_, BUF_SIZE * 2, 0, 0, NULL, 0));

    ESP_LOGI(TAG, "UART initialised: TX=%d, RX=%d, Baud=%d", TX_PIN, RX_PIN, BAUD_RATE);
}


// =====================================================================
//  Send
// =====================================================================

void UartK210::SendData(const char* data, size_t len) {
    uart_write_bytes(UART_NUM_, data, len);
    ESP_LOGD(TAG, "→ K210: %.*s", (int)len, data);
}

void UartK210::SendCommand(const std::string& cmd) {
    std::string line = cmd;
    if (line.empty() || line.back() != '\n') {
        line += '\n';
    }
    SendData(line.c_str(), line.size());
}


// =====================================================================
//  Receive (raw)
// =====================================================================

int UartK210::ReceiveData(uint8_t* buffer, size_t max_len, uint32_t timeout_ms) {
    return uart_read_bytes(UART_NUM_, buffer, max_len, pdMS_TO_TICKS(timeout_ms));
}


// =====================================================================
//  CRC-8 computation
// =====================================================================

uint8_t UartK210::ComputeCRC8(const uint8_t* data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc = crc8_table[(crc ^ data[i]) & 0xFF];
    }
    return crc;
}


// =====================================================================
//  JSON packet parser
// =====================================================================

bool UartK210::ParseEmotionPacket(const char* json_line, VisionEmotionPacket& out) {
    /**
     * Expected JSON format from K210:
     * {
     *   "seq": 42, "ts": 123456,
     *   "face": true,
     *   "bbox": [80, 60, 120, 120],
     *   "emo": [0.10, 0.70, 0.15, 0.05],
     *   "pitch": 45.0, "roll": 12.3,
     *   "trk": true,
     *   "crc": "A3"
     * }
     */

    cJSON* root = cJSON_Parse(json_line);
    if (!root) {
        ESP_LOGW(TAG, "JSON parse failed: %.80s...", json_line);
        return false;
    }

    bool ok = true;

    // ── Sequence number ────────────────────────────────────────────
    cJSON* j_seq = cJSON_GetObjectItem(root, "seq");
    out.seq = cJSON_IsNumber(j_seq) ? (uint32_t)j_seq->valueint : 0;

    // ── Timestamp ──────────────────────────────────────────────────
    cJSON* j_ts = cJSON_GetObjectItem(root, "ts");
    out.ts = cJSON_IsNumber(j_ts) ? (uint32_t)j_ts->valueint : 0;

    // ── Face detected ──────────────────────────────────────────────
    cJSON* j_face = cJSON_GetObjectItem(root, "face");
    out.face_detected = cJSON_IsTrue(j_face);

    // ── Bounding box ───────────────────────────────────────────────
    memset(out.bbox, 0, sizeof(out.bbox));
    cJSON* j_bbox = cJSON_GetObjectItem(root, "bbox");
    if (cJSON_IsArray(j_bbox) && cJSON_GetArraySize(j_bbox) == 4) {
        for (int i = 0; i < 4; i++) {
            cJSON* item = cJSON_GetArrayItem(j_bbox, i);
            out.bbox[i] = cJSON_IsNumber(item) ? item->valueint : 0;
        }
    }

    // ── Emotion probability array ──────────────────────────────────
    out.emo_probs.fill(0.0f);
    cJSON* j_emo = cJSON_GetObjectItem(root, "emo");
    if (cJSON_IsArray(j_emo) && cJSON_GetArraySize(j_emo) == (int)kVisionEmotions) {
        for (size_t i = 0; i < kVisionEmotions; i++) {
            cJSON* item = cJSON_GetArrayItem(j_emo, (int)i);
            out.emo_probs[i] = cJSON_IsNumber(item) ? (float)item->valuedouble : 0.0f;
        }
    } else {
        ESP_LOGW(TAG, "Missing or malformed 'emo' array");
        ok = false;
    }

    // ── Gimbal state ───────────────────────────────────────────────
    cJSON* j_pitch = cJSON_GetObjectItem(root, "pitch");
    out.pitch = cJSON_IsNumber(j_pitch) ? (float)j_pitch->valuedouble : 0.0f;

    cJSON* j_roll = cJSON_GetObjectItem(root, "roll");
    out.roll = cJSON_IsNumber(j_roll) ? (float)j_roll->valuedouble : 0.0f;

    cJSON* j_trk = cJSON_GetObjectItem(root, "trk");
    out.tracking = cJSON_IsTrue(j_trk);

    // ── CRC-8 validation ───────────────────────────────────────────
    // Strategy: compute CRC over the JSON string up to (but not including)
    // the "crc" field.  We find the "crc" key and compute CRC on everything
    // before it.
    out.crc_valid = false;
    cJSON* j_crc = cJSON_GetObjectItem(root, "crc");
    if (cJSON_IsString(j_crc) && j_crc->valuestring) {
        // Parse expected CRC from hex string
        uint8_t expected_crc = (uint8_t)strtoul(j_crc->valuestring, nullptr, 16);

        // To validate, we need the JSON without the crc field.
        // Quick approach: remove the crc field, re-serialise, compute CRC.
        cJSON_DeleteItemFromObject(root, "crc");
        char* payload_str = cJSON_PrintUnformatted(root);
        if (payload_str) {
            uint8_t actual_crc = ComputeCRC8(
                (const uint8_t*)payload_str, strlen(payload_str));
            out.crc_valid = (actual_crc == expected_crc);
            if (!out.crc_valid) {
                ESP_LOGW(TAG, "CRC mismatch: expected=0x%02X actual=0x%02X",
                         expected_crc, actual_crc);
            }
            cJSON_free(payload_str);
        }
    } else {
        // No CRC field — accept anyway but flag
        ESP_LOGD(TAG, "Packet has no CRC field (seq=%u)", out.seq);
        out.crc_valid = true;   // lenient: accept legacy packets
    }

    cJSON_Delete(root);
    return ok;
}


// =====================================================================
//  Background receive task
// =====================================================================

void UartK210::StartReceiveTask() {
    xTaskCreate([](void* param) {
        UartK210* uart = static_cast<UartK210*>(param);
        uint8_t buffer[BUF_SIZE];
        size_t index = 0;

        ESP_LOGI(TAG, "Receive task started");

        while (true) {
            uint8_t byte;
            int len = uart->ReceiveData(&byte, 1, 100);

            if (len > 0) {
                if (byte == '\n') {
                    // ── Complete line received ─────────────────────
                    buffer[index] = '\0';

                    if (index == 0) {
                        // Empty line, ignore
                        continue;
                    }

                    // Check if it's a JSON packet (starts with '{')
                    if (buffer[0] == '{') {
                        VisionEmotionPacket pkt;
                        bool parsed = uart->ParseEmotionPacket(
                            (const char*)buffer, pkt);

                        if (parsed) {
                            // ── Dropped packet detection ───────────
                            if (uart->last_seq_ >= 0) {
                                int32_t expected = uart->last_seq_ + 1;
                                int32_t delta = (int32_t)pkt.seq - expected;
                                if (delta > 0) {
                                    uart->drop_count_ += delta;
                                    ESP_LOGW(TAG, "Dropped %d packet(s) "
                                             "(seq %d→%d, total drops=%d)",
                                             delta, uart->last_seq_,
                                             pkt.seq, uart->drop_count_);
                                }
                            }
                            uart->last_seq_ = (int32_t)pkt.seq;

                            // ── Deliver to fusion engine via callback
                            if (uart->vision_callback_) {
                                uart->vision_callback_(pkt);
                            }

                            ESP_LOGD(TAG, "Pkt seq=%u face=%d emo=[%.2f,%.2f,%.2f,%.2f]",
                                     pkt.seq, pkt.face_detected,
                                     pkt.emo_probs[0], pkt.emo_probs[1],
                                     pkt.emo_probs[2], pkt.emo_probs[3]);
                        }
                    } else {
                        // Plain-text line (e.g., ACK from K210)
                        ESP_LOGI(TAG, "← K210 (text): %s", buffer);
                    }

                    index = 0;

                } else if (index < BUF_SIZE - 1) {
                    buffer[index++] = byte;
                } else {
                    ESP_LOGW(TAG, "Buffer overflow — discarding line");
                    index = 0;
                }
            }
        }
    }, "uart_k210_rx", 4096, this, 5, NULL);
}
