/**
 * @file  uart_k210.h
 * @brief UART bridge to K210 — upgraded with JSON emotion packet parsing
 * ========================================================================
 * Project : SIEVOX — ESP32-S3 side
 * ========================================================================
 *
 * Changes from original:
 *   1. ParseEmotionPacket() — decodes the JSON telemetry from K210.
 *   2. Sequence-number tracking for dropped-packet detection.
 *   3. CRC-8 validation of incoming packets.
 *   4. Callback mechanism for delivering parsed results to the fusion engine.
 *   5. Non-blocking receive with line buffering (no busy-wait).
 */

#ifndef UART_K210_H
#define UART_K210_H

#include <driver/uart.h>
#include <functional>
#include <array>
#include <string>
#include <cstdint>

// Must match K210 side: [happy, sad, neutral, anger]
constexpr size_t kVisionEmotions = 4;

/**
 * @brief Parsed emotion telemetry packet from K210.
 */
struct VisionEmotionPacket {
    uint32_t seq;                                    ///< Sequence number
    uint32_t ts;                                     ///< K210 uptime (ms)
    bool     face_detected;                          ///< Face in frame?
    int      bbox[4];                                ///< [x, y, w, h] or zeros
    std::array<float, kVisionEmotions> emo_probs;    ///< [H, S, N, A]
    float    pitch;                                  ///< Gimbal pitch (°)
    float    roll;                                   ///< Gimbal roll  (°)
    bool     tracking;                               ///< Tracking enabled?
    bool     crc_valid;                              ///< CRC check passed?
};

/**
 * @brief Callback type invoked when a valid emotion packet is received.
 */
using VisionPacketCallback = std::function<void(const VisionEmotionPacket&)>;


class UartK210 {
public:
    void Init();
    void SendData(const char* data, size_t len);
    int  ReceiveData(uint8_t* buffer, size_t max_len, uint32_t timeout_ms);

    /**
     * @brief Start the background receive task.
     *
     * Spawns a FreeRTOS task that continuously reads UART lines,
     * parses JSON emotion packets, validates CRC, and invokes the
     * registered callback for each valid packet.
     *
     * Plain-text lines (command ACKs from K210) are logged but not
     * forwarded to the callback.
     */
    void StartReceiveTask();

    /**
     * @brief Register a callback for incoming emotion packets.
     * @param cb  Function to call with each parsed VisionEmotionPacket.
     */
    void SetVisionCallback(VisionPacketCallback cb) {
        vision_callback_ = std::move(cb);
    }

    /**
     * @brief Send a command to K210 (e.g., "GET_STATE\n").
     * Convenience wrapper that appends \n if missing.
     */
    void SendCommand(const std::string& cmd);

private:
    static constexpr uart_port_t UART_NUM_ = UART_NUM_1;
    static constexpr int TX_PIN    = 17;
    static constexpr int RX_PIN    = 18;
    static constexpr int BAUD_RATE = 115200;
    static constexpr int BUF_SIZE  = 2048;

    VisionPacketCallback vision_callback_;

    // Sequence tracking
    int32_t last_seq_    = -1;
    int     drop_count_  = 0;

    /**
     * @brief Parse a JSON line into a VisionEmotionPacket.
     * @param json_line  Null-terminated JSON string (one line).
     * @param out        Output struct.
     * @return true if parsing succeeded and CRC is valid.
     */
    bool ParseEmotionPacket(const char* json_line, VisionEmotionPacket& out);

    /**
     * @brief Compute CRC-8 (poly 0x07) over a byte string.
     * @param data  Pointer to data bytes.
     * @param len   Number of bytes.
     * @return CRC-8 value.
     */
    static uint8_t ComputeCRC8(const uint8_t* data, size_t len);
};

#endif // UART_K210_H
