"""
main.py — SIEVOX K210 Main Loop (Upgraded with Emotion Recognition)
=====================================================================
Changes from original:
  1. Imports FEREngine for face detection + expression classification.
  2. Every loop iteration runs FER inference and packages the emotion
     probability array into the UART telemetry packet sent to ESP32.
  3. The existing gimbal-tracking (AudioTargetDetector) and manual
     command-handling logic are fully preserved.
  4. FER runs at a *lower cadence* than the gimbal loop (configurable)
     to avoid starving the servos of CPU time on K210.
"""

import time
import sys
import gc
from machine import Timer, PWM

# Existing modules
from audio_detector import AudioTargetDetector
from servo import Servo
from doa_calc import PID
from gimbal import Gimbal
from led_ring import lcd_setup, lcd_show_gimbal_delta
from config import load_config_from_json

# ── NEW: emotion recognition + upgraded UART ────────────────────────────
from fer_engine import FEREngine, EMOTION_LABELS
from uart_comm import UartComm


def main():
    # ==================================================================
    #  Hardware init — PWM timers for gimbal servos
    # ==================================================================
    tim_pitch = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
    tim_roll  = Timer(Timer.TIMER0, Timer.CHANNEL1, mode=Timer.MODE_PWM)
    pwm_pitch = PWM(tim_pitch, freq=50, duty=0, pin=7)
    pwm_roll  = PWM(tim_roll,  freq=50, duty=0, pin=8)

    # ==================================================================
    #  Load JSON config (PID gains, servo limits, etc.)
    # ==================================================================
    config = load_config_from_json()

    # ==================================================================
    #  Audio target detector (Sound Source Localisation)
    # ==================================================================
    detector = AudioTargetDetector(
        out_range    = config["audio_range"],
        ignore_limit = config["ignore_threshold"]
    )

    # ==================================================================
    #  PID controllers for pitch / roll axes
    # ==================================================================
    pid_pitch = PID(*config["pitch_pid"], is_roll=False)
    pid_roll  = PID(*config["roll_pid"],  is_roll=True)

    # ==================================================================
    #  Servo objects
    # ==================================================================
    pitch_servo = Servo(
        pwm_pitch,
        dir=config["init_pitch"],
        range_min=0, range_max=100,
        is_roll=False
    )
    roll_servo = Servo(
        pwm_roll,
        dir=config["init_roll"],
        range_min=0, range_max=100,
        is_roll=True
    )

    # ==================================================================
    #  Gimbal controller (wraps PID + servos)
    # ==================================================================
    gimbal = Gimbal(
        pitch=pitch_servo, pid_pitch=pid_pitch,
        roll=roll_servo,   pid_roll=pid_roll
    )

    # ==================================================================
    #  LCD / display
    # ==================================================================
    lcd_setup(rotation=0)
    init_pitch = config["init_pitch"]
    init_roll  = config["init_roll"]
    pitch_disp_invert = config.get("pitch_disp_invert", False)
    roll_disp_invert  = config.get("roll_disp_invert",  False)

    # ==================================================================
    #  UART bridge to ESP32
    # ==================================================================
    uart_esp = UartComm()

    # ==================================================================
    #  ★  NEW — Facial Expression Recognition engine  ★
    # ==================================================================
    fer = FEREngine(
        face_model_path = config.get("face_model", "/sd/models/face_detect.kmodel"),
        fer_model_path  = config.get("fer_model",  "/sd/models/fer_mobilenet.kmodel"),
        face_threshold  = config.get("face_threshold", 0.5),
    )

    # FER cadence: run inference every N main-loop iterations to balance
    # CPU between gimbal control and vision.  Default = every 3rd frame.
    fer_every_n     = config.get("fer_every_n", 3)
    fer_tick        = 0
    last_emo_probs  = {label: 0.25 for label in EMOTION_LABELS}
    last_bbox       = None

    # ==================================================================
    #  Loop timing
    # ==================================================================
    loop_delay_s  = config.get("loop_delay", 0.05)
    loop_delay_ms = max(0, int(loop_delay_s * 1000))

    # ==================================================================
    #  Runtime state
    # ==================================================================
    state = {
        "tracking_enabled": True,
        "manual_pitch_step": 5,
        "manual_roll_step":  5,
    }

    # ------------------------------------------------------------------
    #  Helper: format a human-readable status string for command ACKs
    # ------------------------------------------------------------------
    def format_status(label="STATE"):
        return "{} pitch={:.1f} roll={:.1f} tracking={}".format(
            label,
            pitch_servo.get_position(),
            roll_servo.get_position() if roll_servo else 0.0,
            1 if state["tracking_enabled"] else 0
        )

    # ------------------------------------------------------------------
    #  Command handler (ESP32 → K210)
    # ------------------------------------------------------------------
    def handle_command(raw_cmd):
        if not raw_cmd:
            return None
        cmd = raw_cmd.strip().upper()
        if not cmd:
            return None

        if cmd == "GET_STATE":
            return format_status("STATE")

        if cmd == "ROLL_LEFT":
            if roll_servo:
                state["tracking_enabled"] = False
                roll_servo.turn_left(state["manual_roll_step"])
                return format_status("ACK ROLL_LEFT")
            return "ERROR roll_servo_unavailable"

        if cmd == "ROLL_RIGHT":
            if roll_servo:
                state["tracking_enabled"] = False
                roll_servo.turn_right(state["manual_roll_step"])
                return format_status("ACK ROLL_RIGHT")
            return "ERROR roll_servo_unavailable"

        if cmd == "PITCH_UP":
            state["tracking_enabled"] = False
            pitch_servo.drive(state["manual_pitch_step"])
            return format_status("ACK PITCH_UP")

        if cmd == "PITCH_DOWN":
            state["tracking_enabled"] = False
            pitch_servo.drive(-state["manual_pitch_step"])
            return format_status("ACK PITCH_DOWN")

        if cmd == "HOLD_POSITION":
            state["tracking_enabled"] = False
            return format_status("ACK HOLD_POSITION")

        if cmd == "RESET":
            state["tracking_enabled"] = False
            pitch_servo.set_position(init_pitch)
            if roll_servo:
                roll_servo.set_position(init_roll)
            return format_status("ACK RESET")

        if cmd == "ENABLE_TRACKING":
            state["tracking_enabled"] = True
            return format_status("ACK ENABLE_TRACKING")

        if cmd == "DISABLE_TRACKING":
            state["tracking_enabled"] = False
            return format_status("ACK DISABLE_TRACKING")

        return "ERROR unknown_command: {}".format(raw_cmd.strip())

    # ==================================================================
    #  ★  MAIN LOOP  ★
    # ==================================================================
    try:
        while True:

            # ── 1. Audio-based gimbal tracking ──────────────────────
            if state["tracking_enabled"]:
                height_err, horizontal_err = detector.get_target_err()
                gimbal.run(
                    height_err     = height_err,
                    horizontal_err = horizontal_err,
                    pitch_reverse  = config["pitch_reverse"],
                    roll_reverse   = config["roll_reverse"],
                )

            # ── 2. FER inference (throttled) ────────────────────────
            fer_tick += 1
            if fer_tick >= fer_every_n:
                fer_tick = 0
                try:
                    last_emo_probs, last_bbox = fer.infer()
                except Exception as e:
                    print("[MAIN] FER error: {}".format(e))
                    # Keep previous result on failure — graceful degradation
                gc.collect()      # reclaim KPU intermediate buffers

            # ── 3. LCD status display ───────────────────────────────
            lcd_show_gimbal_delta(
                pitch_val    = pitch_servo.value,
                roll_val     = roll_servo.value,
                init_pitch   = init_pitch,
                init_roll    = init_roll,
                pitch_scale  = 1.8,
                roll_scale   = 1.8,
                pitch_invert = pitch_disp_invert,
                roll_invert  = roll_disp_invert,
            )

            # ── 4. Send emotion telemetry to ESP32 ──────────────────
            uart_esp.send_emotion_packet(
                emo_probs        = last_emo_probs,
                bbox             = last_bbox,
                pitch_val        = pitch_servo.value,
                roll_val         = roll_servo.value,
                tracking_enabled = state["tracking_enabled"],
            )

            # ── 5. Receive & handle ESP32 commands (non-blocking) ───
            data = uart_esp.receive_line(timeout_ms=50)
            if data:
                # Ignore JSON packets echoed back (they start with '{')
                if not data.startswith("{"):
                    print("(k210) CMD from ESP32: {}".format(data))
                    response = handle_command(data)
                    if response:
                        uart_esp.send(response + "\n")
                        print("(k210) ACK sent: {}".format(response))

            # ── 6. Yield ────────────────────────────────────────────
            if loop_delay_ms > 0:
                time.sleep_ms(loop_delay_ms)

    except Exception as e:
        print("[MAIN] Fatal: {}".format(e))
    finally:
        # Graceful shutdown
        fer.deinit()
        detector.deinit()
        pitch_servo.drive(config["init_pitch"])
        roll_servo.drive(config["init_roll"])
        pitch_servo.enable(False)
        roll_servo.enable(False)


# ======================================================================
if __name__ == "__main__":
    main()
