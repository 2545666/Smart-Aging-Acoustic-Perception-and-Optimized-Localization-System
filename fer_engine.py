"""
fer_engine.py — Facial Expression Recognition Engine for K210 / MaixCAM
=========================================================================
Project : SIEVOX — Multimodal Elderly Care Robot
Target  : K210 (MaixPy) or MaixCAM (MaixPy3)
Author  : SIEVOX Vision Group
---------------------------------------------------------------------------

This module wraps a lightweight KPU-accelerated face detection + emotion
classification pipeline.  It outputs a normalised probability vector over
4 basic emotions:  [happy, sad, neutral, anger].

Model files required on SD-card:
    /sd/models/face_detect.kmodel      — YOLO-based face bbox detector
    /sd/models/fer_mobilenet.kmodel    — MobileNetV2-0.35 emotion classifier

Both models must be INT8-quantised and compiled with nncase for the K210 KPU.
"""

import time
import gc

# --------------------------------------------------------------------------
# Platform-adaptive imports
# MaixPy (K210): uses KPU + sensor + image modules directly
# MaixPy3 (MaixCAM): nearly identical API, minor differences handled below
# --------------------------------------------------------------------------
try:
    import KPU as kpu
    import sensor
    import image
    _PLATFORM = "K210"
except ImportError:
    # MaixCAM fallback — adjust imports as needed for your BSP
    from maix import camera, display, nn
    _PLATFORM = "MAIXCAM"


# ---------------------------------------------------------------------------
# Emotion label order — MUST match the classifier's softmax output order.
# The same order is used on the ESP32 side for D-S fusion.
# ---------------------------------------------------------------------------
EMOTION_LABELS = ["happy", "sad", "neutral", "anger"]
NUM_EMOTIONS   = len(EMOTION_LABELS)

# Confidence floor: if the highest softmax output is below this threshold
# we report a uniform "uncertain" distribution instead of a noisy spike.
CONFIDENCE_FLOOR = 0.25


class FEREngine:
    """
    Lightweight Facial Expression Recognition engine.

    Usage
    -----
    >>> engine = FEREngine()
    >>> probs, bbox = engine.infer()        # returns (dict, tuple|None)
    >>> # probs = {"happy": 0.1, "sad": 0.7, "neutral": 0.15, "anger": 0.05}
    >>> # bbox  = (x, y, w, h) of the detected face, or None if no face
    >>> engine.deinit()
    """

    def __init__(
        self,
        face_model_path:  str = "/sd/models/face_detect.kmodel",
        fer_model_path:   str = "/sd/models/fer_mobilenet.kmodel",
        face_threshold:   float = 0.5,
        fer_input_size:   tuple = (48, 48),      # emotion classifier input HxW
        camera_framesize: int = None,             # sensor.QVGA by default
    ):
        """
        Parameters
        ----------
        face_model_path   : path to the KPU face-detection model
        fer_model_path    : path to the KPU emotion classifier model
        face_threshold    : minimum bbox confidence for a valid face
        fer_input_size    : spatial size the emotion model expects (H, W)
        camera_framesize  : MaixPy sensor constant (default QVGA = 320x240)
        """
        self._face_threshold = face_threshold
        self._fer_size       = fer_input_size
        self._last_probs     = self._uniform_probs()
        self._last_bbox      = None

        # ------------------------------------------------------------------
        # K210 / MaixPy initialisation
        # ------------------------------------------------------------------
        if _PLATFORM == "K210":
            # Camera
            sensor.reset()
            sensor.set_pixformat(sensor.RGB565)
            if camera_framesize is None:
                camera_framesize = sensor.QVGA          # 320 × 240
            sensor.set_framesize(camera_framesize)
            sensor.set_hmirror(False)
            sensor.set_vflip(False)
            sensor.run(1)

            # Load face detector onto KPU
            self._face_kpu = kpu.load(face_model_path)
            kpu.set_outputs(self._face_kpu, 0, 5, 5, 25)   # adjust anchors for your model

            # Load emotion classifier onto KPU (second model slot)
            self._fer_kpu = kpu.load(fer_model_path)

            print("[FER] K210 models loaded  face={} fer={}".format(
                face_model_path, fer_model_path))
        else:
            # MaixCAM placeholder — adapt to maix.nn API
            self._cam = camera.Camera()
            self._face_det = nn.load(face_model_path)
            self._fer_cls  = nn.load(fer_model_path)
            print("[FER] MaixCAM models loaded")

    # ======================================================================
    #  Public API
    # ======================================================================

    def infer(self):
        """
        Capture one frame, detect the largest face, classify its expression.

        Returns
        -------
        probs : dict   — {"happy": float, "sad": float, ...} summing to ~1.0
        bbox  : tuple  — (x, y, w, h) of the detected face or None
        """
        if _PLATFORM == "K210":
            return self._infer_k210()
        else:
            return self._infer_maixcam()

    def get_last_result(self):
        """Return the most recent (probs, bbox) without running inference."""
        return self._last_probs, self._last_bbox

    def deinit(self):
        """Release KPU resources and stop the camera."""
        if _PLATFORM == "K210":
            try:
                kpu.deinit(self._face_kpu)
            except:
                pass
            try:
                kpu.deinit(self._fer_kpu)
            except:
                pass
            sensor.run(0)
        else:
            self._cam.close()
        gc.collect()
        print("[FER] Engine deinitialised")

    # ======================================================================
    #  Internal — K210 / MaixPy path
    # ======================================================================

    def _infer_k210(self):
        """Full pipeline on K210 KPU."""
        img = sensor.snapshot()

        # ------ Step 1: Face detection ------
        # Run the face detector — returns list of bounding-box dicts
        try:
            face_results = kpu.run_yolo2(self._face_kpu, img)
        except Exception as e:
            print("[FER] Face detect error: {}".format(e))
            return self._uniform_probs(), None

        if not face_results:
            # No face in frame — return last known or uniform
            self._last_bbox = None
            return self._last_probs, None

        # Pick the largest face (closest to the camera)
        best = max(face_results, key=lambda r: r.w() * r.h())
        if best.value() < self._face_threshold:
            self._last_bbox = None
            return self._last_probs, None

        x, y, w, h = best.x(), best.y(), best.w(), best.h()
        self._last_bbox = (x, y, w, h)

        # ------ Step 2: Crop + preprocess for emotion classifier ------
        # Add a small margin around the face for context
        margin = max(w, h) // 6
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.width(),  x + w + margin)
        y2 = min(img.height(), y + h + margin)

        face_roi = img.copy((x1, y1, x2 - x1, y2 - y1))
        face_roi = face_roi.resize(self._fer_size[1], self._fer_size[0])
        face_roi.pix_to_ai()                    # convert to KPU input format

        # ------ Step 3: Emotion classification ------
        try:
            fmap = kpu.forward(self._fer_kpu, face_roi)
            raw_output = list(fmap[:])           # flatten feature-map to list
        except Exception as e:
            print("[FER] Classifier error: {}".format(e))
            return self._last_probs, self._last_bbox

        # ------ Step 4: Softmax normalisation ------
        probs = self._softmax(raw_output[:NUM_EMOTIONS])

        # Guard against low-confidence garbage
        if max(probs) < CONFIDENCE_FLOOR:
            probs = [1.0 / NUM_EMOTIONS] * NUM_EMOTIONS

        result = {}
        for i, label in enumerate(EMOTION_LABELS):
            result[label] = round(probs[i], 4)

        self._last_probs = result
        return result, self._last_bbox

    # ======================================================================
    #  Internal — MaixCAM path  (stub — fill in when hardware arrives)
    # ======================================================================

    def _infer_maixcam(self):
        """Placeholder for MaixCAM pipeline — structurally identical."""
        frame = self._cam.read()
        # TODO: adapt to maix.nn inference API
        # faces = self._face_det.detect(frame)
        # ...
        return self._uniform_probs(), None

    # ======================================================================
    #  Utility helpers
    # ======================================================================

    @staticmethod
    def _softmax(logits):
        """
        Numerically-stable softmax for a short list of floats.
        Avoids importing numpy (not available on MaixPy).
        """
        max_val = max(logits)
        exps = []
        for v in logits:
            # Clamp to avoid overflow on K210 (no double-precision FPU)
            e = v - max_val
            if e < -20.0:
                e = -20.0
            # Hand-rolled exp() — math.exp is available on MaixPy
            from math import exp
            exps.append(exp(e))
        total = sum(exps)
        if total == 0:
            return [1.0 / len(logits)] * len(logits)
        return [e / total for e in exps]

    @staticmethod
    def _uniform_probs():
        """Return a uniform distribution dict (used when no face detected)."""
        u = round(1.0 / NUM_EMOTIONS, 4)
        return {label: u for label in EMOTION_LABELS}
