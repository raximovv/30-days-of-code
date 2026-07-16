import cv2
import base64
import hashlib
import sys
import os
import time
import threading
import queue
import platform

import requests

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("mediapipe not found — using full frame crop")

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    print("pyttsx3 not found — TTS disabled")

VISION_MODEL   = "moondream"
LLM_MODEL      = "qwen2.5:3b"
OLLAMA_URL     = "http://localhost:11434"
CAPTURE_INTERVAL = 5.0
AUTO_CAPTURE   = False
MAX_QUEUE_SIZE = 2
MOUTH_CROP_SIZE = 256


def check_ollama():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        names = [m["name"] for m in resp.json().get("models", [])]
        missing = [m for m in [VISION_MODEL, "qwen2.5:3b"] if not any(m in n for n in names)]
        if missing:
            print(f"Missing models: {', '.join(missing)}")
            print("Fix: " + " && ".join(f"ollama pull {m}" for m in missing))
            sys.exit(1)
        print("ollama models ready")
    except Exception as e:
        print(f"ERROR: ollama not reachable — {e}\nRun: ollama serve")
        sys.exit(1)


class LatencyTracker:
    def __init__(self):
        self.stages = {s: [] for s in ("capture", "vision", "generate", "speak", "total")}
        self.lock = threading.Lock()

    def record(self, stage, ms):
        with self.lock:
            self.stages[stage].append(ms)
            self.stages[stage] = self.stages[stage][-20:]

    def avg(self, stage):
        with self.lock:
            vals = self.stages[stage]
            return sum(vals) / len(vals) if vals else 0.0

    def report(self):
        print("\n" + "=" * 55)
        print("  Pipeline Latency Report")
        print("=" * 55)
        for stage in ("capture", "vision", "generate", "speak"):
            avg = self.avg(stage)
            bar = "█" * max(1, int(avg / 200)) if avg > 0 else ""
            print(f"  {stage:<10} {avg:6.0f}ms  {bar}")
        total = sum(self.avg(s) for s in ("capture", "vision", "generate", "speak"))
        print(f"  {'TOTAL':<10} {total:6.0f}ms")
        bottleneck = max(("capture", "vision", "generate", "speak"), key=self.avg)
        print(f"\n  Bottleneck: {bottleneck} ({self.avg(bottleneck):.0f}ms avg)")
        print("=" * 55 + "\n")


latency = LatencyTracker()

capture_queue  = queue.Queue(maxsize=MAX_QUEUE_SIZE)
vision_queue   = queue.Queue(maxsize=MAX_QUEUE_SIZE)
generate_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
pipeline_running = threading.Event()
pipeline_running.set()


def frame_to_b64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def frame_hash(img):
    small = cv2.resize(img, (16, 16))
    return hashlib.md5(small.tobytes()).hexdigest()


def ollama_vision(image_b64, prompt):
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
    return resp.json().get("response", "").strip()


def ollama_generate(prompt):
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
    return resp.json().get("response", "").strip()


def capture_stage(cap):
    face_mesh = None
    if HAS_MEDIAPIPE:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    while pipeline_running.is_set():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)

        if face_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            mouth_crop = None
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                mouth_idxs = [61, 146, 91, 181, 84, 17, 314, 405,
                              321, 375, 291, 308, 324, 318, 402, 317,
                              14, 87, 178, 88, 95]
                xs = [int(lms[i].x * w) for i in mouth_idxs]
                ys = [int(lms[i].y * h) for i in mouth_idxs]
                m = 30
                x1, y1 = max(0, min(xs) - m), max(0, min(ys) - m)
                x2, y2 = min(w, max(xs) + m), min(h, max(ys) + m)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    mouth_crop = cv2.resize(crop, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))
            if mouth_crop is None:
                mouth_crop = cv2.resize(frame, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))
        else:
            h, w = frame.shape[:2]
            cy, cx = h // 2 + h // 8, w // 2
            sz = min(h, w) // 4
            crop = frame[max(0, cy - sz):cy + sz, max(0, cx - sz):cx + sz]
            mouth_crop = cv2.resize(crop, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))

        latency.record("capture", (time.time() - t0) * 1000)

        try:
            capture_queue.put_nowait((frame, mouth_crop, time.time()))
        except queue.Full:
            pass

        if AUTO_CAPTURE:
            elapsed = time.time() - t0
            time.sleep(max(0, CAPTURE_INTERVAL - elapsed))
        else:
            time.sleep(0.1)


_vision_cache: dict = {}

def vision_stage():
    while pipeline_running.is_set():
        try:
            frame, mouth_crop, t_cap = capture_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.time()
        img_hash = frame_hash(mouth_crop)

        if img_hash in _vision_cache:
            description = _vision_cache[img_hash]
        else:
            b64 = frame_to_b64(mouth_crop)
            prompt = (
                "Describe this image of a person's mouth area in one sentence. "
                "Is the mouth open or closed? Are the lips moving?"
            )
            try:
                description = ollama_vision(b64, prompt)
            except Exception as e:
                description = f"vision error: {e}"
            _vision_cache[img_hash] = description
            if len(_vision_cache) > 50:
                oldest = next(iter(_vision_cache))
                del _vision_cache[oldest]

        ms = (time.time() - t0) * 1000
        latency.record("vision", ms)
        print(f"[Vision {ms:.0f}ms] {description[:90]}")

        try:
            vision_queue.put_nowait((description, t_cap, time.time()))
        except queue.Full:
            pass


_generate_cache: dict = {}

def generate_stage():
    while pipeline_running.is_set():
        try:
            description, t_cap, t_vis = vision_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.time()

        if description in _generate_cache:
            response = _generate_cache[description]
        else:
            prompt = (
                f"Camera sees: '{description}'\n"
                "You are a silent assistant. What might this person be communicating? "
                "Reply in one short sentence."
            )
            try:
                response = ollama_generate(prompt)
            except Exception as e:
                response = f"generate error: {e}"
            _generate_cache[description] = response
            if len(_generate_cache) > 50:
                oldest = next(iter(_generate_cache))
                del _generate_cache[oldest]

        ms = (time.time() - t0) * 1000
        latency.record("generate", ms)
        print(f"[Generate {ms:.0f}ms] {response}")

        try:
            generate_queue.put_nowait((response, t_cap, time.time()))
        except queue.Full:
            pass


def speak_stage():
    engine = None
    if HAS_TTS:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            print("TTS engine ready")
        except Exception as e:
            print(f"TTS init failed: {e}")

    last_spoken = None

    while pipeline_running.is_set():
        try:
            response, t_cap, t_gen = generate_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.time()

        if response != last_spoken:
            if engine:
                try:
                    engine.say(response)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
            else:
                print(f"[TTS disabled] {response}")
                time.sleep(0.3)
            last_spoken = response
        else:
            print(f"[TTS skipped — duplicate]")

        ms = (time.time() - t0) * 1000
        total_ms = (time.time() - t_cap) * 1000
        latency.record("speak", ms)
        latency.record("total", total_ms)
        print(f"[Speak {ms:.0f}ms] [End-to-end {total_ms:.0f}ms]")


def show_webcam(cap):
    while pipeline_running.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "SilentAssistant D29",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        mode = "AUTO" if AUTO_CAPTURE else "MANUAL (s=trigger)"
        cv2.putText(frame, mode,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("SilentAssistant - Day 29", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            pipeline_running.clear()
            break


def main():
    print("\n" + "=" * 55)
    print("  SilentAssistant — Day 29 (EXPERT)")
    print("=" * 55)
    print(f"  Vision : {VISION_MODEL}  |  LLM: {LLM_MODEL}")
    print(f"  TTS    : {'pyttsx3' if HAS_TTS else 'disabled'}")
    print(f"  Mode   : {'AUTO' if AUTO_CAPTURE else 'MANUAL'}")
    print()
    print("  Commands: s=trigger  r=latency report  q=quit")
    print("=" * 55 + "\n")

    check_ollama()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No webcam found.")
        sys.exit(1)
    print("Webcam open\n")

    threads = [
        threading.Thread(target=capture_stage, args=(cap,), daemon=True, name="capture"),
        threading.Thread(target=vision_stage,             daemon=True, name="vision"),
        threading.Thread(target=generate_stage,           daemon=True, name="generate"),
        threading.Thread(target=speak_stage,              daemon=True, name="speak"),
        threading.Thread(target=show_webcam, args=(cap,), daemon=True, name="display"),
    ]
    for t in threads:
        t.start()
    print("Pipeline running\n")

    try:
        while pipeline_running.is_set():
            try:
                cmd = input().strip().lower()
            except EOFError:
                break

            if cmd == "q":
                break
            elif cmd == "r":
                latency.report()
            elif cmd == "s":
                if AUTO_CAPTURE:
                    print("Already in auto mode")
                else:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        crop = cv2.resize(frame, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))
                        try:
                            capture_queue.put_nowait((frame, crop, time.time()))
                            print("Frame queued")
                        except queue.Full:
                            print("Queue full — still processing")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline_running.clear()
        cap.release()
        cv2.destroyAllWindows()
        latency.report()
        print("SilentAssistant shut down. Ship it.")


if __name__ == "__main__":
    main()