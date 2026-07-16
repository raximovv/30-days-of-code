import cv2
import mediapipe as mp
import numpy as np
import platform
import subprocess
import sys
 
OS = platform.system()
 
 
def set_system_volume(percent):
    percent = max(0, min(100, int(percent)))
    try:
        if OS == "Darwin":
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {percent}"],
                capture_output=True, timeout=2
            )
        elif OS == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(percent / 100.0, None)
            except (ImportError, AttributeError):
                pass
        else:
            result = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
                capture_output=True, timeout=2
            )
            if result.returncode != 0:
                subprocess.run(
                    ["amixer", "set", "Master", f"{percent}%"],
                    capture_output=True, timeout=2
                )
    except Exception:
        pass
 
 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)
 
ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]
 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils
 
WRIST = 0
current_volume = 50
smoothed_volume = 50.0
SMOOTHING = 0.3
 
DEAD_ZONE_TOP = 0.10
DEAD_ZONE_BOTTOM = 0.90
 
 
def fist_to_volume(y_normalized):
    if y_normalized < DEAD_ZONE_TOP:
        return 100.0
    elif y_normalized > DEAD_ZONE_BOTTOM:
        return 0.0
    return np.interp(y_normalized, [DEAD_ZONE_TOP, DEAD_ZONE_BOTTOM], [100, 0])
 
 
print("\nVolumeKnuckle is running!")
print(f"OS: {OS}")
print(f"Dead zones: top {DEAD_ZONE_TOP*100:.0f}%, bottom {(1-DEAD_ZONE_BOTTOM)*100:.0f}%")
print("Fist UP = louder. Fist DOWN = quieter.")
print("Show open hand first so MediaPipe can detect, then close fist.")
print("Press 'q' to quit.\n")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
 
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        fist_y = hand_landmarks.landmark[WRIST].y
        raw_volume = fist_to_volume(fist_y)
        smoothed_volume = smoothed_volume + SMOOTHING * (raw_volume - smoothed_volume)
        current_volume = int(smoothed_volume)
 
        set_system_volume(current_volume)
 
        bar_x = FRAME_W - 60
        bar_top = 50
        bar_bottom = FRAME_H - 50
        bar_height = bar_bottom - bar_top
 
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 30, bar_bottom), (50, 50, 50), -1)
 
        fill_height = int(bar_height * current_volume / 100)
        fill_top = bar_bottom - fill_height
 
        if current_volume < 33:
            bar_color = (0, 200, 0)
        elif current_volume < 66:
            bar_color = (0, 220, 220)
        else:
            bar_color = (0, 80, 255)
 
        cv2.rectangle(frame, (bar_x, fill_top), (bar_x + 30, bar_bottom), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 30, bar_bottom), (255, 255, 255), 2)
 
        dz_top_px = bar_top + int(bar_height * DEAD_ZONE_TOP)
        dz_bot_px = bar_top + int(bar_height * DEAD_ZONE_BOTTOM)
        cv2.line(frame, (bar_x - 5, dz_top_px), (bar_x + 35, dz_top_px), (100, 100, 255), 1)
        cv2.line(frame, (bar_x - 5, dz_bot_px), (bar_x + 35, dz_bot_px), (100, 100, 255), 1)
 
        fist_px_y = int(fist_y * FRAME_H)
        cv2.circle(frame, (bar_x + 15, fist_px_y), 8, (255, 255, 255), -1)
 
        cv2.putText(frame, f"Volume: {current_volume}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, bar_color, 3)
        cv2.putText(frame, f"Fist Y: {fist_y:.2f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
 
    else:
        cv2.putText(frame, "No hand — show open hand first",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Volume: {current_volume}%", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
 
    cv2.imshow("VolumeKnuckle - Day 03", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
print(f"\nVolumeKnuckle ended. Final volume: {current_volume}%")