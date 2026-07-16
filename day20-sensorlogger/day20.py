import time
import csv
import math
from collections import deque
from datetime import datetime

import psutil
import cv2
import numpy as np
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group


SAMPLE_INTERVAL = 0.1
RING_BUFFER_SIZE = 100
CSV_FILE = "sensor_log.csv"

THRESHOLDS = {
    "mic_rms": 0.08,
    "camera_motion": 25.0,
    "cpu_percent": 85.0,
    "memory_percent": 90.0,
}


class RingBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, sample):
        self.buffer.append(sample)

    def get_all(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)


def get_cpu_usage():
    return psutil.cpu_percent(interval=None)


def get_memory_usage():
    return psutil.virtual_memory().percent


def get_battery_or_temp():
    battery = psutil.sensors_battery()

    if battery is not None:
        return battery.percent

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            first_sensor = list(temps.values())[0][0]
            return first_sensor.current
    except Exception:
        pass

    return 0.0


def get_fake_mic_rms():
    """
    Simple microphone RMS simulation.

    This keeps the project easy to run even if PyAudio is not installed.
    It creates a small changing signal with occasional spikes.
    """
    base = np.random.normal(0.02, 0.005)

    if np.random.random() < 0.04:
        base += np.random.uniform(0.08, 0.15)

    return max(0.0, float(base))


class CameraMotionSensor:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.previous_gray = None
        self.camera_available = self.cap.isOpened()

    def read_motion(self):
        if not self.camera_available:
            return 0.0

        ret, frame = self.cap.read()

        if not ret:
            return 0.0

        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_gray is None:
            self.previous_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = float(np.mean(magnitude) * 100)

        self.previous_gray = gray
        return motion_score

    def release(self):
        if self.cap is not None:
            self.cap.release()


def detect_anomalies(sample):
    anomalies = []

    for key, threshold in THRESHOLDS.items():
        if sample[key] > threshold:
            anomalies.append(key)

    return anomalies


def make_dashboard(sample, anomalies, ring_buffer):
    table = Table(title="Day 20 SensorLogger Live Dashboard")

    table.add_column("Channel", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Threshold", style="yellow")
    table.add_column("Status", style="magenta")

    for key in THRESHOLDS:
        value = sample[key]
        threshold = THRESHOLDS[key]

        status = "SPIKE" if key in anomalies else "normal"

        table.add_row(
            key,
            f"{value:.3f}",
            f"{threshold:.3f}",
            status,
        )

    info = Table.grid()
    info.add_row(f"Timestamp: {sample['timestamp']}")
    info.add_row(f"Ring Buffer: {len(ring_buffer)} / {RING_BUFFER_SIZE}")
    info.add_row(f"Sample Interval: {SAMPLE_INTERVAL * 1000:.0f} ms")

    if anomalies:
        anomaly_text = ", ".join(anomalies)
        anomaly_panel = Panel(
            f"ANOMALY DETECTED: {anomaly_text}",
            title="Alert",
            style="bold red",
        )
    else:
        anomaly_panel = Panel(
            "No anomaly",
            title="Alert",
            style="green",
        )

    return Group(info, table, anomaly_panel)


def write_csv_header():
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "mic_rms",
            "camera_motion",
            "cpu_percent",
            "memory_percent",
            "battery_or_temp",
        ])


def append_sample_to_csv(sample):
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            sample["timestamp"],
            sample["mic_rms"],
            sample["camera_motion"],
            sample["cpu_percent"],
            sample["memory_percent"],
            sample["battery_or_temp"],
        ])


def main():
    ring_buffer = RingBuffer(RING_BUFFER_SIZE)
    camera_sensor = CameraMotionSensor(camera_index=0)

    write_csv_header()

    try:
        with Live(refresh_per_second=4, screen=True) as live:
            while True:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                mic_rms = get_fake_mic_rms()
                camera_motion = camera_sensor.read_motion()
                cpu_percent = get_cpu_usage()
                memory_percent = get_memory_usage()
                battery_or_temp = get_battery_or_temp()

                sample = {
                    "timestamp": timestamp,
                    "mic_rms": mic_rms,
                    "camera_motion": camera_motion,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "battery_or_temp": battery_or_temp,
                }

                ring_buffer.add(sample)
                append_sample_to_csv(sample)

                anomalies = detect_anomalies(sample)
                dashboard = make_dashboard(sample, anomalies, ring_buffer)

                live.update(dashboard)

                time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        camera_sensor.release()
        print("\nSensor logging stopped.")
        print(f"Data saved to {CSV_FILE}")


if __name__ == "__main__":
    main()