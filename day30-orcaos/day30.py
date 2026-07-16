from __future__ import annotations

import collections
import platform
import subprocess
import sys
import threading
import time

# ── Textual ────────────────────────────────────────────────────────────────────
try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal
    from textual.reactive import reactive
    from textual import work
    from textual.widgets import Footer, Header, Input, Label, Log, Static
except ImportError:
    print("Textual not found. Run:  pip install textual")
    sys.exit(1)

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[warn] psutil missing — CPU/MEM disabled.  pip install psutil")

try:
    import pyaudio
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("[warn] pyaudio/numpy missing — mic disabled.  pip install pyaudio numpy")

try:
    import cv2
    import mediapipe as mp
    HAS_GESTURE = True
except ImportError:
    HAS_GESTURE = False
    print("[warn] cv2/mediapipe missing — gesture disabled.  pip install opencv-python mediapipe")

# ── Constants ──────────────────────────────────────────────────────────────────
LLM_MODEL   = "qwen2.5:3b"
LOG_MAXLEN  = 80
SENSOR_HZ   = 0.10   # seconds between sensor reads
GESTURE_HZ  = 0.05   # seconds between frame reads


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE  (thread-safe, RTOS-style queue substitute)
# ══════════════════════════════════════════════════════════════════════════════

class OrcaState:
    """
    Central state store for all subsystems.
    All writes use a single reentrant lock — safe from any thread.
    Think of this as the shared-memory region between RTOS tasks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # — Gesture channel —
        self.gesture_label      = "Waiting for webcam…"
        self.gesture_confidence = 0.0

        # — Sensor channel —
        self.mic_rms  = 0.0
        self.cpu_pct  = 0.0
        self.mem_pct  = 0.0

        # — System-pulse channel —
        self.net_sent_kb   = 0.0   # KB/s sent
        self.net_recv_kb   = 0.0   # KB/s received
        self.disk_read_kb  = 0.0   # KB/s read
        self.disk_write_kb = 0.0   # KB/s written
        self.process_count = 0

        # — LLM channel —
        self.llm_busy = False

        # — Event log —
        self.event_log: collections.deque[str] = collections.deque(maxlen=LOG_MAXLEN)

    # ── lock helpers ──────────────────────────────────────────────────────────

    def read(self, *attrs: str):
        """Return a tuple of attribute values under the lock."""
        with self._lock:
            return tuple(getattr(self, a) for a in attrs)

    def write(self, **kwargs) -> None:
        """Update any number of attributes under the lock."""
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self.event_log.append(f"[{ts}] {msg}")


# Global singleton (one shared state, like a firmware BSS segment)
state = OrcaState()


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND THREADS  (daemon — auto-killed when main thread exits)
# ══════════════════════════════════════════════════════════════════════════════

def _sensor_thread() -> None:
    """
    ISR-equivalent: reads CPU, MEM, and mic at SENSOR_HZ.
    Never touches the Textual event loop — only writes to OrcaState.
    """
    pa = stream = None

    if HAS_AUDIO:
        try:
            pa = pyaudio.PyAudio()
            dev_idx = next(
                (i for i in range(pa.get_device_count())
                 if pa.get_device_info_by_index(i)["maxInputChannels"] > 0),
                None,
            )
            if dev_idx is not None:
                stream = pa.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=16_000,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=1_600,
                )
        except Exception:
            pass

    while True:
        try:
            cpu = psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0
            mem = psutil.virtual_memory().percent   if HAS_PSUTIL else 0.0
            mic = 0.0
            if stream:
                try:
                    raw = stream.read(1_600, exception_on_overflow=False)
                    samples = np.frombuffer(raw, dtype=np.float32)
                    mic = float(np.sqrt(np.mean(samples ** 2)))
                except Exception:
                    pass
            state.write(cpu_pct=cpu, mem_pct=mem, mic_rms=mic)
        except Exception:
            pass
        time.sleep(SENSOR_HZ)


def _system_pulse_thread() -> None:
    """
    ISR-equivalent: reads network and disk I/O rates + process count.
    Computes delta-per-second between samples.
    """
    if not HAS_PSUTIL:
        return

    prev_net  = psutil.net_io_counters()
    prev_disk = psutil.disk_io_counters()
    prev_time = time.monotonic()

    while True:
        time.sleep(1.0)
        try:
            now_net  = psutil.net_io_counters()
            now_disk = psutil.disk_io_counters()
            now_time = time.monotonic()
            dt = now_time - prev_time or 1.0

            state.write(
                net_sent_kb   = (now_net.bytes_sent  - prev_net.bytes_sent)  / 1024 / dt,
                net_recv_kb   = (now_net.bytes_recv  - prev_net.bytes_recv)  / 1024 / dt,
                disk_read_kb  = (now_disk.read_bytes  - prev_disk.read_bytes)  / 1024 / dt,
                disk_write_kb = (now_disk.write_bytes - prev_disk.write_bytes) / 1024 / dt,
                process_count = len(psutil.pids()),
            )

            prev_net, prev_disk, prev_time = now_net, now_disk, now_time
        except Exception:
            pass


def _gesture_thread() -> None:
    """
    ISR-equivalent: reads webcam frames and classifies hand gestures.
    Falls back gracefully if hardware is unavailable.
    """
    if not HAS_GESTURE:
        state.write(gesture_label="MediaPipe not installed", gesture_confidence=0.0)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        state.write(gesture_label="No webcam found", gesture_confidence=0.0)
        state.log("Gesture: no webcam")
        return

    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    def _classify(hand_lm) -> tuple[str, float]:
        lm = hand_lm.landmark
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        up = [lm[t].y < lm[p].y for t, p in zip(tips[1:], pips[1:])]
        up.insert(0, lm[4].x < lm[3].x)   # thumb heuristic
        n = sum(up)
        if n == 0:                            return "Fist ✊",         0.90
        if n == 5:                            return "Open Hand 🖐",    0.90
        if up[1] and not any(up[2:]):         return "Point ☝️",        0.85
        if up[1] and up[2] and not any(up[3:]): return "Peace ✌️",      0.85
        if up[0] and not any(up[1:]):         return "Thumbs Up 👍",    0.85
        return f"{n} fingers",               0.70

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_model.process(rgb)

        if result.multi_hand_landmarks:
            label, conf = _classify(result.multi_hand_landmarks[0])
            (old_label,) = state.read("gesture_label")
            if old_label != label:
                state.log(f"Gesture → {label}")
            state.write(gesture_label=label, gesture_confidence=conf)
        else:
            state.write(gesture_label="No hand", gesture_confidence=0.0)

        time.sleep(GESTURE_HZ)

    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# LLM QUERY  (runs in Textual worker thread — never blocks the event loop)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are OrcaOS, a terse hardware engineering assistant embedded in a TUI. "
    "You specialise in embedded systems, RTOS, signal processing, and firmware. "
    "Reply in 1–3 sentences max. Be direct and technical."
)


def _query_llm(prompt: str) -> str:
    full = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nOrcaOS:"
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL, full],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() or "[empty response]"
    except subprocess.TimeoutExpired:
        return "[LLM timed out — is ollama running?]"
    except FileNotFoundError:
        return "[ollama not found — install from ollama.com]"
    except Exception as exc:
        return f"[Error: {exc}]"


# ══════════════════════════════════════════════════════════════════════════════
# TEXTUAL WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 10, char: str = "█") -> str:
    """Render a simple ASCII bar scaled 0–100."""
    filled = max(0, min(width, int(value / 100 * width)))
    return char * filled + "░" * (width - filled)


def _kb_bar(kbps: float, cap: float = 1024.0, width: int = 8) -> str:
    pct = min(100.0, kbps / cap * 100)
    filled = max(0, min(width, int(pct / 100 * width)))
    return "▓" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
class GesturePanel(Static):
    """Week 1 — hand gesture detection via MediaPipe."""

    DEFAULT_CSS = """
    GesturePanel {
        border: solid cyan;
        height: 7;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("✋  Gesture Input", id="g-title")
        yield Label("Initialising…",     id="g-label")
        yield Label("",                   id="g-conf")

    def on_mount(self) -> None:
        self.set_interval(0.2, self._refresh)

    def _refresh(self) -> None:
        label, conf = state.read("gesture_label", "gesture_confidence")
        bar = _bar(conf * 100, width=10, char="▮")
        self.query_one("#g-label", Label).update(f"  {label}")
        self.query_one("#g-conf",  Label).update(
            f"  Confidence  {bar}  {conf * 100:.0f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
class SensorPanel(Static):
    """Week 3 — live sensor channels (CPU, MEM, mic RMS)."""

    DEFAULT_CSS = """
    SensorPanel {
        border: solid green;
        height: 7;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("📊  Sensor Monitor", id="s-title")
        yield Label("CPU: —",             id="s-cpu")
        yield Label("MEM: —",             id="s-mem")
        yield Label("MIC: —",             id="s-mic")

    def on_mount(self) -> None:
        self.set_interval(0.5, self._refresh)

    def _refresh(self) -> None:
        cpu, mem, mic = state.read("cpu_pct", "mem_pct", "mic_rms")
        mic_bar_len = min(10, int(mic * 200))
        mic_bar     = "▮" * mic_bar_len + "░" * (10 - mic_bar_len)
        self.query_one("#s-cpu", Label).update(f"  CPU  {_bar(cpu)}  {cpu:5.1f}%")
        self.query_one("#s-mem", Label).update(f"  MEM  {_bar(mem)}  {mem:5.1f}%")
        self.query_one("#s-mic", Label).update(f"  MIC  {mic_bar}   {mic:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
class LLMPanel(Static):
    """Week 2 — local LLM reasoning via ollama."""

    DEFAULT_CSS = """
    LLMPanel {
        border: solid yellow;
        height: 13;
        padding: 1 2;
    }
    #llm-response {
        height: 4;
        overflow-y: scroll;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("🤖  OrcaOS LLM  —  " + LLM_MODEL, id="llm-title")
        yield Label("Ask a hardware / firmware question and press Enter:", id="llm-hint")
        yield Input(placeholder="e.g. Explain DMA vs interrupt-driven I/O", id="llm-input")
        yield Label("", id="llm-response")
        yield Label("  Ready", id="llm-status")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.clear()
        self.query_one("#llm-status",   Label).update("  ⏳ Thinking…")
        self.query_one("#llm-response", Label).update("")
        state.log(f"LLM Q: {prompt[:50]}")
        self._run_llm(prompt)

    @work(thread=True)
    def _run_llm(self, prompt: str) -> None:
        response = _query_llm(prompt)
        state.log(f"LLM A: {response[:50]}")

        def _update() -> None:
            self.query_one("#llm-response", Label).update(f"  {response}")
            self.query_one("#llm-status",   Label).update("  ✓ Ready")

        self.app.call_from_thread(_update)


# ─────────────────────────────────────────────────────────────────────────────
class SystemPulsePanel(Static):
    """
    Week 4 — system pulse: network I/O, disk I/O, process count.
    This is the 4th integrated component completing the MVD.
    """

    DEFAULT_CSS = """
    SystemPulsePanel {
        border: solid magenta;
        height: 8;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("⚡  System Pulse",  id="sp-title")
        yield Label("NET ↑ —",           id="sp-net-up")
        yield Label("NET ↓ —",           id="sp-net-dn")
        yield Label("DSK R/W — / —",     id="sp-disk")
        yield Label("PROCS —",           id="sp-procs")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh)

    def _refresh(self) -> None:
        sent, recv, dr, dw, procs = state.read(
            "net_sent_kb", "net_recv_kb",
            "disk_read_kb", "disk_write_kb",
            "process_count",
        )
        self.query_one("#sp-net-up", Label).update(
            f"  NET ↑  {_kb_bar(sent)}  {sent:7.1f} KB/s"
        )
        self.query_one("#sp-net-dn", Label).update(
            f"  NET ↓  {_kb_bar(recv)}  {recv:7.1f} KB/s"
        )
        self.query_one("#sp-disk",  Label).update(
            f"  DISK   R {dr:6.1f} KB/s  W {dw:6.1f} KB/s"
        )
        self.query_one("#sp-procs", Label).update(
            f"  PROCS  {procs}"
        )


# ─────────────────────────────────────────────────────────────────────────────
class EventLog(Static):
    """Scrolling event log — shows timestamped messages from all subsystems."""

    DEFAULT_CSS = """
    EventLog {
        border: solid $panel;
        height: 10;
        padding: 1 2;
    }
    """

    _last_count: int = 0

    def compose(self) -> ComposeResult:
        yield Label("📋  Event Log", id="el-title")
        yield Log(id="el-log", auto_scroll=True)

    def on_mount(self) -> None:
        state.log(f"OrcaOS v1.0 — {platform.system()} {platform.release()}")
        state.log(f"LLM model : {LLM_MODEL}")
        state.log(f"Gesture   : {'active' if HAS_GESTURE else 'DISABLED'}")
        state.log(f"Audio     : {'active' if HAS_AUDIO   else 'DISABLED'}")
        state.log(f"psutil    : {'active' if HAS_PSUTIL  else 'DISABLED'}")
        state.log("All systems nominal.")
        self.set_interval(0.5, self._refresh)

    def _refresh(self) -> None:
        with state._lock:
            events = list(state.event_log)
        log_widget = self.query_one("#el-log", Log)
        new = events[self._last_count:]
        for line in new:
            log_widget.write_line(line)
        self._last_count = len(events)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class OrcaOS(App):
    """
    OrcaOS — RTOS-in-software capstone TUI.

    Architecture overview:
      ┌──────────────────────────────────────────────┐
      │  daemon threads  →  OrcaState  →  widgets    │
      │  (ISRs / tasks)     (queues)     (display)   │
      └──────────────────────────────────────────────┘

    Press  q  to quit,  ctrl+l  to clear the event log.
    """

    TITLE     = "OrcaOS  v1.0"
    SUB_TITLE = "BUILDCORED ORCAS — Day 30 Capstone"

    BINDINGS = [
        Binding("q",      "quit",      "Quit"),
        Binding("ctrl+l", "clear_log", "Clear log"),
    ]

    CSS = """
    Screen {
        background: #07070f;
    }
    Header {
        background: #0d6e71;
        color: white;
    }
    Footer {
        background: #12121e;
    }
    #main {
        padding: 1 1;
    }
    #top-row {
        height: auto;
        margin-bottom: 1;
    }
    GesturePanel, SensorPanel {
        width: 1fr;
    }
    #mid-row {
        height: auto;
        margin-bottom: 1;
    }
    LLMPanel, SystemPulsePanel {
        width: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with Horizontal(id="top-row"):
                yield GesturePanel()
                yield SensorPanel()
            with Horizontal(id="mid-row"):
                yield LLMPanel()
                yield SystemPulsePanel()
            yield EventLog()
        yield Footer()

    def action_quit(self) -> None:
        self.exit()

    def action_clear_log(self) -> None:
        state.event_log.clear()
        state.log("Event log cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _start_daemon(target) -> None:
    t = threading.Thread(target=target, daemon=True, name=target.__name__)
    t.start()


def main() -> None:
    banner = r"""
  ___             ___  ____
 / _ \ _ __ ___ / _ \/ ___|
| | | | '__/ __| | | \___ \
| |_| | | | (__| |_| |___) |
 \___/|_|  \___|\___/|____/   v1.0  —  Day 30 Capstone
"""
    print(banner)
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  LLM      : {LLM_MODEL}")
    print(f"  Gesture  : {'✓' if HAS_GESTURE else '✗  (pip install opencv-python mediapipe)'}")
    print(f"  Audio    : {'✓' if HAS_AUDIO   else '✗  (pip install pyaudio numpy)'}")
    print(f"  psutil   : {'✓' if HAS_PSUTIL  else '✗  (pip install psutil)'}")
    print()

    # Start background ISR threads
    _start_daemon(_sensor_thread)
    _start_daemon(_gesture_thread)
    _start_daemon(_system_pulse_thread)

    time.sleep(0.4)   # brief warm-up so first paint has real values

    OrcaOS().run()

    # Post-exit instructions
    print()
    print("=" * 55)
    print("  🐋  30 days. 30 projects. Ship it.")
    print()
    print("  Tag your release:")
    print('    git add -A')
    print('    git commit -m "Day 30: OrcaOS — 30 days complete"')
    print('    git tag v1.0')
    print('    git push origin main --tags')
    print()
    print("  See you in v2.0  🔌")
    print("=" * 55)


if __name__ == "__main__":
    main()
