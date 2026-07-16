import time
import sys
import threading
import subprocess

try:
    from pynput import keyboard
except ImportError:
    print("pip install pynput")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.text import Text
    console = Console()
except ImportError:
    print("pip install rich")
    sys.exit(1)

MODEL = "qwen2.5:3b"

MORSE_TABLE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6",
    "--...": "7", "---..": "8", "----.": "9", ".-.-.-": ".",
    "--..--": ",", "..--..": "?", "-..-.": "/", "---...": ":",
}

REVERSE_TABLE = {v: k for k, v in MORSE_TABLE.items()}

DASH_MULTIPLIER = 2.5
LETTER_GAP_MULTIPLIER = 3.0
WORD_GAP_MULTIPLIER = 7.0
DEFAULT_DOT_MS = 300


class AutoBaud:
    def __init__(self, n=5):
        self.n = n
        self.durations = []
        self.calibrated = False
        self.dot_ms = None

    def record(self, ms):
        if self.calibrated:
            return
        self.durations.append(ms)
        count = len(self.durations)
        if count < self.n:
            console.print(f"\r  [dim]Calibrating {count}/{self.n}...[/dim]", end="")
        else:
            self._run()

    def _run(self):
        s = sorted(self.durations)
        if len(s) == 1:
            self.dot_ms = s[0]
            self.calibrated = True
            console.print(f"\n[green]Calibrated: dot ≈ {self.dot_ms:.0f}ms[/green]")
            return

        gaps = [(s[i + 1] - s[i], i) for i in range(len(s) - 1)]
        biggest_gap, split = max(gaps, key=lambda x: x[0])

        if biggest_gap < 50:
            self.dot_ms = s[0] * 1.2
            console.print(
                f"\n[yellow]Weak calibration (gap={biggest_gap:.0f}ms). "
                f"dot ≈ {self.dot_ms:.0f}ms — mix dots and dashes next time.[/yellow]"
            )
        else:
            dots = s[: split + 1]
            self.dot_ms = sum(dots) / len(dots)
            console.print(
                f"\n[green]Calibrated: dot ≈ {self.dot_ms:.0f}ms  "
                f"dash ≥ {self.dot_ms * DASH_MULTIPLIER:.0f}ms[/green]"
            )

        self.calibrated = True

    def get_dot(self):
        return self.dot_ms if self.calibrated else DEFAULT_DOT_MS


auto_baud = AutoBaud(n=5)


def decode_symbol(seq):
    return MORSE_TABLE.get(seq, "?")


def encode_to_morse(text):
    parts = []
    for ch in text.upper():
        if ch == " ":
            parts.append("/")
        elif ch in REVERSE_TABLE:
            parts.append(REVERSE_TABLE[ch])
    return " ".join(parts)


def render_morse(morse_str):
    console.print(f"\n[bold]Morse:[/bold]")
    for word in morse_str.split(" / "):
        line = ""
        for letter in word.split():
            for sym in letter:
                if sym == ".":
                    line += "[cyan]·[/cyan]"
                elif sym == "-":
                    line += "[cyan]—[/cyan]"
            line += "  "
        console.print("  " + line.strip())
    console.print(f"[dim]{morse_str}[/dim]\n")


def ask_llm(text):
    prompt = (
        f"Someone sent you this via Morse code: '{text.strip()}'. "
        f"Reply in one short friendly sentence."
    )
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() or "..."
    except Exception:
        return "CANNOT REACH LOCAL MODEL"


class State:
    IDLE = 0
    DOWN = 1
    BETWEEN = 2


class Decoder:
    def __init__(self):
        self.state = State.IDLE
        self.down_at = None
        self.up_at = None
        self.seq = ""
        self.word = ""
        self.text = ""
        self.lock = threading.Lock()
        self._alive = True
        self._word_queue = []
        t = threading.Thread(target=self._watcher, daemon=True)
        t.start()

    def press(self):
        with self.lock:
            self.down_at = time.time()
            self.up_at = None
            self.state = State.DOWN

    def release(self):
        with self.lock:
            if self.down_at is None:
                return
            ms = (time.time() - self.down_at) * 1000
            self.up_at = time.time()
            self.down_at = None
            auto_baud.record(ms)
            dot = auto_baud.get_dot()
            self.seq += "." if ms < dot * DASH_MULTIPLIER else "-"
            self.state = State.BETWEEN

    def _watcher(self):
        while self._alive:
            time.sleep(0.04)
            with self.lock:
                if self.state != State.BETWEEN or self.up_at is None:
                    continue
                silence = (time.time() - self.up_at) * 1000
                dot = auto_baud.get_dot()

                if silence >= dot * WORD_GAP_MULTIPLIER:
                    if self.seq:
                        self.word += decode_symbol(self.seq)
                        self.seq = ""
                    if self.word:
                        self.text += self.word + " "
                        self._word_queue.append(self.word)
                        self.word = ""
                    self.state = State.IDLE

                elif silence >= dot * LETTER_GAP_MULTIPLIER:
                    if self.seq:
                        self.word += decode_symbol(self.seq)
                        self.seq = ""
                    self.state = State.IDLE

    def pop_words(self):
        with self.lock:
            out = self._word_queue[:]
            self._word_queue.clear()
            return out

    def snapshot(self):
        with self.lock:
            return self.text, self.word, self.seq

    def reset_text(self):
        with self.lock:
            self.text = ""

    def stop(self):
        self._alive = False


def main():
    console.print("\n[bold cyan]MorseDecoder — Day 28[/bold cyan]")
    console.print("[dim]SPACE = tap  |  ESC = quit[/dim]")
    console.print("[dim]First 5 taps calibrate timing. Mix dots & dashes.[/dim]\n")
    console.print("  Morse cheat sheet:")
    console.print("  E=.  T=-  I=..  A=.-  S=...  O=---  M=--  N=-.\n")

    decoder = Decoder()
    quit_evt = threading.Event()
    last_text_len = 0
    pending_llm = False

    def on_press(key):
        if key == keyboard.Key.space:
            decoder.press()
        elif key == keyboard.Key.esc:
            quit_evt.set()
            return False

    def on_release(key):
        if key == keyboard.Key.space:
            decoder.release()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    console.print("[green]Ready. Start tapping.[/green]\n")

    try:
        while not quit_evt.is_set():
            time.sleep(0.08)

            text, word, seq = decoder.snapshot()
            parts = []
            if text:
                parts.append(f"[white]{text}[/white]")
            if word:
                parts.append(f"[yellow]{word}[/yellow]")
            if seq:
                parts.append(f"[cyan]{seq}[/cyan]")
            if parts:
                console.print("\r  " + " ".join(parts) + "   ", end="")

            new_words = decoder.pop_words()
            for w in new_words:
                console.print(f"\n[green]✓ '{w}'[/green]")

            if len(text) > last_text_len:
                last_text_len = len(text)

            word_count = len(text.strip().split()) if text.strip() else 0
            if word_count >= 2 and not pending_llm:
                pending_llm = True
                snapshot = text.strip()
                console.print(f"\n\n[bold]→ Sending to LLM:[/bold] '{snapshot}'")
                response = ask_llm(snapshot)
                console.print(f"[bold magenta]LLM:[/bold magenta] {response}")
                morse = encode_to_morse(response)
                render_morse(morse)
                decoder.reset_text()
                last_text_len = 0
                pending_llm = False

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        decoder.stop()
        final, _, _ = decoder.snapshot()
        if final.strip():
            console.print(f"\n[bold]Session text:[/bold] '{final.strip()}'")
        console.print("\n[dim]Done. Day 29 awaits.[/dim]")


if __name__ == "__main__":
    main()
