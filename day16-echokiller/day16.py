import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

SAMPLE_RATE = 16000
FILTER_ORDER = 64
LEARNING_RATE = 0.02
ECHO_DELAY_MS = 150
ECHO_DECAY = 0.5

def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-9)

def resample_audio(x, orig_sr, target_sr):
    if orig_sr == target_sr:
        return x.astype(np.float32)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    y = sig.resample_poly(x, up, down)
    return y.astype(np.float32)

def generate_synthetic_signal(duration=3.0, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    x = np.zeros_like(t)

    bursts = [0.2, 0.9, 1.6, 2.3]
    for start in bursts:
        length = 0.45
        mask = (t >= start) & (t < start + length)
        f0 = np.random.uniform(110, 190)
        part = np.zeros_like(t)
        for h in range(1, 5):
            part += np.sin(2 * np.pi * f0 * h * t) / h
        env = np.zeros_like(t)
        env[mask] = np.sin(np.pi * (t[mask] - start) / length)
        x += part * env

    x = normalize(x) * 0.6
    return x.astype(np.float32)

def load_audio():
    wav_files = [f for f in os.listdir(".") if f.lower().endswith(".wav")]
    wav_files = [f for f in wav_files if f not in ["echoed_input.wav", "cleaned_output.wav"]]

    if wav_files and HAS_SOUNDFILE:
        path = wav_files[0]
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        data = resample_audio(data, sr, SAMPLE_RATE)
        data = normalize(data) * 0.6
        return data.astype(np.float32), path

    return generate_synthetic_signal(), None

def add_echo(x, delay_ms=ECHO_DELAY_MS, decay=ECHO_DECAY, sample_rate=SAMPLE_RATE):
    delay_samples = int(sample_rate * delay_ms / 1000)
    y = np.copy(x)
    if delay_samples < len(x):
        y[delay_samples:] += decay * x[:-delay_samples]
    return y.astype(np.float32)

def adaptive_fir_filter(reference, mixed, filter_order=FILTER_ORDER, mu=LEARNING_RATE):
    n = len(mixed)
    w = np.zeros(filter_order, dtype=np.float32)
    e = np.zeros(n, dtype=np.float32)
    xbuf = np.zeros(filter_order, dtype=np.float32)

    for i in range(n):
        xbuf[1:] = xbuf[:-1]
        xbuf[0] = reference[i]
        y_hat = np.dot(w, xbuf)
        err = mixed[i] - y_hat
        e[i] = err
        power = np.dot(xbuf, xbuf) + 1e-6
        w += (mu / power) * err * xbuf

    return e, w

def plot_results(echoed, cleaned, coeffs, sample_rate=SAMPLE_RATE):
    echoed_n = normalize(echoed)
    cleaned_n = normalize(cleaned)

    t = np.arange(len(echoed_n)) / sample_rate
    tap_t = np.arange(len(coeffs)) / sample_rate * 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t, echoed_n, label="Before")
    axes[0].plot(t, cleaned_n, label="After", alpha=0.85)
    axes[0].set_title("Before vs After Waveforms")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Normalized Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(tap_t, coeffs, width=(1000 / sample_rate) * 0.8)
    axes[1].set_title("Learned FIR Coefficients")
    axes[1].set_xlabel("Delay (ms)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    reference, loaded_path = load_audio()

    if loaded_path:
        print(f"Loaded audio file: {loaded_path}")
    else:
        print("No input WAV found, using generated synthetic signal.")

    echoed = add_echo(reference)
    start = time.time()
    cleaned, coeffs = adaptive_fir_filter(reference, echoed)
    elapsed = time.time() - start

    echoed_energy = np.mean(echoed ** 2)
    cleaned_energy = np.mean(cleaned ** 2)
    reduction_db = 10 * np.log10((echoed_energy + 1e-9) / (cleaned_energy + 1e-9))

    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Filter order: {FILTER_ORDER}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Echo delay: {ECHO_DELAY_MS} ms")
    print(f"Done in {elapsed:.2f}s")
    print(f"Echoed energy: {echoed_energy:.6f}")
    print(f"Cleaned energy: {cleaned_energy:.6f}")
    print(f"Reduction: {reduction_db:.2f} dB")

    if HAS_SOUNDFILE:
        sf.write("echoed_input.wav", normalize(echoed), SAMPLE_RATE)
        sf.write("cleaned_output.wav", normalize(cleaned), SAMPLE_RATE)
        print("Saved echoed_input.wav")
        print("Saved cleaned_output.wav")

    if HAS_SOUNDDEVICE:
        try:
            print("Playing echoed audio...")
            sd.play(normalize(echoed), SAMPLE_RATE)
            sd.wait()
            print("Playing cleaned audio...")
            sd.play(normalize(cleaned), SAMPLE_RATE)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()

    plot_results(echoed, cleaned, coeffs)

if __name__ == "__main__":
    main()