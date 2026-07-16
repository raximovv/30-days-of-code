import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1

try:
    pa = pyaudio.PyAudio()
    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            device_index = i
            print(f"Using mic: {info['name']}")
            break
    if device_index is None:
        print("ERROR: No microphone found.")
        sys.exit(1)

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
except Exception as e:
    print(f"ERROR opening mic: {e}")
    print("Mac: brew install portaudio && pip install pyaudio")
    print("Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    print("Windows: pip install pipwin && pipwin install pyaudio")
    sys.exit(1)

freqs = np.fft.rfftfreq(CHUNK, d=1.0 / RATE)

BANDS = [
    ("Sub-bass", 20, 60, "#e53935"),
    ("Bass", 60, 250, "#fb8c00"),
    ("Mid", 250, 4000, "#43a047"),
    ("Treble", 4000, 16000, "#1e88e5"),
]

def band_level(fft_magnitudes, freq_array, band):
    _, low, high, _ = band
    mask = (freq_array >= low) & (freq_array < high)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(fft_magnitudes[mask]))

def find_peak_frequency(fft_magnitudes, freq_array, min_freq=50):
    valid = freq_array >= min_freq
    if not valid.any():
        return 0.0, 0.0
    valid_mags = fft_magnitudes.copy()
    valid_mags[~valid] = 0
    peak_idx = np.argmax(valid_mags)
    return float(freq_array[peak_idx]), float(valid_mags[peak_idx])

fig, (ax_spec, ax_bands) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("AudioScope — Day 15 | Speak or play music", fontsize=13)

ax_spec.set_xscale("log")
ax_spec.set_xlim(20, 20000)
ax_spec.set_ylim(0, 0.05)
ax_spec.set_xlabel("Frequency (Hz)")
ax_spec.set_ylabel("Magnitude")
ax_spec.set_title("Frequency Spectrum")
ax_spec.grid(True, which="both", alpha=0.3)

spectrum_line, = ax_spec.plot([], [], color="#4fc3f7", linewidth=1.5)
peak_marker, = ax_spec.plot([], [], "o", color="#ff5252", markersize=7)

band_names = [b[0] for b in BANDS]
band_colors = [b[3] for b in BANDS]
bars = ax_bands.bar(band_names, [0] * len(BANDS), color=band_colors)
ax_bands.set_ylim(0, 0.05)
ax_bands.set_ylabel("Average Magnitude")
ax_bands.set_title("Frequency Bands")
ax_bands.grid(True, axis="y", alpha=0.3)

peak_text = ax_spec.text(
    0.02,
    0.92,
    "Peak: --",
    transform=ax_spec.transAxes,
    fontsize=11,
    fontweight="bold",
    color="white",
    bbox=dict(boxstyle="round", facecolor="#222", alpha=0.8),
)

plt.tight_layout()

def update(frame_num):
    try:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(audio_data, dtype=np.float32)

        if np.max(np.abs(samples)) < 1e-6:
            peak_text.set_text("Peak: no signal")
            spectrum_line.set_data(freqs, np.zeros_like(freqs))
            peak_marker.set_data([], [])
            for bar in bars:
                bar.set_height(0)
            return spectrum_line, peak_marker, peak_text, *bars

        windowed = samples * np.hanning(len(samples))
        fft_result = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_result) / CHUNK

        spectrum_line.set_data(freqs, magnitudes)

        peak_mag_total = max(magnitudes.max(), 0.001)
        ax_spec.set_ylim(0, peak_mag_total * 1.2)

        levels = [band_level(magnitudes, freqs, band) for band in BANDS]
        for bar, level in zip(bars, levels):
            bar.set_height(level)

        max_band = max(max(levels), 0.001)
        ax_bands.set_ylim(0, max_band * 1.2)

        peak_freq, peak_mag = find_peak_frequency(magnitudes, freqs)
        if peak_freq > 0 and peak_mag > 0:
            peak_marker.set_data([peak_freq], [peak_mag])
            peak_text.set_text(f"Peak: {peak_freq:.0f} Hz")
        else:
            peak_marker.set_data([], [])
            peak_text.set_text("Peak: --")

    except Exception as e:
        peak_text.set_text(f"Error: {e}")

    return spectrum_line, peak_marker, peak_text, *bars

print("\nAudioScope is running!")
print(f"Sample rate: {RATE} Hz | FFT size: {CHUNK} | Resolution: {RATE / CHUNK:.1f} Hz/bin")
print(f"Nyquist limit: {RATE / 2} Hz")
print(f"Bands: {', '.join(b[0] for b in BANDS)}")
print("\nTry: whistle, clap, speak, play music")
print("Close the plot window to quit.\n")

try:
    ani = animation.FuncAnimation(
        fig,
        update,
        interval=int(1000 * CHUNK / RATE),
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\nAudioScope ended.")