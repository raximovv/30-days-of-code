import sys
import time
import collections
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, lfilter
 
RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
CUTOFF_HZ = 0.5
FILTER_ORDER = 2
BREATH_THRESHOLD = 0.005
HISTORY_LENGTH = 500
WINDOW_SEC = 30
 
try:
    pa = pyaudio.PyAudio()
    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
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
    print(f"ERROR opening microphone: {e}")
    print("Mac:   brew install portaudio && pip install pyaudio")
    print("Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    sys.exit(1)
 
effective_rate = RATE / CHUNK
nyquist = effective_rate / 2
normalized_cutoff = min(CUTOFF_HZ / nyquist, 0.95)
b_coeff, a_coeff = butter(FILTER_ORDER, normalized_cutoff, btype='low')
filter_state = np.zeros(max(len(a_coeff), len(b_coeff)) - 1)
 
raw_history = collections.deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
envelope_history = collections.deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
breath_times = []
is_above_threshold = False
current_bpm = 0.0
 
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#0d0d0d')
ax.set_xlim(0, HISTORY_LENGTH)
ax.set_ylim(0, 0.03)
ax.set_title('BreathClock — Day 06', color='white', fontsize=14, fontweight='bold')
ax.set_ylabel('Amplitude', color='#aaaaaa')
ax.set_xlabel('Time →', color='#aaaaaa')
ax.tick_params(colors='#555555')
for spine in ax.spines.values():
    spine.set_edgecolor('#222222')
 
line_raw, = ax.plot([], [], color='#1a3a4a', linewidth=1, label='Raw')
line_env, = ax.plot([], [], color='#00ffcc', linewidth=2, label='Envelope')
threshold_line = ax.axhline(y=BREATH_THRESHOLD, color='#ff4444',
                             linestyle='--', linewidth=1, label='Threshold')
ax.legend(loc='upper right', facecolor='#1a1a1a', labelcolor='white', fontsize=9)
 
bpm_text = ax.text(0.02, 0.92, 'BPM: --', transform=ax.transAxes,
                    fontsize=16, fontweight='bold', color='#00ffcc',
                    verticalalignment='top')
status_text = ax.text(0.02, 0.75, 'Listening...', transform=ax.transAxes,
                       fontsize=12, color='white', verticalalignment='top')
 
plt.tight_layout()
 
 
def compute_bpm():
    now = time.time()
    recent = [t for t in breath_times if now - t < WINDOW_SEC]
    breath_times.clear()
    breath_times.extend(recent)
    if len(recent) < 2:
        return 0.0
    intervals = [recent[i+1] - recent[i] for i in range(len(recent) - 1)]
    avg_interval = sum(intervals) / len(intervals)
    return 60.0 / avg_interval if avg_interval > 0 else 0.0
 
 
def update(frame_num):
    global filter_state, is_above_threshold, current_bpm
 
    try:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(audio_data, dtype=np.float32)
 
        rms = np.sqrt(np.mean(samples ** 2))
        raw_history.append(rms)
 
        filtered, filter_state = lfilter(b_coeff, a_coeff, [rms], zi=filter_state)
        envelope_val = abs(filtered[0])
        envelope_history.append(envelope_val)
 
        if envelope_val > BREATH_THRESHOLD and not is_above_threshold:
            is_above_threshold = True
            breath_times.append(time.time())
            current_bpm = compute_bpm()
        elif envelope_val < BREATH_THRESHOLD * 0.7:
            is_above_threshold = False
 
        x = list(range(HISTORY_LENGTH))
        line_raw.set_data(x, list(raw_history))
        line_env.set_data(x, list(envelope_history))
 
        recent_raw = list(raw_history)[-100:]
        recent_env = list(envelope_history)[-100:]
        raw_max = max(recent_raw) if any(recent_raw) else 0.01
        env_max = max(recent_env) if any(recent_env) else 0.01
        ax.set_ylim(0, max(raw_max, env_max) * 1.5)
 
        threshold_line.set_ydata([BREATH_THRESHOLD])
 
        bpm_text.set_text(f'BPM: {current_bpm:.1f}' if current_bpm > 0 else 'BPM: -- (breathe near mic)')
 
        if is_above_threshold:
            status_text.set_text('● BREATH DETECTED')
            status_text.set_color('#00ffcc')
        else:
            status_text.set_text('Listening...')
            status_text.set_color('#aaaaaa')
 
    except Exception as e:
        status_text.set_text(f'Error: {e}')
 
    return line_raw, line_env, bpm_text, status_text, threshold_line
 
 
print("BreathClock running. Breathe slowly near your mic. Close window to quit.")
 
try:
    ani = animation.FuncAnimation(fig, update,
                                   interval=int(1000 * CHUNK / RATE),
                                   blit=False,
                                   cache_frame_data=False)
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()