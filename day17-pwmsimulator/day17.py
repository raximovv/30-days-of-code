import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
 
# ============================================================
# PWM PARAMETERS
# ============================================================
VCC = 3.3
PWM_FREQ_1 = 1000   # Channel 1: 1 kHz — typical LED dimming
PWM_FREQ_2 = 500    # Channel 2: 500 Hz — slower, e.g. servo range
DISPLAY_CYCLES = 5
SAMPLE_POINTS = 1000
 
current_duty_1 = 50.0
current_duty_2 = 30.0
 
 
# ============================================================
# PWM WAVEFORM GENERATION
# ============================================================
def generate_pwm_wave(duty_percent, freq=PWM_FREQ_1, cycles=DISPLAY_CYCLES, points=SAMPLE_POINTS):
    period = 1.0 / freq
    total_time = cycles * period
    t = np.linspace(0, total_time, points)
    position_in_period = (t % period) / period
    duty_fraction = duty_percent / 100.0
    voltage = np.where(position_in_period < duty_fraction, VCC, 0.0)
    return t * 1000, voltage
 
 
def compute_average_voltage(duty_percent):
    return VCC * (duty_percent / 100.0)
 
 
# ============================================================
# PLOT SETUP
# ============================================================
fig = plt.figure(figsize=(12, 8))
fig.suptitle("PWMSimulator — Day 17 | Dual Channel", fontsize=14, fontweight='bold')
 
# --- Channel 1 waveform ---
ax_wave1 = plt.axes([0.08, 0.62, 0.65, 0.30])
ax_wave1.set_xlim(0, DISPLAY_CYCLES / PWM_FREQ_1 * 1000)
ax_wave1.set_ylim(-0.3, VCC + 0.3)
ax_wave1.set_ylabel("Voltage (V)")
ax_wave1.set_title(f"Channel 1 — {PWM_FREQ_1} Hz")
ax_wave1.grid(True, alpha=0.3)
 
wave1_line, = ax_wave1.plot([], [], color='#43a047', linewidth=2)
avg1_line = ax_wave1.axhline(y=VCC / 2, color='#ff6f00',
                              linestyle='--', linewidth=1.5, label='Average V')
ax_wave1.legend(loc='upper right')
 
# --- Channel 2 waveform ---
ax_wave2 = plt.axes([0.08, 0.30, 0.65, 0.28])
ax_wave2.set_xlim(0, DISPLAY_CYCLES / PWM_FREQ_2 * 1000)
ax_wave2.set_ylim(-0.3, VCC + 0.3)
ax_wave2.set_xlabel("Time (ms)")
ax_wave2.set_ylabel("Voltage (V)")
ax_wave2.set_title(f"Channel 2 — {PWM_FREQ_2} Hz")
ax_wave2.grid(True, alpha=0.3)
 
wave2_line, = ax_wave2.plot([], [], color='#1565c0', linewidth=2)
avg2_line = ax_wave2.axhline(y=VCC * 0.3, color='#ff6f00',
                              linestyle='--', linewidth=1.5, label='Average V')
ax_wave2.legend(loc='upper right')
 
# --- Virtual LED 1 ---
ax_led1 = plt.axes([0.78, 0.62, 0.18, 0.30])
ax_led1.set_xlim(-1.5, 1.5)
ax_led1.set_ylim(-1.5, 1.5)
ax_led1.set_aspect('equal')
ax_led1.axis('off')
ax_led1.set_title("Virtual LED — CH1", fontweight='bold')
 
led1_circle = Circle((0, 0), 1.0, color='#ffeb3b', alpha=0.5)
led1_glow   = Circle((0, 0), 1.3, color='#ffeb3b', alpha=0.2)
led1_base   = plt.Rectangle((-0.6, -1.4), 1.2, 0.3, color='#424242')
ax_led1.add_patch(led1_glow)
ax_led1.add_patch(led1_circle)
ax_led1.add_patch(led1_base)
info1_text = ax_led1.text(0, -1.8, "", ha='center', fontsize=10,
                           fontweight='bold', color='#333')
 
# --- Virtual LED 2 ---
ax_led2 = plt.axes([0.78, 0.30, 0.18, 0.28])
ax_led2.set_xlim(-1.5, 1.5)
ax_led2.set_ylim(-1.5, 1.5)
ax_led2.set_aspect('equal')
ax_led2.axis('off')
ax_led2.set_title("Virtual LED — CH2", fontweight='bold')
 
led2_circle = Circle((0, 0), 1.0, color='#42a5f5', alpha=0.3)
led2_glow   = Circle((0, 0), 1.3, color='#42a5f5', alpha=0.1)
led2_base   = plt.Rectangle((-0.6, -1.4), 1.2, 0.3, color='#424242')
ax_led2.add_patch(led2_glow)
ax_led2.add_patch(led2_circle)
ax_led2.add_patch(led2_base)
info2_text = ax_led2.text(0, -1.8, "", ha='center', fontsize=10,
                           fontweight='bold', color='#333')
 
# --- Sliders ---
slider1_ax = plt.axes([0.15, 0.20, 0.7, 0.04])
duty_slider1 = Slider(slider1_ax, "CH1 Duty (%)", 0, 100,
                      valinit=current_duty_1, valstep=1, color='#0f7173')
 
slider2_ax = plt.axes([0.15, 0.13, 0.7, 0.04])
duty_slider2 = Slider(slider2_ax, "CH2 Duty (%)", 0, 100,
                      valinit=current_duty_2, valstep=1, color='#1565c0')
 
# --- Stats panel ---
stats_ax = plt.axes([0.08, 0.03, 0.84, 0.08])
stats_ax.axis('off')
stats_text = stats_ax.text(
    0.5, 0.5, "",
    ha='center', va='center', fontsize=10,
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='#f0f0f0', pad=1),
    transform=stats_ax.transAxes
)
 
 
# ============================================================
# UPDATE LOGIC
# ============================================================
def _led_state(brightness):
    if brightness == 0:   return "OFF"
    if brightness < 0.3:  return "DIM"
    if brightness < 0.7:  return "MEDIUM"
    return "BRIGHT"
 
def _led1_color(brightness):
    if brightness < 0.3:  return '#ef6c00'
    if brightness < 0.7:  return '#ffa000'
    return '#ffeb3b'
 
def update_display(duty1, duty2):
    t1, v1 = generate_pwm_wave(duty1, PWM_FREQ_1)
    wave1_line.set_data(t1, v1)
    avg_v1 = compute_average_voltage(duty1)
    avg1_line.set_ydata([avg_v1, avg_v1])
 
    t2, v2 = generate_pwm_wave(duty2, PWM_FREQ_2)
    wave2_line.set_data(t2, v2)
    avg_v2 = compute_average_voltage(duty2)
    avg2_line.set_ydata([avg_v2, avg_v2])
 
    b1 = duty1 / 100.0
    led1_circle.set_alpha(0.15 + b1 * 0.85)
    led1_glow.set_alpha(b1 * 0.4)
    led1_circle.set_color(_led1_color(b1))
    info1_text.set_text(_led_state(b1))
 
    b2 = duty2 / 100.0
    led2_circle.set_alpha(0.15 + b2 * 0.85)
    led2_glow.set_alpha(b2 * 0.4)
    info2_text.set_text(_led_state(b2))
 
    p1_us = 1e6 / PWM_FREQ_1
    p2_us = 1e6 / PWM_FREQ_2
    stats = (
        f"CH1: {PWM_FREQ_1} Hz | Duty {duty1:.0f}% | "
        f"HIGH {p1_us * b1:.0f} µs | LOW {p1_us * (1-b1):.0f} µs | Avg {avg_v1:.2f} V        "
        f"CH2: {PWM_FREQ_2} Hz | Duty {duty2:.0f}% | "
        f"HIGH {p2_us * b2:.0f} µs | LOW {p2_us * (1-b2):.0f} µs | Avg {avg_v2:.2f} V"
    )
    stats_text.set_text(stats)
    fig.canvas.draw_idle()
 
 
def on_slider1(val):
    global current_duty_1
    current_duty_1 = val
    update_display(current_duty_1, current_duty_2)
 
def on_slider2(val):
    global current_duty_2
    current_duty_2 = val
    update_display(current_duty_1, current_duty_2)
 
 
duty_slider1.on_changed(on_slider1)
duty_slider2.on_changed(on_slider2)
 
update_display(current_duty_1, current_duty_2)
 
 
# ============================================================
# RUN
# ============================================================
print("\n" + "=" * 55)
print(" 💡 PWMSimulator — Day 17 | Dual Channel")
print("=" * 55)
print()
print(" CH1 slider → 1000 Hz channel")
print(" CH2 slider →  500 Hz channel")
print()
print(" Hardware fact: your CPU does this at ~10,000,000 Hz")
print(" using nothing but a digital output pin.")
print()
print(" Close the plot window to quit.")
print()
 
plt.show()
