# 30 Days of Code

30 small Python projects, one per day, built for [Orcas v1.0](https://buildcored.com/), the 30 day challenge run by Buildcored, an open source builder community from Tashkent. The rule was simple: ship something that works every single day, finished or not. These are day by day builds, not polished products.

The month ran in four themes: body as input (days 1 to 7), local AI on the laptop (days 8 to 14), signals and systems (days 15 to 21), and full systems (days 22 to 30). Most days use the webcam, the microphone, or a local LLM through Ollama. Common libraries: OpenCV, MediaPipe, NumPy, Matplotlib. Each project lives in its own folder with its code and original notes, including run instructions.

| Day | Project | What it does |
|-----|---------|--------------|
| 01 | [RockLook](day01-rocklook) | Music plays when you look down and pauses when you look up. Webcam gaze detection. |
| 02 | [AirCanvas](day02-aircanvas) | Draw in the air with your finger. Hand tracking with OpenCV and MediaPipe. |
| 03 | [VolumeKnuckle](day03-volumeknuckle) | Control system volume with your fist height through the webcam. |
| 04 | [BlinkLock](day04-blinklock) | Lock your screen with an eye blink. Webcam blink detection. |
| 05 | [FaceEQ](day05-faceeq) | Your head becomes a playback control. Turn to seek, tilt to change speed. MediaPipe face tracking. |
| 06 | [BreathClock](day06-breathclock) | Detects your breathing rate in real time from the laptop microphone using a Butterworth low pass filter. |
| 07 | [KeyboardOscilloscope](day07-keyboardoscilloscope) | Keyboard as a synthesizer. Each key plays a sine wave, and chords show superposition on a live oscilloscope. |
| 08 | [PocketAgent](day08-pocketagent) | CLI agent running a 3B parameter LLM fully on the laptop. No cloud, no API keys. |
| 09 | [WhisperDesk](day09-whisperdesk) | Offline speech to text from the microphone. |
| 10 | [TerminalBrain](day10-terminalbrain) | Terminal wrapper that catches errors in real time and asks a local LLM (Ollama) to suggest fixes. |
| 11 | [MoodSynth](day11-moodsynth) | Turns a plain English mood description into live ambient audio. |
| 12 | [SnapAnnotator](day12-snapannotator) | Captures a webcam frame and analyzes it with the moondream vision model via Ollama. |
| 13 | [DailyDebrief](day13-dailydebrief) | Local developer activity summarizer. |
| 14 | [RegisterBot](day14-registerbot) | Tiny CPU simulator in pure Python. Fetch, decode, execute cycle with 8 registers and an ALU. |
| 15 | [AudioScope](day15-audioscope) | Live microphone spectrum analyzer using FFT. |
| 16 | [EchoKiller](day16-echokiller) | Echo reduction with an adaptive FIR filter. |
| 17 | [PWMSimulator](day17-pwmsimulator) | Interactive two channel PWM simulator with live square waves and virtual LEDs. NumPy, Matplotlib, Tkinter. |
| 18 | [DepthMapper](day18-depthmapper) | Live monocular depth estimation from the webcam using MiDaS, with a depth heatmap and point cloud export. |
| 19 | [I2CPlayground](day19-i2cplayground) | Simulates the I2C protocol in Python. |
| 20 | [SensorLogger](day20-sensorlogger) | Treats the laptop as a multi channel sensor array. |
| 21 | [UDPOscilloscope](day21-udposcilloscope) | A simulated sensor streams data over UDP and a receiver plots it live. |
| 22 | [CircuitWhisperer](day22-circuitwhisperer) | Interprets hand drawn circuit schematics with a local vision model. |
| 23 | [ObjectFollower](day23-objectfollower) | Real time color based object tracking with OpenCV. |
| 24 | [HardwareTA](day24-hardwareta) | Local retrieval augmented generation system for hardware datasheet PDFs. |
| 25 | [FirmwarePatcher](day25-firmwarepatcher) | Local LLM analysis plus hex inspection on simulated firmware. |
| 26 | [CacheSim](day26-cachesim) | Two level CPU cache simulator (L1 and L2) with LRU eviction. |
| 27 | [SignalFlags](day27-signalflags) | Real time semaphore gesture recognition through the webcam. |
| 28 | [MorseDecoder](day28-morsedecoder) | Tap Morse code on the spacebar. Timing is decoded live and sent to a local LLM. |
| 29 | [SilentAssistant](day29-silentassistant) | The webcam watches your mouth, a vision model describes it, a local LLM answers, and the answer is spoken back. |
| 30 | [OrcaOS](day30-orcaos) | One terminal dashboard combining the challenge projects. Textual, MediaPipe, Ollama, psutil. |
