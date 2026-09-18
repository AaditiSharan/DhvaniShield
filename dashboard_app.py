import asyncio
from contextlib import asynccontextmanager
import ctypes
import json
import queue
import threading
import time
import socket
import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import uvicorn

TARGET_RATE = 16000
VAD_THRESHOLD = 0.55
MIN_SPEECH_SECONDS = 1.2
MAX_SPEECH_SECONDS = 4.0
SILENCE_PAD_FRAMES = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("     VOICE SENTINEL : ZERO-FREEZE TELEMETRY ENGINE")
print(f" Compute Device: {device.upper()}")
print("=" * 60)

# 1. Load Neural Models
print("[*] Loading Neural VAD (Silero)...")
vad_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad_model.to(device).eval()

print("[*] Loading Deepfake Transformer (Wav2Vec2)...")
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
deepfake_model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
deepfake_model.eval()

# Thread-safe queues
telemetry_queue = asyncio.Queue()
raw_audio_queue = queue.Queue()
inference_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    raw_audio_queue.put(indata[:, 0].copy())

def start_safe_audio_stream():
    try:
        ctypes.windll.ole32.CoInitialize(None)
    except Exception:
        pass

    devices = sd.query_devices()
    candidates = []

    for idx, d in enumerate(devices):
        if "CABLE Output" in d["name"] and d["max_input_channels"] > 0:
            hostapi = sd.query_hostapis(d["hostapi"])["name"]
            if "WDM-KS" not in hostapi:
                priority = 0 if "WASAPI" in hostapi else (1 if "DirectSound" in hostapi else 2)
                candidates.append((priority, idx, d["name"], hostapi, int(d["default_samplerate"]), min(d["max_input_channels"], 2)))

    candidates.sort(key=lambda x: x[0])

    for _, idx, name, hostapi, rate, ch in candidates:
        try:
            block = int(rate * 0.032)
            test_stream = sd.InputStream(
                device=idx,
                samplerate=rate,
                channels=ch,
                blocksize=block,
                callback=audio_callback
            )
            test_stream.start()
            print(f"[✔] Linked to Device [{idx}] {name} via {hostapi} ({rate} Hz)")
            return test_stream, rate, block
        except Exception:
            continue

    raise RuntimeError("Could not open CABLE Output stream.")

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Sentinel | Live Defense</title>
    <style>
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --safe: #16a34a;
            --safe-bg: #dcfce7;
            --danger: #dc2626;
            --danger-bg: #fee2e2;
            --accent: #2563eb;
        }

        body {
            margin: 0;
            padding: 32px 16px;
            background-color: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            justify-content: center;
        }

        .dashboard {
            width: 100%;
            max-width: 720px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border);
            padding-bottom: 14px;
        }

        .header h1 {
            margin: 0;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .badge-mode {
            background: #e0e7ff;
            color: var(--accent);
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 6px;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 16px;
            background: #f1f5f9;
            color: var(--text-muted);
            transition: all 0.3s ease;
        }

        .visualizer-card {
            background: #f1f5f9;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 20px;
            position: relative;
        }

        .visualizer-label {
            position: absolute;
            top: 8px;
            left: 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        canvas {
            width: 100%;
            height: 90px;
            display: block;
        }

        .meter-title {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .meter-bar {
            height: 12px;
            background: #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 20px;
        }

        .meter-fill {
            height: 100%;
            width: 0%;
            background: var(--safe);
            transition: width 0.3s ease, background-color 0.3s ease;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }

        .metric-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }

        .metric-label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 700;
        }

        .log-heading {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 10px;
        }

        .log-list {
            max-height: 180px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
        }

        .log-entry {
            display: flex;
            justify-content: space-between;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }

        .log-entry:last-child {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>VOICE SENTINEL</h1>
            <span class="badge-mode">Digital Stream Intercept</span>
        </div>

        <div class="card">
            <div id="statusBadge" class="status-indicator">● Standby &mdash; Monitoring Line</div>

            <div class="visualizer-card">
                <div class="visualizer-label">Live Signal Stream (48 kHz WASAPI)</div>
                <canvas id="scopeCanvas" width="700" height="90"></canvas>
            </div>

            <div class="meter-title">
                <span>Threat Gauge</span>
                <span id="threatScore">0.0% Synthetic</span>
            </div>
            <div class="meter-bar">
                <div id="meterFill" class="meter-fill"></div>
            </div>

            <div class="grid">
                <div class="metric-box">
                    <div class="metric-label">Human Authenticity</div>
                    <div id="humanVal" class="metric-value" style="color: var(--safe);">--%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">AI Clone Likelihood</div>
                    <div id="aiVal" class="metric-value" style="color: var(--danger);">--%</div>
                </div>
            </div>

            <div class="log-heading">Intercept Log</div>
            <div id="logArea" class="log-list"></div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('scopeCanvas');
        const ctx = canvas.getContext('2d');
        const ws = new WebSocket(`ws://${location.host}/ws`);
        const badge = document.getElementById('statusBadge');
        const fill = document.getElementById('meterFill');
        const threatScore = document.getElementById('threatScore');
        const humanVal = document.getElementById('humanVal');
        const aiVal = document.getElementById('aiVal');
        const logArea = document.getElementById('logArea');

        let wavePoints = new Array(64).fill(0);

        function drawWaveform() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.beginPath();
            ctx.strokeStyle = '#cbd5e1';
            ctx.lineWidth = 1;
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();

            ctx.beginPath();
            ctx.strokeStyle = '#2563eb';
            ctx.lineWidth = 2.2;
            ctx.lineJoin = 'round';

            const sliceWidth = canvas.width / (wavePoints.length - 1);
            let x = 0;

            for (let i = 0; i < wavePoints.length; i++) {
                const normalized = wavePoints[i];
                const y = (canvas.height / 2) - (normalized * (canvas.height / 2) * 1.8);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.stroke();
            requestAnimationFrame(drawWaveform);
        }
        requestAnimationFrame(drawWaveform);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'wave') {
                wavePoints = data.points;
            } else if (data.type === 'verdict') {
                const spoof = data.spoof.toFixed(1);
                const bonafide = data.bonafide.toFixed(1);

                humanVal.innerText = `${bonafide}%`;
                aiVal.innerText = `${spoof}%`;
                threatScore.innerText = `${spoof}% Synthetic`;
                fill.style.width = `${spoof}%`;

                let tagColor = "var(--safe)";
                let tagText = "Human Voice";

                if (data.spoof >= 65.0) {
                    badge.style.background = "var(--danger-bg)";
                    badge.style.color = "var(--danger)";
                    badge.innerText = "🚨 WARNING: SYNTHETIC VOICE DETECTED";
                    fill.style.backgroundColor = "var(--danger)";
                    tagColor = "var(--danger)";
                    tagText = "AI Clone";
                } else {
                    badge.style.background = "var(--safe-bg)";
                    badge.style.color = "var(--safe)";
                    badge.innerText = "✔ VERIFIED: AUTHENTIC HUMAN CALLER";
                    fill.style.backgroundColor = "var(--safe)";
                }

                const row = document.createElement('div');
                row.className = 'log-entry';
                row.innerHTML = `
                    <span><strong>${data.time}</strong> &mdash; ${data.duration}s utterance</span>
                    <span style="color: ${tagColor}; font-weight: 600;">${tagText} (${spoof}%)</span>
                `;
                logArea.prepend(row);
            }
        };
    </script>
</body>
</html>
"""

# Dedicated Background Inference Worker (Runs heavy Wav2Vec2 without blocking the audio stream)
def inference_worker(loop):
    while True:
        utterance_16k, timestamp_str = inference_queue.get()
        try:
            inputs = feature_extractor(utterance_16k, sampling_rate=TARGET_RATE, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = deepfake_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]

            spoof_pct = float(probs[1]) * 100
            bonafide_pct = float(probs[0]) * 100

            payload = {
                "type": "verdict",
                "spoof": spoof_pct,
                "bonafide": bonafide_pct,
                "duration": round(len(utterance_16k) / TARGET_RATE, 2),
                "time": timestamp_str
            }
            asyncio.run_coroutine_threadsafe(telemetry_queue.put(payload), loop)
        except Exception as e:
            pass

# High-frequency Audio Loop (Exclusively pumps waveforms and collects utterances)
def audio_worker(loop):
    stream, native_rate, frame_size = start_safe_audio_stream()
    resampler = T.Resample(orig_freq=native_rate, new_freq=TARGET_RATE).to(device)

    with stream:
        speech_buffer = []
        silence_frames = 0
        is_speaking = False
        vad_model.reset_states()
        frame_counter = 0

        while True:
            frame = raw_audio_queue.get()
            frame_counter += 1

            # High-FPS streaming to browser (never blocked)
            if frame_counter % 2 == 0:
                downsampled = frame[::max(1, len(frame) // 64)][:64]
                wave_payload = {
                    "type": "wave",
                    "points": [float(np.clip(p, -1.0, 1.0)) for p in downsampled]
                }
                asyncio.run_coroutine_threadsafe(telemetry_queue.put(wave_payload), loop)

            rms = float(np.sqrt(np.mean(frame**2)))

            if rms < 0.001:
                if is_speaking:
                    silence_frames += 1
                    speech_buffer.append(frame)
                else:
                    vad_model.reset_states()
                    continue
            else:
                tensor_native = torch.from_numpy(frame).to(device)
                tensor_16k = resampler(tensor_native)

                with torch.no_grad():
                    speech_prob = vad_model(tensor_16k, TARGET_RATE).item()

                if speech_prob >= VAD_THRESHOLD:
                    is_speaking = True
                    speech_buffer.append(frame)
                    silence_frames = 0
                elif is_speaking:
                    silence_frames += 1
                    speech_buffer.append(frame)

            if is_speaking:
                total_duration = (len(speech_buffer) * frame_size) / native_rate

                if (silence_frames >= SILENCE_PAD_FRAMES and total_duration >= MIN_SPEECH_SECONDS) or (total_duration >= MAX_SPEECH_SECONDS):
                    full_utterance_native = np.concatenate(speech_buffer)
                    utterance_tensor = torch.from_numpy(full_utterance_native).to(device)
                    utterance_16k = resampler(utterance_tensor).cpu().numpy()

                    # Hand off utterance immediately to inference thread without waiting
                    inference_queue.put((utterance_16k, time.strftime("%H:%M:%S")))

                    speech_buffer = []
                    silence_frames = 0
                    is_speaking = False
                    vad_model.reset_states()

                elif silence_frames >= SILENCE_PAD_FRAMES:
                    speech_buffer = []
                    silence_frames = 0
                    is_speaking = False
                    vad_model.reset_states()

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    threading.Thread(target=audio_worker, args=(loop,), daemon=True).start()
    threading.Thread(target=inference_worker, args=(loop,), daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

@app.websocket("/ws")
async def telemetry_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await telemetry_queue.get()
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass

def get_free_port(preferred=8000):
    for port in range(preferred, preferred + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found")

if __name__ == "_main_":
    port = get_free_port(8000)
    print(f"[*] Binding on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)