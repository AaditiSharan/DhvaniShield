import asyncio
import json
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Audio parameters
SAMPLE_RATE = 16000
WINDOW_DURATION = 2.5
TOTAL_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
BLOCK_SIZE = int(SAMPLE_RATE * 0.4)
SILENCE_THRESHOLD = 0.012

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Initializing Sentinel Engine on {device.upper()}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Streaming state
audio_queue = queue.Queue()
audio_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
session_active = False

# Temporal smoothing
smoothed_fake_pct = 0.0
smoothed_real_pct = 100.0

connected_clients = set()

def audio_worker():
    def callback(indata, frames, time_info, status):
        if session_active:
            audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=callback):
        while True:
            sd.sleep(100)

threading.Thread(target=audio_worker, daemon=True).start()

def run_inference(waveform):
    global smoothed_fake_pct, smoothed_real_pct
    rms = float(np.sqrt(np.mean(waveform**2)))

    if rms < SILENCE_THRESHOLD:
        smoothed_fake_pct = max(0.0, smoothed_fake_pct * 0.85)
        smoothed_real_pct = min(100.0, 100.0 - smoothed_fake_pct)
        return {
            "is_speech": False,
            "rms": round(rms, 4),
            "fake_score": round(smoothed_fake_pct, 1),
            "real_score": round(smoothed_real_pct, 1),
            "waveform": waveform[::160].tolist()
        }

    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    raw_fake = float(probs[1]) * 100
    raw_real = float(probs[0]) * 100

    # Exponential Moving Average (EMA) smoothing
    ALPHA = 0.35
    smoothed_fake_pct = (ALPHA * raw_fake) + ((1.0 - ALPHA) * smoothed_fake_pct)
    smoothed_real_pct = 100.0 - smoothed_fake_pct

    return {
        "is_speech": True,
        "rms": round(rms, 4),
        "fake_score": round(smoothed_fake_pct, 1),
        "real_score": round(smoothed_real_pct, 1),
        "waveform": waveform[::160].tolist()
    }

async def background_analyzer():
    global audio_buffer
    while True:
        try:
            if session_active:
                chunk = audio_queue.get_nowait()
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk

                if connected_clients:
                    result = run_inference(audio_buffer)
                    payload = json.dumps({"type": "stream", "data": result})
                    for ws in list(connected_clients):
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            connected_clients.discard(ws)
            else:
                await asyncio.sleep(0.1)
        except queue.Empty:
            await asyncio.sleep(0.04)
        except Exception:
            await asyncio.sleep(0.04)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_analyzer())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    global session_active, audio_buffer, smoothed_fake_pct, smoothed_real_pct
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            cmd = json.loads(msg)
            if cmd.get("action") == "start":
                audio_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
                smoothed_fake_pct = 0.0
                smoothed_real_pct = 100.0
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
                session_active = True
            elif cmd.get("action") == "stop":
                session_active = False
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)

@app.get("/")
def get_dashboard():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AegisVoice // Real-Time Call Sentinel</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen font-sans flex flex-col justify-between">

        <header class="border-b border-slate-800 bg-slate-900/80 px-8 py-4 flex justify-between items-center"> 
            <div class="flex items-center space-x-3">
                <div class="w-3 h-3 rounded-full bg-cyan-400"></div>
                <h1 class="text-lg font-bold uppercase tracking-wider text-cyan-400">AegisVoice Call Defense</h1>
            </div>
            <div id="micIndicator" class="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                <span class="w-2 h-2 rounded-full bg-slate-500"></span>
                <span>MIC INACTIVE</span>
            </div>
        </header>

        <main class="max-w-4xl w-full mx-auto p-6 space-y-6">

            <div class="bg-slate-900 p-6 rounded-2xl border border-slate-800 flex items-center justify-between shadow-lg">
                <div class="flex items-center space-x-4">
                    <button id="sessionBtn" onclick="toggleSession()" class="px-6 py-3 rounded-xl font-semibold text-sm transition bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg">
                        ▶ Start Call Monitoring
                    </button>
                    <div>
                        <p class="text-xs text-slate-400 uppercase tracking-wider">Session Duration</p>
                        <p id="timer" class="text-xl font-bold font-mono text-slate-200">00:00</p>
                    </div>
                </div>

                <button onclick="triggerChallenge()" class="px-4 py-2.5 rounded-xl border border-indigo-500/40 bg-indigo-950/40 hover:bg-indigo-900/50 text-indigo-300 text-xs font-semibold uppercase tracking-wider transition">
                    ⚡ Acoustic Liveness Challenge
                </button>
            </div>

            <div id="challengeBox" class="hidden p-4 rounded-xl bg-indigo-950/30 border border-indigo-700 text-sm">
                <span class="text-xs uppercase font-bold text-indigo-400 block mb-1">Instruct Caller to Read Aloud:</span>
                <p id="challengeText" class="font-mono text-slate-200"></p>
            </div>

            <div id="monitorCard" class="p-8 rounded-2xl bg-slate-900 border border-slate-800 space-y-6 transition-all duration-300">
                <div class="flex justify-between items-center">
                    <div>
                        <span class="text-xs font-mono uppercase tracking-widest text-slate-400">Live Call Status</span>
                        <h2 id="liveStatusText" class="text-2xl font-black text-slate-400">READY TO MONITOR</h2>
                    </div>
                    <span id="threatBadge" class="px-4 py-1 rounded-full text-xs font-bold uppercase bg-slate-800 text-slate-400">STANDBY</span>
                </div>

                <div class="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <canvas id="waveformCanvas" width="800" height="90" class="w-full h-24"></canvas>
                </div>

                <div class="grid grid-cols-2 gap-6">
                    <div class="p-5 rounded-xl bg-slate-950 border border-slate-800">
                        <div class="flex justify-between mb-2">
                            <span class="text-xs uppercase font-semibold text-slate-400">Human Authenticity</span>
                            <span id="realValue" class="text-xl font-bold font-mono text-emerald-400">--%</span>
                        </div>
                        <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                            <div id="realBar" class="bg-emerald-500 h-full w-0 transition-all duration-300"></div>
                        </div>
                    </div>

                    <div class="p-5 rounded-xl bg-slate-950 border border-slate-800">
                        <div class="flex justify-between mb-2">
                            <span class="text-xs uppercase font-semibold text-slate-400">AI Clone Probability</span>
                            <span id="fakeValue" class="text-xl font-bold font-mono text-red-400">--%</span>
                        </div>
                        <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                            <div id="fakeBar" class="bg-red-500 h-full w-0 transition-all duration-300"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="summaryModal" class="hidden p-6 rounded-2xl bg-slate-900 border border-slate-700 space-y-4 shadow-2xl">
                <h3 class="text-lg font-bold text-slate-200 border-b border-slate-800 pb-3">Session Forensic Summary</h3>
                <div class="grid grid-cols-3 gap-4 text-center">
                    <div class="p-4 rounded-xl bg-slate-950">
                        <p class="text-xs text-slate-400">Call Duration</p>
                        <p id="sumDuration" class="text-xl font-bold font-mono text-slate-200">0s</p>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-950">
                        <p class="text-xs text-slate-400">Peak AI Spoof Score</p>
                        <p id="sumPeak" class="text-xl font-bold font-mono text-red-400">0.0%</p>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-950">
                        <p class="text-xs text-slate-400">Final Verdict</p>
                        <p id="sumVerdict" class="text-sm font-bold mt-1">SAFE</p>
                    </div>
                </div>
                <button onclick="resetSession()" class="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold uppercase tracking-wider transition">
                    Done / New Call
                </button>
            </div>
        </main>

        <footer class="text-center py-4 text-xs font-mono text-slate-500">
            AegisVoice Real-Time Audio Interception // Local Processing
        </footer>

        <script>
            let ws = null;
            const canvas = document.getElementById('waveformCanvas');
            const ctx = canvas.getContext('2d');

            let isRunning = false;
            let timerInterval = null;
            let secondsElapsed = 0;
            let peakFake = 0.0;

            const challenges = [
                "Quickly repeat: 'Seven silver swans swam swiftly Southward'",
                "Say: 'Blue breeze blew bright bricks' without pausing",
                "Spell your primary email backward right now",
                "Pronounce immediately: 'Unique New York, Pacific Specific'"
            ];

            function connectWS() {
                ws = new WebSocket(`ws://${location.host}/ws/audio`);
                ws.onmessage = (event) => {
                    if (!isRunning) return;
                    const payload = JSON.parse(event.data);
                    const d = payload.data;

                    if (d.fake_score > peakFake) peakFake = d.fake_score;

                    document.getElementById('realValue').innerText = d.real_score + '%';
                    document.getElementById('fakeValue').innerText = d.fake_score + '%';
                    document.getElementById('realBar').style.width = d.real_score + '%';
                    document.getElementById('fakeBar').style.width = d.fake_score + '%';

                    const card = document.getElementById('monitorCard');
                    const title = document.getElementById('liveStatusText');
                    const badge = document.getElementById('threatBadge');

                    if (d.fake_score >= 65.0) {
                        card.className = 'p-8 rounded-2xl bg-red-950/40 border border-red-600 space-y-6 shadow-[0_0_30px_rgba(220,38,38,0.3)] transition-all duration-300';
                        title.innerText = '🚨 WARNING: AI VOICE CLONE ACTIVE';
                        title.className = 'text-2xl font-black text-red-400 animate-pulse';
                        badge.innerText = 'SPOOF DETECTED';
                        badge.className = 'px-4 py-1 rounded-full text-xs font-bold uppercase bg-red-600 text-white';
                    } else if (d.is_speech) {
                        card.className = 'p-8 rounded-2xl bg-emerald-950/30 border border-emerald-600 space-y-6 shadow-[0_0_25px_rgba(16,185,129,0.2)] transition-all duration-300';
                        title.innerText = 'VERIFIED AUTHENTIC HUMAN SPEECH';
                        title.className = 'text-2xl font-black text-emerald-400';
                        badge.innerText = 'AUTHENTIC';
                        badge.className = 'px-4 py-1 rounded-full text-xs font-bold uppercase bg-emerald-600 text-white';
                    } else {
                        title.innerText = 'LISTENING // AMBIENT ROOM';
                        title.className = 'text-2xl font-black text-slate-400';
                        badge.innerText = 'SILENCE / LISTENING';
                        badge.className = 'px-4 py-1 rounded-full text-xs font-bold uppercase bg-slate-800 text-slate-400';
                    }

                    drawWave(d.waveform, d.fake_score >= 65.0);
                };
                ws.onclose = () => { setTimeout(connectWS, 1000); };
            }

            connectWS();

            function toggleSession() {
                if (!isRunning) {
                    isRunning = true;
                    secondsElapsed = 0;
                    peakFake = 0.0;
                    document.getElementById('summaryModal').classList.add('hidden');
                    document.getElementById('sessionBtn').innerText = '⏹ End Call & Review';
                    document.getElementById('sessionBtn').className = 'px-6 py-3 rounded-xl font-semibold text-sm transition bg-red-600 hover:bg-red-500 text-white shadow-lg';
                    
                    document.getElementById('micIndicator').innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span><span class="text-emerald-400">ACTIVE LISTENING</span>';
                    document.getElementById('liveStatusText').innerText = 'ANALYZING INCOMING CALL...';
                    document.getElementById('liveStatusText').className = 'text-2xl font-black text-slate-300';
                    document.getElementById('threatBadge').innerText = 'INSPECTION ACTIVE';

                    timerInterval = setInterval(() => {
                        secondsElapsed++;
                        const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
                        const secs = String(secondsElapsed % 60).padStart(2, '0');
                        document.getElementById('timer').innerText = `${mins}:${secs}`;
                    }, 1000);

                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ action: "start" }));
                    }
                } else {
                    isRunning = false;
                    clearInterval(timerInterval);
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ action: "stop" }));
                    }

                    document.getElementById('sessionBtn').innerText = '▶ Start Call Monitoring';
                    document.getElementById('sessionBtn').className = 'px-6 py-3 rounded-xl font-semibold text-sm transition bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg';
                    
                    document.getElementById('micIndicator').innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-500"></span><span>MIC INACTIVE</span>';
                    
                    document.getElementById('summaryModal').classList.remove('hidden');
                    document.getElementById('sumDuration').innerText = secondsElapsed + 's';
                    document.getElementById('sumPeak').innerText = peakFake.toFixed(1) + '%';
                    
                    const verdictElem = document.getElementById('sumVerdict');
                    if (peakFake >= 65.0) {
                        verdictElem.innerText = '🚨 HIGH THREAT (AI CLONE DETECTED)';
                        verdictElem.className = 'text-sm font-bold mt-1 text-red-400';
                    } else {
                        verdictElem.innerText = '✅ SAFE (AUTHENTIC HUMAN)';
                        verdictElem.className = 'text-sm font-bold mt-1 text-emerald-400';
                    }
                }
            }

            function triggerChallenge() {
                const box = document.getElementById('challengeBox');
                box.classList.remove('hidden');
                document.getElementById('challengeText').innerText = challenges[Math.floor(Math.random() * challenges.length)];
            }

            function resetSession() {
                document.getElementById('summaryModal').classList.add('hidden');
                document.getElementById('challengeBox').classList.add('hidden');
                document.getElementById('timer').innerText = '00:00';
                document.getElementById('realValue').innerText = '--%';
                document.getElementById('fakeValue').innerText = '--%';
                document.getElementById('realBar').style.width = '0%';
                document.getElementById('fakeBar').style.width = '0%';
                document.getElementById('liveStatusText').innerText = 'READY TO MONITOR';
                document.getElementById('threatBadge').innerText = 'STANDBY';
                document.getElementById('threatBadge').className = 'px-4 py-1 rounded-full text-xs font-bold uppercase bg-slate-800 text-slate-400';
                document.getElementById('monitorCard').className = 'p-8 rounded-2xl bg-slate-900 border border-slate-800 space-y-6 transition-all duration-300';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            function drawWave(samples, isSpoof) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.lineWidth = 2;
                ctx.strokeStyle = isSpoof ? '#ef4444' : '#10b981';
                ctx.beginPath();
                const step = canvas.width / samples.length;
                let x = 0;
                for (let i = 0; i < samples.length; i++) {
                    const y = (canvas.height / 2) + (samples[i] * 110);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                    x += step;
                }
                ctx.stroke();
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)