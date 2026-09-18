import asyncio
from contextlib import asynccontextmanager
import csv
import ctypes
import json
import os
import queue
import subprocess
import threading
import time
import wave
import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import uvicorn
import socket
from silero_vad import load_silero_vad

TARGET_RATE = 16000
VAD_THRESHOLD = 0.55
MIN_SPEECH_SECONDS = 1.2
MAX_SPEECH_SECONDS = 4.0
SILENCE_PAD_FRAMES = 16

EVIDENCE_DIR = os.path.join(os.getcwd(), "evidence")
LOG_FILE = os.path.join(EVIDENCE_DIR, "forensic_audit_log.csv")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Classification", "Spoof_Score_Pct", "Bonafide_Score_Pct", "Duration_Sec", "Evidence_File"])

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("     DHVANISHIELD (ध्वनि-शील्ड) : POST-CALL INCIDENT ESCALATION")
print(f" Compute Engine:  {device.upper()}")
print(f" Local Vault:     {EVIDENCE_DIR}")
print("=" * 65)

# Models
print("[*] Loading Silero Neural VAD...")
#vad_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad_model = load_silero_vad()
vad_model.to(device).eval()

print("[*] Loading Wav2Vec2 Deepfake Transformer...")
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
deepfake_model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
deepfake_model.eval()

telemetry_queue = asyncio.Queue()
raw_audio_queue = queue.Queue()

recording_lock = threading.Lock()
is_manual_recording = False
manual_record_buffer = []

def audio_callback(indata, frames, time_info, status):
    chunk = indata[:, 0].copy()
    raw_audio_queue.put(chunk)
    with recording_lock:
        if is_manual_recording:
            manual_record_buffer.append(chunk)

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

def archive_utterance(utterance_16k, spoof_pct, bonafide_pct, duration, timestamp_str):
    filename = f"INCIDENT_{time.strftime('%Y%m%d_%H%M%S')}_{int(spoof_pct)}PCT.wav"
    filepath = os.path.join(EVIDENCE_DIR, filename)

    scaled = np.int16(np.clip(utterance_16k, -1.0, 1.0) * 32767)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(scaled.tobytes())

    classification = "AI_CLONE_SPOOF" if spoof_pct >= 65.0 else "BONAFIDE_HUMAN"
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp_str, classification, f"{spoof_pct:.2f}", f"{bonafide_pct:.2f}", f"{duration:.2f}", filename])

    return filename

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DhvaniShield (ध्वनि-शील्ड) | Forensic Defense</title>
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
            --navy: #1e293b;
        }

        body {
            margin: 0;
            padding: 24px 16px;
            background-color: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            justify-content: center;
        }

        .dashboard {
            width: 100%;
            max-width: 860px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border);
            padding-bottom: 12px;
        }

        .header h1 {
            margin: 0;
            font-size: 19px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .actions {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .btn {
            border: 1px solid var(--border);
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }

        .btn-white {
            background: #ffffff;
            color: var(--text-main);
        }
        .btn-white:hover { background: #f1f5f9; }

        .btn-rec {
            background: #ffffff;
            color: var(--danger);
            border-color: #fca5a5;
        }
        .btn-rec.recording {
            background: var(--danger);
            color: #ffffff;
            border-color: var(--danger);
            animation: pulse-border 1.5s infinite;
        }

        .btn-end-call {
            background: #0f172a;
            color: #ffffff;
            border-color: #0f172a;
        }
        .btn-end-call:hover { background: #334155; }

        @keyframes pulse-border {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.75; }
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12.5px;
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
            margin-bottom: 18px;
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
            height: 85px;
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
            margin-bottom: 18px;
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
            gap: 14px;
            margin-bottom: 18px;
        }

        .metric-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }

        .metric-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 26px;
            font-weight: 700;
        }

        .cybercell-card {
            background: #f8fafc;
            border: 1.5px solid #cbd5e1;
            border-radius: 10px;
            padding: 14px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .cc-left {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .cc-badge {
            font-size: 10px;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .cc-title {
            font-size: 13.5px;
            font-weight: 700;
            color: var(--navy);
        }

        .cc-actions {
            display: flex;
            gap: 8px;
        }

        .btn-call-1930 {
            background: #0f172a;
            color: #ffffff;
            border: none;
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 11.5px;
            font-weight: 700;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .btn-call-1930:hover { background: #334155; }

        .btn-portal {
            background: #ffffff;
            color: var(--navy);
            border: 1px solid #cbd5e1;
            padding: 7px 11px;
            border-radius: 6px;
            font-size: 11.5px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .btn-portal:hover { background: #f1f5f9; }

        .log-heading {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .log-list {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
        }

        .log-entry {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            font-size: 12.5px;
        }
        .log-entry:last-child { border-bottom: none; }

        .vault-link {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: #fee2e2;
            color: #b91c1c;
            padding: 3px 8px;
            border-radius: 4px;
            text-decoration: none;
            font-weight: 700;
            font-size: 11px;
            margin-left: 8px;
        }
        .vault-link:hover { background: #fecaca; }

        /* POST-CALL INCIDENT POPUP MODAL */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(6px);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            padding: 20px;
            box-sizing: border-box;
        }

        .modal-box {
            background: #ffffff;
            border-radius: 14px;
            max-width: 540px;
            width: 100%;
            padding: 28px;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25);
            border: 1px solid #e2e8f0;
            animation: modalPop 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes modalPop {
            0% { transform: scale(0.92); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .modal-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }

        .modal-badge {
            background: #fee2e2;
            color: #dc2626;
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            padding: 4px 8px;
            border-radius: 6px;
        }

        .modal-title {
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }

        .modal-desc {
            font-size: 13.5px;
            color: #475569;
            line-height: 1.55;
            margin-bottom: 20px;
        }

        .modal-actions {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }

        /* Step-by-Step Procedure View */
        .procedure-steps {
            display: none;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .procedure-steps h4 {
            margin: 0 0 10px 0;
            font-size: 13px;
            font-weight: 700;
            color: #1e293b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .step-item {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 12.5px;
            color: #334155;
            line-height: 1.45;
        }

        .step-item:last-child { margin-bottom: 0; }

        .step-num {
            background: #0f172a;
            color: #ffffff;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            font-size: 10px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin-top: 1px;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <div>
                <h1>DHVANISHIELD <span style="font-weight: 500; font-size: 15px; color: var(--text-muted);">(ध्वनि-शील्ड)</span></h1>
                <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">AICTE Cyber Security Cell &bull; Real-Time Voice Defense</div>
            </div>
            <div class="actions">
                <button id="recBtn" class="btn btn-rec" onclick="toggleCallRecording()">
                    <span id="recDot">●</span> <span id="recLabel">Record Call Evidence</span>
                </button>
                <button class="btn btn-end-call" onclick="endCallSession()">⏹ End Call / Wrap Up</button>
                <button class="btn btn-white" onclick="openVaultFolder()">📂 Open Vault</button>
                <a href="/download-log" class="btn btn-white">📁 Export CSV</a>
            </div>
        </div>

        <!-- Cyber Cell Incident Reporting Banner -->
        <div class="cybercell-card">
            <div class="cc-left">
                <span class="cc-badge">Official Legal & Enforcement Channel</span>
                <span class="cc-title">National Cyber Crime Reporting Cell</span>
                <span style="font-size: 11.5px; color: var(--text-muted);">Report digital voice fraud directly under Ministry of Home Affairs guidelines</span>
            </div>
            <div class="cc-actions">
                <button class="btn btn-white" style="font-size: 11px;" onclick="copyIncidentReport()">📋 Copy FIR Brief</button>
                <a href="https://cybercrime.gov.in" target="_blank" class="btn-portal">🌐 Open Portal</a>
                <a href="tel:1930" class="btn-call-1930">📞 Dial 1930</a>
            </div>
        </div>

        <div class="card">
            <div id="statusBadge" class="status-indicator">● Standby &mdash; Monitoring Digital Call Stream</div>

            <div class="visualizer-card">
                <div class="visualizer-label">Live Signal Stream (48 kHz WASAPI)</div>
                <canvas id="scopeCanvas" width="700" height="85"></canvas>
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

            <div class="log-heading">
                <span>Forensic Intercept Telemetry Log</span>
                <span style="font-size: 10.5px; color: var(--text-muted); font-weight: normal;">Click any vaulted file to download</span>
            </div>
            <div id="logArea" class="log-list"></div>
        </div>
    </div>

    <!-- POST-CALL COMPLAINT MODAL -->
    <div id="complaintModal" class="modal-overlay">
        <div class="modal-box">
            <div class="modal-header">
                <span class="modal-badge">Fraud Alert</span>
                <h3 class="modal-title">AI Voice Clone Detected</h3>
            </div>
            
            <p id="modalPrompt" class="modal-desc">
                During this call session, DhvaniShield intercepted synthetic speech with a peak AI confidence of <strong id="modalPeakScore" style="color: #dc2626;">95%</strong>. 
                <br><br>
                <strong>Do you want to file a cybercrime complaint against this caller?</strong>
            </p>

            <!-- Standard Operating Procedure Steps (Revealed when clicking Yes) -->
            <div id="procedureSteps" class="procedure-steps">
                <h4>Official Filing Procedure (MHA Guidelines):</h4>
                <div class="step-item">
                    <span class="step-num">1</span>
                    <span><strong>Save Audio Evidence:</strong> Download the recorded `.wav` audio incident file from your Vault folder.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">2</span>
                    <span><strong>Generate FIR Brief:</strong> Click "Copy FIR Brief" to extract the timestamp and neural classification.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">3</span>
                    <span><strong>Lodge on Portal / 1930:</strong> Visit <em>cybercrime.gov.in</em> or immediately dial <strong>1930</strong> to report financial fraud or imposter extortion.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">4</span>
                    <span><strong>Bank Freeze:</strong> If OTP or bank transfer was requested, notify your bank immediately with this incident brief.</span>
                </div>
            </div>

            <div class="modal-actions" id="initialActions">
                <button class="btn btn-white" onclick="closeComplaintModal()">No, Dismiss</button>
                <button class="btn btn-rec recording" style="background: #dc2626; color: #fff;" onclick="proceedWithComplaint()">Yes, File Complaint</button>
            </div>

            <div class="modal-actions" id="procedureActions" style="display: none;">
                <button class="btn btn-white" onclick="copyIncidentReport()">📋 Copy FIR Brief</button>
                <a href="https://cybercrime.gov.in" target="_blank" class="btn btn-portal">🌐 Open Portal</a>
                <a href="tel:1930" class="btn btn-call-1930">📞 Dial 1930</a>
                <button class="btn btn-white" onclick="closeComplaintModal()">Done</button>
            </div>
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
        const recBtn = document.getElementById('recBtn');
        const recLabel = document.getElementById('recLabel');

        // Session Tracking State
        let isRecording = false;
        let sessionDetectedAI = false;
        let sessionPeakSpoof = 0.0;
        let sessionLastFile = "None";

        function toggleCallRecording() {
            if (!isRecording) {
                isRecording = true;
                recBtn.classList.add('recording');
                recLabel.innerText = "Recording... (Click to Save)";
                ws.send(JSON.stringify({ action: "start_record" }));
            } else {
                isRecording = false;
                recBtn.classList.remove('recording');
                recLabel.innerText = "Saving to Laptop...";
                ws.send(JSON.stringify({ action: "stop_record" }));

                // Stopping recording ends the recorded call; check for post-call alert
                setTimeout(evaluateCallEnd, 1000);
            }
        }

        function endCallSession() {
            if (isRecording) {
                toggleCallRecording();
            } else {
                evaluateCallEnd();
            }
        }

        function evaluateCallEnd() {
            if (sessionDetectedAI) {
                // Show Complaint Pop-up
                document.getElementById('modalPeakScore').innerText = `${sessionPeakSpoof}%`;
                document.getElementById('procedureSteps').style.display = 'none';
                document.getElementById('initialActions').style.display = 'flex';
                document.getElementById('procedureActions').style.display = 'none';
                document.getElementById('complaintModal').style.display = 'flex';
            } else {
                alert("Call Ended: Verified Authentic Caller. No AI voice clone activity detected.");
                resetCallSession();
            }
        }

        function proceedWithComplaint() {
            document.getElementById('procedureSteps').style.display = 'block';
            document.getElementById('initialActions').style.display = 'none';
            document.getElementById('procedureActions').style.display = 'flex';
        }

        function closeComplaintModal() {
            document.getElementById('complaintModal').style.display = 'none';
            resetCallSession();
        }

        function resetCallSession() {
            sessionDetectedAI = false;
            sessionPeakSpoof = 0.0;
            sessionLastFile = "None";
            badge.style.background = "#f1f5f9";
            badge.style.color = "var(--text-muted)";
            badge.innerText = "● Standby — Monitoring Digital Call Stream";
            fill.style.width = "0%";
            threatScore.innerText = "0.0% Synthetic";
            humanVal.innerText = "--%";
            aiVal.innerText = "--%";
        }

        function openVaultFolder() {
            fetch('/open-folder');
        }

        function copyIncidentReport() {
            const reportText = `[DHVANISHIELD CYBER FRAUD INCIDENT BRIEF]
Date/Time: ${new Date().toLocaleString()}
System Classification: AI VOICE CLONE (FRAUD ALERT)
Peak Synthetic Likelihood: ${sessionPeakSpoof}%
Digital Loopback Driver: 48 kHz WASAPI Downlink
Evidence File Vaulted: ${sessionLastFile}
Recommended Action: Lodge complaint on cybercrime.gov.in or dial 1930 with evidence attached.`;

            navigator.clipboard.writeText(reportText).then(() => {
                alert("Incident Brief copied to clipboard! You can paste it into the cybercrime.gov.in portal or police report.");
            });
        }

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
                const y = (canvas.height / 2) - (wavePoints[i] * (canvas.height / 2) * 1.8);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
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
            } else if (data.type === 'record_saved') {
                recLabel.innerText = "Record Call Evidence";

                const downloadLink = document.createElement('a');
                downloadLink.href = `/download-evidence/${data.filename}`;
                downloadLink.download = data.filename;
                document.body.appendChild(downloadLink);
                downloadLink.click();
                downloadLink.remove();
            } else if (data.type === 'verdict') {
                const spoof = data.spoof.toFixed(1);
                const bonafide = data.bonafide.toFixed(1);

                humanVal.innerText = `${bonafide}%`;
                aiVal.innerText = `${spoof}%`;
                threatScore.innerText = `${spoof}% Synthetic`;
                fill.style.width = `${spoof}%`;

                let tagColor = "var(--safe)";
                let tagText = "Human Voice";
                let downloadBtnHTML = "";

                // Mark session as flagged if spoof exceeds threshold
                if (data.spoof >= 65.0) {
                    sessionDetectedAI = true;
                    if (data.spoof > sessionPeakSpoof) sessionPeakSpoof = spoof;
                    if (data.file && data.file !== "None") sessionLastFile = data.file;

                    badge.style.background = "var(--danger-bg)";
                    badge.style.color = "var(--danger)";
                    badge.innerText = "🚨 WARNING: SYNTHETIC VOICE DETECTED";
                    fill.style.backgroundColor = "var(--danger)";
                    tagColor = "var(--danger)";
                    tagText = "AI Clone";

                    if (data.file && data.file !== "None") {
                        downloadBtnHTML = `<a href="/download-evidence/${data.file}" download class="vault-link">⬇ Save WAV</a>`;
                    }
                } else {
                    badge.style.background = "var(--safe-bg)";
                    badge.style.color = "var(--safe)";
                    badge.innerText = "✔ VERIFIED: AUTHENTIC HUMAN CALLER";
                    fill.style.backgroundColor = "var(--safe)";
                }

                const row = document.createElement('div');
                row.className = 'log-entry';
                row.innerHTML = `
                    <span><strong>${data.time}</strong> &mdash; ${data.duration}s utterance ${downloadBtnHTML}</span>
                    <span style="color: ${tagColor}; font-weight: 600;">${tagText} (${spoof}%)</span>
                `;
                logArea.prepend(row);
            }
        };
    </script>
</body>
</html>
"""

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

                    inputs = feature_extractor(utterance_16k, sampling_rate=TARGET_RATE, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = deepfake_model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)[0]

                    spoof_pct = float(probs[1]) * 100
                    bonafide_pct = float(probs[0]) * 100
                    timestamp_str = time.strftime("%H:%M:%S")

                    file_saved = "None"
                    if spoof_pct >= 65.0:
                        file_saved = archive_utterance(
                            utterance_16k, spoof_pct, bonafide_pct,
                            round(len(utterance_16k) / TARGET_RATE, 2), timestamp_str
                        )

                    payload = {
                        "type": "verdict",
                        "spoof": spoof_pct,
                        "bonafide": bonafide_pct,
                        "duration": round(len(utterance_16k) / TARGET_RATE, 2),
                        "time": timestamp_str,
                        "file": file_saved
                    }
                    asyncio.run_coroutine_threadsafe(telemetry_queue.put(payload), loop)

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
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_CONTENT

@app.get("/download-log")
async def download_log():
    if os.path.exists(LOG_FILE):
        return FileResponse(LOG_FILE, media_type="text/csv", filename="forensic_audit_log.csv")
    return HTMLResponse("No log available yet.", status_code=404)

@app.get("/download-evidence/{filename}")
async def download_evidence(filename: str):
    file_path = os.path.join(EVIDENCE_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    return HTMLResponse("Evidence file not found.", status_code=404)

@app.get("/open-folder")
async def open_folder():
    try:
        os.startfile(EVIDENCE_DIR)
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}

@app.websocket("/ws")
async def telemetry_feed(websocket: WebSocket):
    global is_manual_recording, manual_record_buffer
    await websocket.accept()

    async def sender():
        try:
            while True:
                payload = await telemetry_queue.get()
                await websocket.send_text(json.dumps(payload))
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            text = await websocket.receive_text()
            cmd = json.loads(text)
            action = cmd.get("action")

            if action == "start_record":
                with recording_lock:
                    manual_record_buffer = []
                    is_manual_recording = True
                print("[!] Full call evidence recording started.")

            elif action == "stop_record":
                with recording_lock:
                    is_manual_recording = False
                    captured = list(manual_record_buffer)
                    manual_record_buffer = []

                if captured:
                    full_call_arr = np.concatenate(captured)
                    rec_filename = f"FULL_CALL_{time.strftime('%Y%m%d_%H%M%S')}.wav"
                    rec_path = os.path.join(EVIDENCE_DIR, rec_filename)

                    scaled = np.int16(np.clip(full_call_arr, -1.0, 1.0) * 32767)
                    with wave.open(rec_path, "w") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(48000)
                        wf.writeframes(scaled.tobytes())

                    print(f"[✔] Full call evidence saved to: {rec_path}")
                    await websocket.send_text(json.dumps({
                        "type": "record_saved",
                        "filename": rec_filename
                    }))
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
def get_free_port(preferred=8000):
    for port in range(preferred, preferred + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found")

if __name__ == "__main__":
    port = get_free_port(8000)
    print(f"[*] Binding on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)