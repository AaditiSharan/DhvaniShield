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

TARGET_RATE = 16000
VAD_THRESHOLD = 0.55
MIN_SPEECH_SECONDS = 1.2
MAX_SPEECH_SECONDS = 4.0
SILENCE_PAD_FRAMES = 16

EVIDENCE_DIR = os.path.join(os.getcwd(), "evidence")
LOG_FILE = os.path.join(EVIDENCE_DIR, "call_audit_log.csv")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Classification", "Spoof_Score_Pct", "Bonafide_Score_Pct", "Duration_Sec", "Audio_File"])

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("     DHVANISHIELD (ध्वनि-शील्ड) : REAL-TIME DEFENSE CENTER")
print(f" Compute Engine:  {device.upper()}")
print(f" Local Vault:     {EVIDENCE_DIR}")
print("=" * 65)

# Models
print("[*] Loading Silero Neural VAD...")
vad_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
vad_model.to(device).eval()

print("[*] Loading Wav2Vec2 Deepfake Transformer...")
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
deepfake_model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
deepfake_model.eval()

# Decoupled Queues for Zero-Latency Streaming
telemetry_queue = asyncio.Queue()
raw_audio_queue = queue.Queue()
inference_queue = queue.Queue()

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
    <title>DHVANISHIELD (ध्वनि-शील्ड) | Voice Defense</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Azure Blue & Crisp White Palette */
            --bg: #f0f6fc;
            --surface: #ffffff;
            --surface-subtle: #f8fbfe;
            --border: #dbe7f4;
            --border-strong: #bfd6f0;
            --text-main: #0b2545;
            --text-muted: #5c7899;
            --primary: #2d8ce1;
            --primary-hover: #1b77cb;
            --primary-subtle: #e3f0fc;
            --safe: #059669;
            --safe-bg: #ecfdf5;
            --safe-border: #a7f3d0;
            --danger: #e11d48;
            --danger-bg: #fff1f2;
            --danger-border: #fecdd3;
            --wave-line: #2d8ce1;
            --shadow: 0 4px 18px rgba(45, 140, 225, 0.08);
            --modal-bg: #ffffff;
        }

        body[data-theme="dark"] {
            /* Stealth Deep Navy & Azure Glow */
            --bg: #0b121e;
            --surface: #111b2b;
            --surface-subtle: #172439;
            --border: #1e314d;
            --border-strong: #29446b;
            --text-main: #f0f6fc;
            --text-muted: #8ea5c4;
            --primary: #38bdf8;
            --primary-hover: #0284c7;
            --primary-subtle: rgba(56, 189, 248, 0.12);
            --safe: #10b981;
            --safe-bg: rgba(16, 185, 129, 0.12);
            --safe-border: rgba(16, 185, 129, 0.3);
            --danger: #f43f5e;
            --danger-bg: rgba(244, 63, 94, 0.14);
            --danger-border: rgba(244, 63, 94, 0.35);
            --wave-line: #38bdf8;
            --shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
            --modal-bg: #111b2b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 20px 28px 48px;
            min-height: 100vh;
        }

        .dashboard-container {
            max-width: 1440px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Top Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 24px;
            box-shadow: var(--shadow);
        }

        .brand-block h1 {
            font-size: 21px;
            font-weight: 800;
            letter-spacing: -0.4px;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .brand-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--primary);
            box-shadow: 0 0 10px var(--primary);
            display: inline-block;
        }

        .header-sub {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 600;
            margin-top: 3px;
        }

        .header-actions {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .btn {
            border: 1px solid var(--border);
            padding: 8px 14px;
            border-radius: 9px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
        }

        .btn-white {
            background: var(--surface);
            color: var(--text-main);
        }
        .btn-white:hover {
            border-color: var(--primary);
            color: var(--primary);
            transform: translateY(-1px);
        }

        .btn-rec {
            background: var(--surface);
            color: var(--danger);
            border-color: var(--danger-border);
        }
        .btn-rec.recording {
            background: var(--danger);
            color: #ffffff;
            border-color: var(--danger);
            animation: pulse-border 1.5s infinite;
        }

        .btn-end-call {
            background: var(--primary);
            color: #ffffff;
            border-color: var(--primary);
        }
        .btn-end-call:hover {
            background: var(--primary-hover);
        }

        @keyframes pulse-border {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.75; }
        }

        /* 3-Column Workspace Grid */
        .workspace-grid {
            display: grid;
            grid-template-columns: 340px 1.35fr 360px;
            gap: 18px;
            align-items: stretch;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
        }

        .card-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: var(--text-muted);
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Left Wing */
        .left-column {
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .cyber-cell-card {
            background: var(--surface);
            border: 1.5px solid var(--border-strong);
            border-radius: 16px;
            padding: 22px;
            box-shadow: var(--shadow);
        }

        .cc-badge {
            font-size: 10px;
            font-weight: 800;
            color: var(--primary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            display: inline-block;
            margin-bottom: 6px;
        }

        .cc-heading {
            font-size: 15px;
            font-weight: 800;
            color: var(--text-main);
            line-height: 1.3;
        }

        .cc-sub {
            font-size: 11.5px;
            color: var(--text-muted);
            margin-top: 6px;
            line-height: 1.5;
        }

        .cc-action-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 16px;
        }

        .btn-call-1930 {
            background: var(--primary);
            color: #ffffff;
            border: none;
            padding: 11px 14px;
            border-radius: 9px;
            font-size: 12.5px;
            font-weight: 700;
            text-decoration: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .btn-call-1930:hover { background: var(--primary-hover); }

        .btn-portal {
            background: var(--primary-subtle);
            color: var(--primary);
            border: 1px solid var(--border);
            padding: 10px 12px;
            border-radius: 9px;
            font-size: 12px;
            font-weight: 700;
            text-decoration: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .btn-portal:hover { border-color: var(--primary); }

        .sop-checklist {
            margin-top: 16px;
            border-top: 1px solid var(--border);
            padding-top: 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .sop-item {
            font-size: 11.5px;
            color: var(--text-muted);
            display: flex;
            gap: 6px;
            line-height: 1.45;
        }

        /* Center Column */
        .center-column {
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 7px 16px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 18px;
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            color: var(--text-muted);
            letter-spacing: 0.3px;
        }

        .visualizer-card {
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 20px;
            position: relative;
        }

        .visualizer-label {
            position: absolute;
            top: 10px;
            left: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        canvas {
            width: 100%;
            height: 120px;
            display: block;
        }

        .meter-title {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .meter-bar {
            height: 13px;
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }

        .meter-fill {
            height: 100%;
            width: 0%;
            background: var(--safe);
            transition: width 0.35s ease, background-color 0.35s ease;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .metric-box {
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }

        .metric-label {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 34px;
            font-weight: 700;
        }

        /* Right Wing - Fixed Height Never Expands */
        .right-column {
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .right-column .card {
            height: 100%;
        }

        .log-container {
            height: 420px;
            min-height: 420px;
            max-height: 420px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--surface-subtle);
            padding: 2px;
        }

        .log-placeholder {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            padding: 20px;
        }

        .log-entry {
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
        }
        .log-entry:last-child { border-bottom: none; }

        .log-entry-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .vault-link {
            display: inline-flex;
            align-items: center;
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: var(--danger);
            padding: 4px 8px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 700;
            font-size: 10.5px;
            margin-top: 5px;
            align-self: flex-start;
        }
        .vault-link:hover { opacity: 0.85; }

        /* How to Use / Key Features Section */
        .features-section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .features-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }

        .features-title {
            font-size: 14px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: var(--text-main);
        }

        .features-tag {
            font-size: 11px;
            color: var(--primary);
            font-weight: 700;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
        }

        .features-card {
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .features-card h4 {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
        }

        .features-card p {
            font-size: 12px;
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* Premium Brand Footer */
        .footer {
            margin-top: 6px;
            padding: 18px 4px 6px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .footer-brand {
            font-weight: 700;
            color: var(--text-main);
        }

        .footer-team {
            font-weight: 700;
            color: var(--primary);
        }

        /* Post-Call Modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(11, 37, 69, 0.5);
            backdrop-filter: blur(8px);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            padding: 20px;
        }

        .modal-box {
            background: var(--modal-bg);
            border-radius: 18px;
            max-width: 540px;
            width: 100%;
            padding: 28px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
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
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: var(--danger);
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            padding: 4px 8px;
            border-radius: 6px;
        }

        .modal-title {
            font-size: 18px;
            font-weight: 800;
            color: var(--text-main);
            margin: 0;
        }

        .modal-desc {
            font-size: 13.5px;
            color: var(--text-muted);
            line-height: 1.55;
            margin-bottom: 20px;
        }

        .modal-actions {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }

        .procedure-steps {
            display: none;
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .procedure-steps h4 {
            margin: 0 0 10px 0;
            font-size: 12.5px;
            font-weight: 700;
            color: var(--text-main);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .step-item {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            font-size: 12.5px;
            color: var(--text-muted);
            line-height: 1.45;
        }
        .step-item:last-child { margin-bottom: 0; }

        .step-num {
            background: var(--primary);
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
    <div class="dashboard-container">
        <!-- Top Navigation -->
        <div class="header">
            <div class="brand-block">
                <h1><span class="brand-dot"></span> DHVANISHIELD <span style="font-weight: 500; font-size: 15px; color: var(--text-muted);">(ध्वनि-शील्ड)</span></h1>
                <div class="header-sub">AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks</div>
            </div>
            <div class="header-actions">
                <button id="themeBtn" class="btn btn-white" onclick="toggleTheme()">Dark Mode</button>
                <button id="recBtn" class="btn btn-rec" onclick="toggleCallRecording()">
                    <span id="recDot">●</span> <span id="recLabel">Record Call Evidence</span>
                </button>
                <button class="btn btn-end-call" onclick="endCallSession()">End Call / Wrap Up</button>
                <button class="btn btn-white" onclick="openVaultFolder()">Open Vault</button>
                <a href="/download-log" class="btn btn-white">Export Activity CSV</a>
            </div>
        </div>

        <!-- 3-Column Workspace -->
        <div class="workspace-grid">
            <!-- Left Column: Cyber Cell Desk -->
            <div class="left-column">
                <div class="cyber-cell-card">
                    <span class="cc-badge">Emergency Escalation Desk</span>
                    <h3 class="cc-heading">National Cyber Crime Reporting Cell</h3>
                    <p class="cc-sub">Direct escalation channel for voice extortion, digital arrest scams, and financial impersonation fraud.</p>

                    <div class="cc-action-list">
                        <a href="tel:1930" class="btn-call-1930">Dial 1930 Helpline</a>
                        <a href="https://cybercrime.gov.in" target="_blank" class="btn-portal">Open cybercrime.gov.in</a>
                        <button class="btn btn-white" style="justify-content: center;" onclick="copyIncidentReport()">Copy FIR Brief</button>
                    </div>

                    <div class="sop-checklist">
                        <div class="sop-item">
                            <span>&bull;</span>
                            <span><strong>Golden Hour:</strong> Dial 1930 immediately to freeze fraudulent money transfers.</span>
                        </div>
                        <div class="sop-item">
                            <span>&bull;</span>
                            <span><strong>Audio Vault:</strong> Attach downloaded audio as proof on the portal.</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Center Column: Continuous Zero-Lag Oscilloscope -->
            <div class="center-column">
                <div class="card">
                    <div id="statusBadge" class="status-indicator">Standby &mdash; Monitoring Digital Call Stream</div>

                    <div class="visualizer-card">
                        <div class="visualizer-label">Live Signal Oscilloscope (48 kHz WASAPI)</div>
                        <canvas id="scopeCanvas" width="700" height="120"></canvas>
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
                </div>
            </div>

            <!-- Right Column: Intercept Telemetry Feed (Fixed Height) -->
            <div class="right-column">
                <div class="card">
                    <div class="card-title">
                        <span>Call Activity Telemetry</span>
                        <span id="entryCount" style="font-weight: normal; font-size: 11px;">0 Chunks</span>
                    </div>
                    <div id="logArea" class="log-container">
                        <div id="logPlaceholder" class="log-placeholder">
                            Awaiting incoming speech bursts on line...
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- How to Use / Key Features Section -->
        <section class="features-section">
            <div class="features-header">
                <span class="features-title">How to Use DhvaniShield</span>
                <span class="features-tag">Simple Step-by-Step Guide</span>
            </div>
            <div class="features-grid">
                <div class="features-card">
                    <h4>1. Live Speech Monitoring</h4>
                    <p>Keep this dashboard open during phone calls. DhvaniShield listens to the incoming audio line continuously and instantly checks whether the caller is a real human or an AI voice clone.</p>
                </div>
                <div class="features-card">
                    <h4>2. Threat Alerts & Recording</h4>
                    <p>If the caller is human, the gauge stays green. If a synthetic clone speaks, the banner flashes red with a visual threat warning. Click "Record Call Evidence" anytime to save the full conversation to your laptop.</p>
                </div>
                <div class="features-card">
                    <h4>3. End Call & File Complaint</h4>
                    <p>When the conversation ends, click "End Call / Wrap Up". If an AI scam was detected, a popup immediately appears with the official 4-step procedure to file a complaint via 1930 and cybercrime.gov.in.</p>
                </div>
            </div>
        </section>

        <!-- Premium Footer -->
        <footer class="footer">
            <div>
                <span class="footer-brand">DhvaniShield (ध्वनि-शील्ड)</span> &mdash; Voice Cloning Detection & Defense System
            </div>
            <div>
                Developed by <strong class="footer-team">C6 Coders</strong> &bull; 2026
            </div>
        </footer>
    </div>

    <!-- POST-CALL COMPLAINT MODAL -->
    <div id="complaintModal" class="modal-overlay">
        <div class="modal-box">
            <div class="modal-header">
                <span class="modal-badge">Fraud Alert</span>
                <h3 class="modal-title">AI Voice Clone Detected</h3>
            </div>
            
            <p id="modalPrompt" class="modal-desc">
                During this call session, DhvaniShield intercepted synthetic speech with a peak AI confidence of <strong id="modalPeakScore" style="color: var(--danger);">95%</strong>. 
                <br><br>
                <strong>Do you want to file an official cybercrime complaint against this caller?</strong>
            </p>

            <div id="procedureSteps" class="procedure-steps">
                <h4>Official Filing Procedure (MHA Guidelines):</h4>
                <div class="step-item">
                    <span class="step-num">1</span>
                    <span><strong>Save Audio Evidence:</strong> Download the recorded audio incident file from your Vault.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">2</span>
                    <span><strong>Generate FIR Brief:</strong> Click "Copy FIR Brief" to extract timestamps and neural scores.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">3</span>
                    <span><strong>Lodge on Portal / 1930:</strong> Visit <em>cybercrime.gov.in</em> or dial <strong>1930</strong> to report imposter extortion.</span>
                </div>
                <div class="step-item">
                    <span class="step-num">4</span>
                    <span><strong>Bank Notification:</strong> If financial information was discussed, contact your bank immediately.</span>
                </div>
            </div>

            <div class="modal-actions" id="initialActions">
                <button class="btn btn-white" onclick="closeComplaintModal()">No, Dismiss</button>
                <button class="btn btn-rec recording" style="background: var(--danger); color: #fff;" onclick="proceedWithComplaint()">Yes, File Complaint</button>
            </div>

            <div class="modal-actions" id="procedureActions" style="display: none;">
                <button class="btn btn-white" onclick="copyIncidentReport()">Copy FIR Brief</button>
                <a href="https://cybercrime.gov.in" target="_blank" class="btn btn-portal">Open Portal</a>
                <a href="tel:1930" class="btn btn-call-1930">Dial 1930</a>
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
        const entryCount = document.getElementById('entryCount');
        const themeBtn = document.getElementById('themeBtn');

        function toggleTheme() {
            const currentTheme = document.body.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.body.setAttribute('data-theme', newTheme);
            themeBtn.innerText = newTheme === 'dark' ? 'Light Mode' : 'Dark Mode';
            localStorage.setItem('dhvani_theme', newTheme);
        }

        const savedTheme = localStorage.getItem('dhvani_theme');
        if (savedTheme) {
            document.body.setAttribute('data-theme', savedTheme);
            themeBtn.innerText = savedTheme === 'dark' ? 'Light Mode' : 'Dark Mode';
        }

        let isRecording = false;
        let sessionDetectedAI = false;
        let sessionPeakSpoof = 0.0;
        let sessionLastFile = "None";
        let totalLogged = 0;

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
                document.getElementById('modalPeakScore').innerText = `${sessionPeakSpoof}%`;
                document.getElementById('procedureSteps').style.display = 'none';
                document.getElementById('initialActions').style.display = 'flex';
                document.getElementById('procedureActions').style.display = 'none';
                document.getElementById('complaintModal').style.display = 'flex';
            } else {
                alert("Call Ended: Authentic Human Caller verified. No AI voice clone activity detected.");
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
            badge.style.background = "var(--surface-subtle)";
            badge.style.borderColor = "var(--border)";
            badge.style.color = "var(--text-muted)";
            badge.innerText = "Standby — Monitoring Digital Call Stream";
            fill.style.width = "0%";
            threatScore.innerText = "0.0% Synthetic";
            humanVal.innerText = "--%";
            aiVal.innerText = "--%";
        }

        function openVaultFolder() {
            fetch('/open-folder');
        }

        function copyIncidentReport() {
            const reportText = `[DHVANISHIELD FRAUD INCIDENT BRIEF]
Date/Time: ${new Date().toLocaleString()}
Classification: AI VOICE CLONE (FRAUD ALERT)
Peak Synthetic Likelihood: ${sessionPeakSpoof}%
Audio Vault Incident File: ${sessionLastFile}
Recommended Action: Lodge complaint on cybercrime.gov.in or dial 1930 with evidence attached.`;

            navigator.clipboard.writeText(reportText).then(() => {
                alert("Incident Brief copied to clipboard! You can paste it into the cybercrime.gov.in portal or police report.");
            });
        }

        let wavePoints = new Array(64).fill(0);

        function drawWaveform() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const computed = getComputedStyle(document.body);
            const lineColor = computed.getPropertyValue('--wave-line').trim();
            const gridColor = computed.getPropertyValue('--border').trim();

            ctx.beginPath();
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();

            ctx.beginPath();
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2.4;
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
                totalLogged++;
                if (totalLogged === 1) {
                    logArea.innerHTML = "";
                }
                entryCount.innerText = `${totalLogged} Chunk${totalLogged > 1 ? 's' : ''}`;

                const spoof = data.spoof.toFixed(1);
                const bonafide = data.bonafide.toFixed(1);

                humanVal.innerText = `${bonafide}%`;
                aiVal.innerText = `${spoof}%`;
                threatScore.innerText = `${spoof}% Synthetic`;
                fill.style.width = `${spoof}%`;

                let tagColor = "var(--safe)";
                let tagText = "Human Voice";
                let downloadBtnHTML = "";

                if (data.spoof >= 65.0) {
                    sessionDetectedAI = true;
                    if (data.spoof > sessionPeakSpoof) sessionPeakSpoof = spoof;
                    if (data.file && data.file !== "None") sessionLastFile = data.file;

                    badge.style.background = "var(--danger-bg)";
                    badge.style.borderColor = "var(--danger-border)";
                    badge.style.color = "var(--danger)";
                    badge.innerText = "WARNING: SYNTHETIC VOICE DETECTED";
                    fill.style.backgroundColor = "var(--danger)";
                    tagColor = "var(--danger)";
                    tagText = "AI Clone";

                    if (data.file && data.file !== "None") {
                        downloadBtnHTML = `<a href="/download-evidence/${data.file}" download class="vault-link">Save Audio</a>`;
                    }
                } else {
                    badge.style.background = "var(--safe-bg)";
                    badge.style.borderColor = "var(--safe-border)";
                    badge.style.color = "var(--safe)";
                    badge.innerText = "VERIFIED: AUTHENTIC HUMAN CALLER";
                    fill.style.backgroundColor = "var(--safe)";
                }

                const row = document.createElement('div');
                row.className = 'log-entry';
                row.innerHTML = `
                    <div class="log-entry-header">
                        <strong>${data.time}</strong>
                        <span style="color: ${tagColor}; font-weight: 700;">${tagText} (${spoof}%)</span>
                    </div>
                    <div style="color: var(--text-muted); font-size: 11px;">Duration: ${data.duration}s call chunk</div>
                    ${downloadBtnHTML}
                `;
                logArea.prepend(row);
            }
        };
    </script>
</body>
</html>
"""

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
            duration = round(len(utterance_16k) / TARGET_RATE, 2)

            file_saved = "None"
            if spoof_pct >= 65.0:
                file_saved = archive_utterance(utterance_16k, spoof_pct, bonafide_pct, duration, timestamp_str)

            payload = {
                "type": "verdict",
                "spoof": spoof_pct,
                "bonafide": bonafide_pct,
                "duration": duration,
                "time": timestamp_str,
                "file": file_saved
            }
            asyncio.run_coroutine_threadsafe(telemetry_queue.put(payload), loop)
        except Exception:
            pass

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

@app.get("/download-log")
async def download_log():
    if os.path.exists(LOG_FILE):
        return FileResponse(LOG_FILE, media_type="text/csv", filename="call_audit_log.csv")
    return HTMLResponse("No log available yet.", status_code=404)

@app.get("/download-evidence/{filename}")
async def download_evidence(filename: str):
    file_path = os.path.join(EVIDENCE_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    return HTMLResponse("File not found.", status_code=404)

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
                print("[!] Full call recording started.")

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

                    print(f"[✔] Full call audio saved to: {rec_path}")
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