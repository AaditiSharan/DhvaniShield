import queue
import time
import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Audio parameters
# We ingest at 48kHz (the native hardware clock of your Intel SST mic) to stop driver clicking
NATIVE_RATE = 48000
TARGET_RATE = 16000
FRAME_SIZE = 1536             # 32ms at 48kHz (1536 / 48000 = 0.032s)
VAD_THRESHOLD = 0.70          # Strict speech threshold
MIN_SPEECH_SECONDS = 1.3
MAX_SPEECH_SECONDS = 4.0
SILENCE_PAD_FRAMES = 18       # ~570ms silence terminates utterance

# Lock directly to Device 9 (Intel Smart Sound via Windows WASAPI)
DEVICE_INDEX = 9

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("     ENTERPRISE FORENSIC ENGINE (WASAPI DIRECT)")
print(f" Compute Device: {device.upper()}")
print(f" Input Device:   WASAPI Direct [Index {DEVICE_INDEX}]")
print(f" Hardware Clock: {NATIVE_RATE} Hz -> {TARGET_RATE} Hz (HQ Resampler)")
print("=" * 60)

# 1. Load Neural VAD
print("[*] Loading Silero Neural VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
vad_model.to(device)
vad_model.eval()

# 2. Load Deepfake Transformer
print("[*] Loading Deepfake Transformer Engine...")
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
deepfake_model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
deepfake_model.eval()

# High-quality sinc resampler (eliminates MME aliasing artifacts)
resampler = T.Resample(orig_freq=NATIVE_RATE, new_freq=TARGET_RATE).to(device)

raw_audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    # Take channel 0 and strip any hardware DC bias
    raw_frame = indata[:, 0]
    clean_frame = raw_frame - np.mean(raw_frame)
    raw_audio_queue.put(clean_frame.copy())

def analyze_complete_utterance(audio_segment_16k):
    inputs = feature_extractor(audio_segment_16k, sampling_rate=TARGET_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = deepfake_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    fake_pct = float(probs[1]) * 100
    real_pct = float(probs[0]) * 100
    duration = len(audio_segment_16k) / TARGET_RATE

    if fake_pct >= 65.0:
        verdict = f"🚨 [HIGH THREAT: AI CLONE]  Spoof: {fake_pct:5.1f}% | Bonafide: {real_pct:5.1f}%"
    else:
        verdict = f"✅ [AUTHENTIC HUMAN VOICE] Bonafide: {real_pct:5.1f}% | Spoof: {fake_pct:5.1f}%"

    print(f"\n{verdict} | Duration: {duration:.2f}s | Time: {time.strftime('%H:%M:%S')}")

# Start Stream on Device 9 (WASAPI)
stream = sd.InputStream(
    device=DEVICE_INDEX,
    samplerate=NATIVE_RATE,
    channels=2,
    blocksize=FRAME_SIZE,
    callback=audio_callback
)

with stream:
    # 3. Discard the initial 1.0 second of driver spin-up artifacts
    print("\n[*] Initializing audio stream and purging driver buffers...")
    time.sleep(1.0)
    while not raw_audio_queue.empty():
        try:
            raw_audio_queue.get_nowait()
        except queue.Empty:
            break

    print("[READY] Engine active. Standby in silence or speak a sentence to test.\n")

    speech_buffer = []
    silence_frames = 0
    is_speaking = False
    vad_model.reset_states()

    try:
        while True:
            frame = raw_audio_queue.get()
            rms = float(np.sqrt(np.mean(frame**2)))

            # Stage 1: Energy Gate (rejects true silence without GPU/CPU load)
            if rms < 0.003:
                if is_speaking:
                    silence_frames += 1
                    speech_buffer.append(frame)
                else:
                    vad_model.reset_states()
                    print(f"\r[Idle / Silent] RMS: {rms:.6f}                  ", end="", flush=True)
                    continue
            else:
                # Stage 2: Convert to 16kHz and evaluate with Neural VAD
                tensor_48k = torch.from_numpy(frame).to(device)
                tensor_16k = resampler(tensor_48k)

                with torch.no_grad():
                    speech_prob = vad_model(tensor_16k, TARGET_RATE).item()

                if speech_prob >= VAD_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        print("\r[VAD]: Verified human speech. Buffering utterance...        ", end="", flush=True)
                    speech_buffer.append(frame)
                    silence_frames = 0
                elif is_speaking:
                    silence_frames += 1
                    speech_buffer.append(frame)

            # Stage 3: Sentence Evaluation
            if is_speaking:
                total_duration = (len(speech_buffer) * FRAME_SIZE) / NATIVE_RATE

                if (silence_frames >= SILENCE_PAD_FRAMES and total_duration >= MIN_SPEECH_SECONDS) or (total_duration >= MAX_SPEECH_SECONDS):
                    full_utterance_48k = np.concatenate(speech_buffer)
                    # Resample entire utterance cleanly to 16kHz for deepfake model
                    utterance_tensor = torch.from_numpy(full_utterance_48k).to(device)
                    utterance_16k = resampler(utterance_tensor).cpu().numpy()

                    analyze_complete_utterance(utterance_16k)

                    speech_buffer = []
                    silence_frames = 0
                    is_speaking = False
                    vad_model.reset_states()
                    print("[Standby]: Listening for next phrase...", end="", flush=True)

                elif silence_frames >= SILENCE_PAD_FRAMES:
                    speech_buffer = []
                    silence_frames = 0
                    is_speaking = False
                    vad_model.reset_states()
                    print("\r[VAD]: Discarded transient sound (< 1.3s). Listening...       ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nEngine safely shut down.")