import queue
import time
import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Audio parameters
NATIVE_RATE = 48000
TARGET_RATE = 16000
FRAME_SIZE = 1536             # 32ms at 48kHz
VAD_THRESHOLD = 0.55          # Clean digital threshold
MIN_SPEECH_SECONDS = 1.2
MAX_SPEECH_SECONDS = 4.0
SILENCE_PAD_FRAMES = 16       # ~500ms silence terminates utterance

# Device 17: CABLE Output via Windows WASAPI
DEVICE_INDEX = 17

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("     DIGITAL CALL INTERCEPTION ENGINE (WASAPI DIRECT)")
print(f" Target Device:   Device {DEVICE_INDEX} [CABLE Output - WASAPI]")
print(f" Compute Device:  {device.upper()}")
print("=" * 60)

print("[*] Loading Neural VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
vad_model.to(device)
vad_model.eval()

print("[*] Loading Forensic Transformer...")
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
deepfake_model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
deepfake_model.eval()

resampler = T.Resample(orig_freq=NATIVE_RATE, new_freq=TARGET_RATE).to(device)
raw_audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    # Pure digital capture from channel 0
    raw_audio_queue.put(indata[:, 0].copy())

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

    print(f"\n{verdict} | Duration: {duration:.2f}s | Timestamp: {time.strftime('%H:%M:%S')}")

stream = sd.InputStream(
    device=DEVICE_INDEX,
    samplerate=NATIVE_RATE,
    channels=2,
    blocksize=FRAME_SIZE,
    callback=audio_callback
)

with stream:
    print("\n[READY] Listening to incoming digital stream...")
    print("[Routing Note]: Play any voice through 'CABLE Input' to test.\n")

    speech_buffer = []
    silence_frames = 0
    is_speaking = False
    vad_model.reset_states()

    try:
        while True:
            frame = raw_audio_queue.get()
            rms = float(np.sqrt(np.mean(frame**2)))

            # Absolute digital silence gate (no noise floor exists in digital loopback)
            if rms < 0.001:
                if is_speaking:
                    silence_frames += 1
                    speech_buffer.append(frame)
                else:
                    vad_model.reset_states()
                    print(f"\r[Digital Stream Idle] Amplitude: 0.000000 ", end="", flush=True)
                    continue
            else:
                tensor_48k = torch.from_numpy(frame).to(device)
                tensor_16k = resampler(tensor_48k)

                with torch.no_grad():
                    speech_prob = vad_model(tensor_16k, TARGET_RATE).item()

                if speech_prob >= VAD_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        print("\r[STREAM]: Speech burst detected. Buffering audio...     ", end="", flush=True)
                    speech_buffer.append(frame)
                    silence_frames = 0
                elif is_speaking:
                    silence_frames += 1
                    speech_buffer.append(frame)

            if is_speaking:
                total_duration = (len(speech_buffer) * FRAME_SIZE) / NATIVE_RATE

                if (silence_frames >= SILENCE_PAD_FRAMES and total_duration >= MIN_SPEECH_SECONDS) or (total_duration >= MAX_SPEECH_SECONDS):
                    full_utterance_48k = np.concatenate(speech_buffer)
                    utterance_tensor = torch.from_numpy(full_utterance_48k).to(device)
                    utterance_16k = resampler(utterance_tensor).cpu().numpy()

                    analyze_complete_utterance(utterance_16k)

                    speech_buffer = []
                    silence_frames = 0
                    is_speaking = False
                    vad_model.reset_states()
                    print("[Standby]: Listening for incoming stream packets...", end="", flush=True)

                elif silence_frames >= SILENCE_PAD_FRAMES:
                    speech_buffer = []
                    silence_frames = 0
                    is_speaking = False
                    vad_model.reset_states()
                    print("\r[STREAM]: Dropped sub-second blip (< 1.2s). Ready...     ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nEngine safely shut down.")