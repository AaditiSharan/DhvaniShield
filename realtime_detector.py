import queue
import time
import numpy as np
import sounddevice as sd
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Audio parameters
SAMPLE_RATE = 16000
WINDOW_DURATION = 2.5   # 2.5-second rolling window
TOTAL_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
BLOCK_SIZE = int(SAMPLE_RATE * 0.5)  # 0.5s audio slices
SILENCE_THRESHOLD = 0.015

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 55)
print("     AI VOICE CLONE SENTINEL (MULTI-THREADED)")
print(f" Compute Device:   {device.upper()}")
print(f" Detection Engine: {MODEL_NAME}")
print("=" * 55)

# Load model & feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Thread-safe audio queue & rolling buffer
audio_queue = queue.Queue()
audio_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    """Callback dumps incoming audio directly to queue without blocking."""
    if status:
        pass
    audio_queue.put(indata[:, 0].copy())

def predict_deepfake(waveform):
    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    id2label = model.config.id2label
    scores = {id2label[i].lower(): float(probabilities[i]) * 100 for i in range(len(probabilities))}
    return scores

print("\nSentinel online! Listening to ambient speech...")
print("-> Speak naturally into your mic.")
print("-> Play a synthetic/cloned voice from your phone.")
print("Press Ctrl+C to terminate.\n")

# Start audio stream
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCK_SIZE,
    callback=audio_callback
)

with stream:
    try:
        while True:
            # Wait for the next 0.5s audio slice
            chunk = audio_queue.get()

            # Roll buffer and insert new chunk
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            # Ignore non-speech silence
            rms = np.sqrt(np.mean(audio_buffer**2))
            if rms < SILENCE_THRESHOLD:
                print(f"\r[Standby] Ambient Silence (RMS: {rms:.4f})              ", end="", flush=True)
                continue

            # Run inference
            scores = predict_deepfake(audio_buffer)
            fake_prob = scores.get("fake", scores.get("spoof", 0.0))
            real_prob = scores.get("real", scores.get("bonafide", 0.0))

            if fake_prob >= 65.0:
                badge = f"🚨 [AI VOICE CLONE DETECTED] Fake: {fake_prob:.1f}% | Real: {real_prob:.1f}%"
            else:
                badge = f"✅ [AUTHENTIC HUMAN VOICE]  Real: {real_prob:.1f}% | Fake: {fake_prob:.1f}%"

            print(f"\r{badge} (RMS: {rms:.3f}) ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nSentinel terminated cleanly.")