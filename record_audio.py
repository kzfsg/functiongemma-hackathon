"""
Simple audio recording utility for CANDOR demo
Requires: pip install sounddevice scipy
"""

import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

SAMPLE_RATE = 16000  # Whisper expects 16kHz
DURATION = 5  # seconds

def record_audio(filename="recorded.wav", duration=DURATION):
    """Record audio from microphone"""
    print(f"Recording for {duration} seconds...")
    print("Speak now!")

    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()

    wav.write(filename, SAMPLE_RATE, recording)
    print(f"Saved to {filename}")
    return filename

if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else DURATION
    filename = sys.argv[2] if len(sys.argv) > 2 else "recorded.wav"
    record_audio(filename, duration)
