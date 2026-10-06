import os
import sys
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf

# Add project root (parent of this file's folder) to sys.path so `detector` is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.preprocessing import preprocess_audio

SAMPLE_RATE = 16000
# CHUNK_SECONDS = 3
# change by aman
CHUNK_SECONDS = 5
# ******************
DEVICE_NAME = "CABLE Output"  # your virtual cable's input device name
OUTPUT_DIR = "./audio/captured_chunks"

audio_queue = queue.Queue()


def find_device_index(name=DEVICE_NAME):
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    raise ValueError(f"No input device matching '{name}' found.")


def _callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def start_capture(stop_event, on_chunk=None):
    """
    Captures audio from DEVICE_NAME in CHUNK_SECONDS windows, preprocesses
    each chunk, saves it as a .wav in OUTPUT_DIR, and optionally calls
    on_chunk(processed_chunk, filepath) (e.g. to run prediction on it).

    stop_event: threading.Event() -- set it to stop capture.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device_index = find_device_index()
    chunk_samples = SAMPLE_RATE * CHUNK_SECONDS
    buffer = np.zeros((0,), dtype=np.float32)

    with sd.InputStream(
        device=device_index,
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        callback=_callback,
    ):
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, data[:, 0]])

            while len(buffer) >= chunk_samples:
                raw_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]
                # change by aman
                SILENCE_THRESHOLD = 0.005
                rms = np.sqrt(np.mean(raw_chunk**2))
                if rms < SILENCE_THRESHOLD:
                    # Agar audio me koi bol nahi raha hai toh AI prediction skip karo
                    continue
                # **************************
                processed_chunk = preprocess_audio(raw_chunk, SAMPLE_RATE)

                filename = os.path.join(OUTPUT_DIR, f"chunk_{int(time.time() * 1000)}.wav")
                sf.write(filename, processed_chunk, SAMPLE_RATE)
                print(f"Saved: {filename}")

                if on_chunk:
                    on_chunk(processed_chunk, filename)


if __name__ == "__main__":
    import threading
    stop_event = threading.Event()
    print("Press Ctrl+C to stop.")
    try:
        start_capture(stop_event)
    except KeyboardInterrupt:
        stop_event.set()