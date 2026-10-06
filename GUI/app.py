import os
import sys
import threading
import tkinter as tk

# Add project root (parent of this file's folder) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch  # noqa: F401  (preloads CUDA/cuDNN DLLs before onnxruntime)
from plyer import notification

from core.main import predict
from audio.capture import start_capture

# ============================================================
# CONFIGURATION
# ============================================================

# AI_THRESHOLD = 0.60          # ai_probability above this triggers alert
CONSECUTIVE_HITS_NEEDED = 2  # reduce false positives from one noisy chunk
# new change by aman
AI_THRESHOLD = 0.70          # ai_probability above this triggers alert
# CONSECUTIVE_HITS_NEEDED = 1  # reduce false positives from one noisy chunk
# *************************************
# ============================================================
# STATE
# ============================================================

stop_event = threading.Event()
consecutive_ai_hits = 0


def send_notification(ai_prob):
    notification.notify(
        title="Possible AI Voice Detected",
        message=f"This call is most likely AI-generated voice ({ai_prob*100:.1f}% confidence).",
        app_name="AI Voice Detector",
        timeout=10,
    )


def make_on_chunk(status_var):
    def on_chunk(chunk, filepath):
        global consecutive_ai_hits

        input_values = chunk.reshape(1, -1).astype(np.float32)
        ai_prob, human_prob = predict(input_values)

        print(f"{filepath}  ai={ai_prob:.3f}  human={human_prob:.3f}")
        status_var.set(f"ai={ai_prob:.2f}  human={human_prob:.2f}")

        if ai_prob >= AI_THRESHOLD:
            consecutive_ai_hits += 1
        else:
            consecutive_ai_hits = 0

        if consecutive_ai_hits >= CONSECUTIVE_HITS_NEEDED:
            send_notification(ai_prob)
            consecutive_ai_hits = 0  # avoid spamming repeated notifications

    return on_chunk


# ============================================================
# GUI
# ============================================================

def main():
    root = tk.Tk()
    root.title("AI Voice Detector")
    root.geometry("320x160")

    status_var = tk.StringVar(value="Idle")
    capture_thread = {"thread": None}

    def start():
        if capture_thread["thread"] and capture_thread["thread"].is_alive():
            return
        stop_event.clear()
        status_var.set("Listening...")
        t = threading.Thread(
            target=start_capture,
            args=(stop_event, make_on_chunk(status_var)),
            daemon=True,
        )
        capture_thread["thread"] = t
        t.start()
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")

    def stop():
        stop_event.set()
        status_var.set("Stopped")
        start_btn.config(state="normal")
        stop_btn.config(state="disabled")

    tk.Label(root, text="AI Voice Call Detector", font=("Segoe UI", 12, "bold")).pack(pady=10)

    start_btn = tk.Button(root, text="Start", width=15, command=start)
    start_btn.pack(pady=5)

    stop_btn = tk.Button(root, text="Stop", width=15, command=stop, state="disabled")
    stop_btn.pack(pady=5)

    tk.Label(root, textvariable=status_var).pack(pady=10)

    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()