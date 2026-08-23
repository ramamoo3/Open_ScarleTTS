import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS

OUT_DIR = Path(__file__).resolve().parent

LINES = [
    "Yes, I'm on it.",
    "All systems are online and ready.",
    "[laugh] Okay, that actually worked!",
    "I knew you could do it. Well done.",
    "Hmm... let me think about that for a second.",
    "Careful! That wire is still live.",
    "Good night. I'll keep watch.",
    "You're asking me? [laugh] Then yes, absolutely.",
    "Something moved outside. Stay here.",
    "Mission complete. Bringing her home.",
]


def load_model():
    if torch.backends.mps.is_available():
        try:
            print("device: mps")
            return ChatterboxTurboTTS.from_pretrained("mps", nano=True), "mps"
        except Exception as e:
            print(f"mps load failed ({e!r}); falling back to cpu")
    print("device: cpu")
    return ChatterboxTurboTTS.from_pretrained("cpu", nano=True), "cpu"


def main():
    t_wall0 = time.perf_counter()
    model, device = load_model()
    sr = model.sr
    print(f"model loaded on {device}, sr={sr}, nano weights ready")

    rows = []
    failures = []
    notes = []

    for i, text in enumerate(LINES, start=1):
        fname = f"line_{i:02d}.wav"
        t0 = time.perf_counter()
        try:
            wav = model.generate(text)
            gen_s = time.perf_counter() - t0
            audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
            sf.write(str(OUT_DIR / fname), audio, sr)
            dur_s = len(audio) / sr
            rtf = gen_s / dur_s if dur_s > 0 else float("nan")
            rows.append(
                {"file": fname, "text": text, "dur_s": round(dur_s, 3),
                 "_gen_s": round(gen_s, 3), "_rtf": round(rtf, 3)}
            )
            print(f"[{i:02d}/10] {fname}  dur={dur_s:.2f}s gen={gen_s:.2f}s rtf={rtf:.3f}")
        except Exception as e:
            gen_s = time.perf_counter() - t0
            failures.append((fname, text, repr(e)))
            print(f"[{i:02d}/10] FAILED ({gen_s:.2f}s): {e!r}")

    total_wall = time.perf_counter() - t_wall0

    with open(OUT_DIR / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"file": r["file"], "text": r["text"], "dur_s": r["dur_s"]}) + "\n")

    name_w = max(len(r["text"]) for r in rows) + 2
    print("\n=== BASELINE RESULTS ===")
    print(f"{'line':<5} {'text':<{name_w}} {'dur_s':>7} {'gen_s':>7} {'rtf':>7}")
    for i, r in enumerate(rows, start=1):
        print(f"{i:<5} {r['text']:<{name_w}} {r['dur_s']:>7.2f} {r['_gen_s']:>7.2f} {r['_rtf']:>7.3f}")

    rtfs = [r["_rtf"] for r in rows]
    mean_rtf = float(np.mean(rtfs)) if rtfs else float("nan")
    print("\n=== 2x SPEED CHECK ===")
    print(f"mean rtf       : {mean_rtf:.3f}")
    print(f"total wall time: {total_wall:.2f}s (incl. model load)")
    print(f"device         : {device}")
    print(f"files written  : {len(rows)} / {len(LINES)}")
    if failures:
        print("failed lines   :")
        for fn, tx, err in failures:
            print(f"  - {fn}: {tx} -> {err}")
    else:
        print("failed lines   : none")


if __name__ == "__main__":
    main()
