# Standard Operating Procedure (SOP) 🛠️

This document outlines instructions for contributing to, modifying, and optimizing `Open_ScarleTTS`.

---

## 1. Customizing and Adding Emotion Profiles

All emotion profiles live in `open_scarletts/tts.py` under `EmotionTTS.DEFAULT_PROFILES`. Each profile has two keys:
* `"speed"`: Speaking rate multiplier fed to Kokoro natively (> 1.0 faster, < 1.0 slower).
* `"pitch"`: Pitch shift multiplier applied after synthesis via resampling (> 1.0 higher, < 1.0 lower). *(Renamed from `pitch_mod` in v0.2.0; `pitch_mod` is still accepted as a legacy alias.)*

### Adding a New Emotion (Recommended)
No source edits needed — register it at runtime:
```python
tts.register_emotion("excited", speed=1.2, pitch=1.08)
```
Valid ranges: `speed` in [0.5, 2.0], `pitch` in [0.5, 2.0].

### Adding a New Emotion (Permanent)
To ship an emotion with the package, add it to `DEFAULT_PROFILES`:
```python
"excited": {"speed": 1.2, "pitch": 1.08}
```
Pitch shifting and output clipping are applied automatically for every emotion; only add extra NumPy DSP inside `_apply_dsp()` if your emotion needs bespoke treatment (e.g., custom gain):
```python
if emotion == "excited":
    samples = samples * 1.1  # slight volume lift; final clip happens after this
```

---

## 2. DSP Manipulation Guide

Since we want to keep the package lightweight and avoid dependencies like `librosa` or `scipy`, all custom audio adjustments are performed directly on the NumPy samples array.

### Simulated Whisper (Amplitude Gating & Noise Injection)
Whispers lack vocal cord vibration, leaving only turbulent airflow. We simulate this by:
1. Gating/damping the vocal amplitude by multiplying the samples by `0.4`.
2. Injecting a faint layer of white noise (Gaussian distribution) to replicate the soft breathy texture:
   ```python
   noise = np.random.normal(loc=0.0, scale=0.005, size=samples.shape)
   samples = samples + noise
   ```

### Pitch Modulation (Resampling)
Implemented in `EmotionTTS._pitch_shift()`. Interpolating the waveform onto a *shorter* timeline raises the playback pitch (and compresses duration by `1/factor`); a longer timeline lowers it. To keep the overall speaking pace on target, `generate()` compensates by passing `speed / pitch` to Kokoro's native speed control before shifting.
```python
# Raise pitch by 'factor' (> 1.0 = higher): resample to fewer points
new_len = max(2, int(round(len(samples) / factor)))
x_old = np.linspace(0.0, 1.0, len(samples))
x_new = np.linspace(0.0, 1.0, new_len)
shifted = np.interp(x_new, x_old, samples).astype(np.float32)
```
Note: every DSP pass ends with `np.clip(samples, -1.0, 1.0)` so downstream DACs never receive out-of-range values.

---

## 3. Optimizing Voice Bin Files for Edge Devices

The default `voices-v1.0.bin` contains embeddings for all available male and female voices (Bella, Heart, Sarah, Michael, etc.), totaling about **30 MB** in size.

For single-voice toys (like a dedicated smart doll or Siri-like assistant), you only need **one** voice. Keeping all of them wastes precious RAM on the Raspberry Pi.

Run this script to prune the binary down to just your selected voice (e.g., `af_heart`), reducing the asset size to **~2.5 MB**:

```python
import numpy as np
import pickle

original_bin_path = "voices-v1.0.bin"
pruned_bin_path = "voices-v1.0-single.bin"
selected_voice = "af_heart"

# 1. Load the original voice index dictionary
print(f"Loading {original_bin_path}...")
voices_dict = np.load(original_bin_path, allow_pickle=True)

# Convert to standard dictionary if it's an Npy object
if hasattr(voices_dict, "item"):
    voices_dict = voices_dict.item()

# 2. Extract only the target voice embedding
if selected_voice in voices_dict:
    pruned_dict = {selected_voice: voices_dict[selected_voice]}
    print(f"Isolated voice: '{selected_voice}'")
else:
    raise ValueError(f"Voice '{selected_voice}' not found in the file! Available: {list(voices_dict.keys())}")

# 3. Save the single-voice dictionary
with open(pruned_bin_path, "wb") as f:
    pickle.dump(pruned_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved optimized voice binary to {pruned_bin_path} (Size reduced by ~90%!)")
```
