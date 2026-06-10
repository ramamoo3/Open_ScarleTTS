# Standard Operating Procedure (SOP) 🛠️

This document outlines instructions for contributing to, modifying, and optimizing `Open_ScarleTTS`.

---

## 1. Customizing and Adding Emotion Profiles

All emotion profiles are defined in `open_scarletts/tts.py` under the `self.emotion_profiles` dictionary. 

### Modifying Speed
You can adjust the speed of any profile by changing its `"speed"` value:
* Speed > `1.0` makes speech faster (suitable for angry/excited tones).
* Speed < `1.0` slows down speech (suitable for sad/ponderous tones).

### Adding a New Emotion Profile
To add a new emotion (e.g. `excited`):
1. Open `open_scarletts/tts.py`.
2. Add your new profile to the `self.emotion_profiles` dictionary:
   ```python
   "excited": {"speed": 1.2, "pitch_mod": 1.08}
   ```
3. Update `generate()` if your emotion requires custom NumPy DSP modifiers (e.g., volume gain):
   ```python
   if emotion == "excited":
       samples = samples * 1.2  # Increase output volume gain slightly
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
While Kokoro natively supports speed modification, pitch can be tweaked via interpolation:
```python
# Modulate pitch up by interpolating (faster rate = higher pitch)
duration = len(samples) / sample_rate
new_samples = np.interp(
    np.linspace(0, duration, int(len(samples) * pitch_mod)),
    np.linspace(0, duration, len(samples)),
    samples
)
```

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
