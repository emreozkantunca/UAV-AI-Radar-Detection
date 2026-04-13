import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: Define radar parameters (matching the IWR6843ISK)
# ============================================================

c = 3e8                    # speed of light in meters/second
fc = 60e9                  # carrier frequency: 60 GHz (our radar's starting freq)
B = 4e9                    # bandwidth: 4 GHz (60-64 GHz sweep)
T_chirp = 60e-6            # chirp duration: 60 microseconds
N_samples = 256            # ADC samples per chirp
N_chirps = 128             # number of chirps per frame
fs = N_samples / T_chirp   # sampling rate (samples/second)
S = B / T_chirp            # chirp slope (Hz per second)

# derived limits
max_range = (fs * c) / (2 * S)                       # max detectable range
range_res = c / (2 * B)                               # range resolution
max_velocity = c / (4 * fc * T_chirp)                  # max detectable velocity
velocity_res = c / (2 * fc * N_chirps * T_chirp)       # velocity resolution

print("=== Radar Parameters ===")
print(f"Max range:        {max_range:.1f} m")
print(f"Range resolution: {range_res*100:.1f} cm")
print(f"Max velocity:     {max_velocity:.1f} m/s")
print(f"Velocity resolution: {velocity_res:.2f} m/s")
print()

# ============================================================
# STEP 2: Define simulated targets
# ============================================================

targets = [
    {"range": 5.0, "velocity": 1.2, "rcs": 1.0, "label": "Person walking"},
    {"range": 12.0, "velocity": 8.0, "rcs": 5.0, "label": "Car moving"},
    {"range": 3.0, "velocity": 0.0, "rcs": 0.3, "label": "Static object"},
]

print("=== Simulated Targets ===")
for t in targets:
    print(f"  {t['label']}: range={t['range']}m, velocity={t['velocity']}m/s")
print()

# ============================================================
# STEP 3: Generate raw ADC data (simulating what the radar outputs)
# ============================================================

t_fast = np.arange(N_samples) / fs                  # fast-time axis (within one chirp)
adc_data = np.zeros((N_chirps, N_samples), dtype=complex)  # 2D matrix: chirps x samples

for target in targets:
    R = target["range"]
    v = target["velocity"]
    rcs = target["rcs"]

    # beat frequency: how far away the target is
    f_beat = (2 * S * R) / c

    # doppler frequency: how fast the target is moving
    f_doppler = (2 * v * fc) / c

    # generate signal for each chirp
    for chirp_idx in range(N_chirps):
        # the phase shifts slightly each chirp due to target motion
        phase_shift = 2 * np.pi * f_doppler * chirp_idx * T_chirp
        signal = rcs * np.exp(1j * 2 * np.pi * f_beat * t_fast + 1j * phase_shift)
        adc_data[chirp_idx, :] += signal

# add noise (real radar data is noisy)
noise_level = 0.1
noise = noise_level * (np.random.randn(*adc_data.shape) + 1j * np.random.randn(*adc_data.shape))
adc_data += noise

print(f"ADC data shape: {adc_data.shape}  (chirps x samples)")
print()

# ============================================================
# STEP 4: Range FFT (first FFT — along each chirp)
# ============================================================

range_fft = np.fft.fft(adc_data, axis=1)             # FFT along columns (fast-time)
range_fft = range_fft[:, :N_samples // 2]             # keep only positive frequencies

print(f"After Range FFT: {range_fft.shape}")

# ============================================================
# STEP 5: Doppler FFT (second FFT — across chirps)
# ============================================================

doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)  # FFT along rows (slow-time)

print(f"After Doppler FFT: {doppler_fft.shape}")
print()

# ============================================================
# STEP 6: Build the Range-Doppler Map
# ============================================================

range_doppler_map = 20 * np.log10(np.abs(doppler_fft) + 1e-10)  # convert to dB

# create axis labels
range_axis = np.linspace(0, max_range, N_samples // 2)
velocity_axis = np.linspace(-max_velocity, max_velocity, N_chirps)

# ============================================================
# STEP 7: Plot it
# ============================================================

plt.figure(figsize=(10, 6))
plt.imshow(
    range_doppler_map,
    aspect="auto",
    extent=[0, max_range, -max_velocity, max_velocity],
    origin="lower",
    cmap="jet",
)
plt.colorbar(label="Power (dB)")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Range-Doppler Map — Simulated IWR6843ISK")

# mark where targets should be
for t in targets:
    plt.plot(t["range"], t["velocity"], "wo", markersize=10, markeredgewidth=2)
    plt.annotate(
        t["label"],
        (t["range"], t["velocity"]),
        textcoords="offset points",
        xytext=(10, 10),
        color="white",
        fontsize=9,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("range_doppler_map.png", dpi=150)
plt.show()

print("Done! Saved range_doppler_map.png")
