import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.ndimage import label

# ============================================================
# STEP 1: Load the image
# ============================================================

img = Image.open("olympic-drones_017.jpg")
img_array = np.array(img)
img_height, img_width = img_array.shape[:2]

print(f"Image size: {img_width} x {img_height} pixels")

# ============================================================
# STEP 2: Radar parameters (IWR6843ISK)
# ============================================================

c = 3e8
fc = 60e9
B = 4e9
T_chirp = 60e-6
N_samples = 256
N_chirps = 128
fs = N_samples / T_chirp
S = B / T_chirp

max_range = (fs * c) / (2 * S)
max_velocity = c / (4 * fc * T_chirp)

# ============================================================
# STEP 3: MIMO antenna configuration
# ============================================================
#
# The IWR6843ISK has 3 TX and 4 RX antennas.
# Using MIMO, we get a "virtual array" of 3 x 4 = 12 elements.
#
# Why does this matter?
# A single antenna can't tell you direction — it just receives
# a signal. But with MULTIPLE antennas spaced apart, the same
# signal arrives at each antenna with a SLIGHTLY different phase.
# That phase difference tells you the angle the signal came from.
#
# More virtual antennas = better angular resolution.
# It's the same trick as having wider-spaced eyes giving
# you better depth perception.

N_tx = 3                              # transmit antennas
N_rx = 4                              # receive antennas
N_virtual = N_tx * N_rx               # 12 virtual elements
wavelength = c / fc                   # wavelength at 60 GHz (~5mm)
d = wavelength / 2                    # antenna spacing (half-wavelength)
N_angle_bins = 64                     # angle FFT size

# angular limits
max_angle = 60                        # IWR6843ISK FOV is ~±60 degrees

print(f"Virtual array: {N_virtual} elements")
print(f"Wavelength: {wavelength*1000:.1f} mm")
print(f"Antenna spacing: {d*1000:.1f} mm")
print(f"Angular FOV: ±{max_angle}°")
print()

# ============================================================
# STEP 4: Define scene targets with ANGLES
# ============================================================
#
# Now each target has a range, velocity, AND an angle.
# The angle is what we've been missing — it tells us
# WHERE in the image the target is, left to right.

scene_targets = [
    # Luge athlete: center-left of image, moving fast
    {"range": 5.5, "velocity": 12.0, "rcs": 2.0,
     "angle": -8.0,     # slightly left of center (degrees)
     "label": "Athlete"},

    # Camera operator: upper right, static
    {"range": 8.0, "velocity": 0.0, "rcs": 0.4,
     "angle": 35.0,     # right side of FOV
     "label": "Camera operator"},

    # Track wall: right side, static, weak
    {"range": 2.5, "velocity": 0.0, "rcs": 0.08,
     "angle": 45.0,     # far right
     "label": "Track wall"},
]

print("=== Scene Targets ===")
for t in scene_targets:
    print(f"  {t['label']}: range={t['range']}m, "
          f"vel={t['velocity']}m/s, angle={t['angle']}°")
print()

# ============================================================
# STEP 5: Generate raw ADC data WITH antenna array
# ============================================================
#
# Before we had a 2D matrix: (chirps x samples)
# Now we have a 3D matrix: (chirps x samples x virtual_antennas)
#
# Each virtual antenna receives the same signal but with a
# phase shift that depends on the target's angle.
# That phase shift is: 2π * d * sin(angle) / wavelength
# per antenna element.

t_fast = np.arange(N_samples) / fs
adc_data = np.zeros((N_chirps, N_samples, N_virtual), dtype=complex)

for target in scene_targets:
    R = target["range"]
    v = target["velocity"]
    rcs = target["rcs"]
    theta = np.radians(target["angle"])    # convert degrees to radians

    f_beat = (2 * S * R) / c
    f_doppler = (2 * v * fc) / c

    for chirp_idx in range(N_chirps):
        phase_shift = 2 * np.pi * f_doppler * chirp_idx * T_chirp
        base_signal = rcs * np.exp(1j * 2 * np.pi * f_beat * t_fast + 1j * phase_shift)

        # for each virtual antenna, add a phase shift based on angle
        for ant_idx in range(N_virtual):
            # this is the key line: phase difference between antennas
            # depends on antenna spacing (d), signal angle (theta),
            # and which antenna we're on (ant_idx)
            spatial_phase = 2 * np.pi * d * ant_idx * np.sin(theta) / wavelength
            adc_data[chirp_idx, :, ant_idx] += base_signal * np.exp(1j * spatial_phase)

# add noise to each antenna
noise_level = 0.1
adc_data += noise_level * (np.random.randn(*adc_data.shape)
                           + 1j * np.random.randn(*adc_data.shape))

print(f"ADC data shape: {adc_data.shape}  (chirps x samples x antennas)")

# ============================================================
# STEP 6: 3D FFT processing — Range, Doppler, AND Angle
# ============================================================

# FFT 1: Range (along samples axis)
range_fft = np.fft.fft(adc_data, axis=1)[:, :N_samples // 2, :]

# FFT 2: Doppler (along chirps axis)
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

# FFT 3: Angle (along antenna axis) — THIS IS THE NEW ONE
# We zero-pad to N_angle_bins for finer angular resolution
angle_fft = np.fft.fftshift(
    np.fft.fft(doppler_fft, n=N_angle_bins, axis=2),
    axes=2
)

rdm_3d = np.abs(angle_fft)
rdm_3d_db = 20 * np.log10(rdm_3d + 1e-10)

print(f"3D cube shape: {rdm_3d.shape}  (doppler x range x angle)")
print()

# ============================================================
# STEP 7: CFAR detection on the 3D cube
# ============================================================

# collapse angle dimension to get range-doppler map for CFAR
rdm_2d = np.max(rdm_3d_db, axis=2)   # take max across angles

def cfar_2d(rdm_db, guard_cells, training_cells, threshold_factor_db):
    n_doppler, n_range = rdm_db.shape
    guard_d, guard_r = guard_cells
    train_d, train_r = training_cells
    window_d = guard_d + train_d
    window_r = guard_r + train_r

    detection_mask = np.zeros_like(rdm_db, dtype=bool)

    for d in range(window_d, n_doppler - window_d):
        for r in range(window_r, n_range - window_r):
            window = rdm_db[d - window_d : d + window_d + 1,
                            r - window_r : r + window_r + 1]
            mask = np.ones_like(window, dtype=bool)
            gd_start = train_d
            gd_end = train_d + 2 * guard_d + 1
            gr_start = train_r
            gr_end = train_r + 2 * guard_r + 1
            mask[gd_start:gd_end, gr_start:gr_end] = False
            noise_estimate = np.mean(window[mask])
            threshold = noise_estimate + threshold_factor_db
            if rdm_db[d, r] > threshold:
                detection_mask[d, r] = True

    return detection_mask

print("Running CFAR on Range-Doppler map...")
detection_mask = cfar_2d(rdm_2d, (2, 2), (5, 5), 13)

# cluster and extract detections
labeled_array, num_raw = label(detection_mask)

detections = []
for det_id in range(1, num_raw + 1):
    cluster_mask = labeled_array == det_id
    cluster_values = rdm_2d * cluster_mask
    peak_idx = np.unravel_index(np.argmax(cluster_values), rdm_2d.shape)
    cluster_size = np.sum(cluster_mask)

    if cluster_size < 2:
        continue

    doppler_idx, range_idx = peak_idx
    det_range = range_idx * max_range / (N_samples // 2)
    det_velocity = (doppler_idx - N_chirps // 2) * (2 * max_velocity) / N_chirps
    det_power = rdm_2d[peak_idx]

    # ============================================================
    # STEP 8: ANGLE ESTIMATION — the missing piece!
    # ============================================================
    #
    # For each detected target, look at the angle FFT at that
    # (range, doppler) bin and find which angle has the most energy.
    # That's the direction the target is in.

    angle_spectrum = rdm_3d_db[doppler_idx, range_idx, :]
    angle_idx = np.argmax(angle_spectrum)

    # convert angle bin index to actual angle in degrees
    # the angle FFT maps sin(theta), so we need arcsin
    angle_bins = np.linspace(-1, 1, N_angle_bins)   # sin(theta) range
    sin_theta = angle_bins[angle_idx]
    sin_theta_clipped = np.clip(sin_theta, -1, 1)
    det_angle = np.degrees(np.arcsin(sin_theta_clipped))

    detections.append({
        "range": det_range,
        "velocity": det_velocity,
        "angle": det_angle,
        "power": det_power,
        "cluster_size": cluster_size,
    })

print(f"\n=== DETECTIONS WITH ANGLE: {len(detections)} targets ===")
for i, det in enumerate(detections):
    print(f"  Target {i+1}: range={det['range']:.1f}m, "
          f"vel={det['velocity']:.1f}m/s, "
          f"angle={det['angle']:.1f}°, "
          f"power={det['power']:.1f}dB")

# ============================================================
# STEP 9: Classify targets
# ============================================================

for det in detections:
    speed = abs(det["velocity"])
    if speed > 10:
        det["class"] = "FAST MOVER"
        det["color"] = "red"
        det["threat"] = "HIGH"
    elif speed > 2:
        det["class"] = "MOVING"
        det["color"] = "orange"
        det["threat"] = "MEDIUM"
    elif speed > 0.3:
        det["class"] = "SLOW"
        det["color"] = "yellow"
        det["threat"] = "LOW"
    else:
        det["class"] = "STATIC"
        det["color"] = "cyan"
        det["threat"] = "NONE"

# ============================================================
# STEP 10: Map detections to image using REAL angle + range
# ============================================================
#
# NOW we can do this properly!
# - Angle tells us the horizontal position (left/right)
# - Range tells us the vertical position (near/far)
# No more guessing from velocity!

def radar_to_image_with_angle(det_range, det_angle, max_r, max_ang, img_w, img_h):
    """
    Map (range, angle) to image pixel coordinates.

    angle: negative = left side of image, positive = right side
    range: small = bottom of image (close), large = top (far)
    """
    # angle -> x position
    # normalize angle to [-1, 1] then map to image width
    x_frac = 0.5 + (det_angle / max_ang) * 0.5
    x_frac = np.clip(x_frac, 0.05, 0.95)

    # range -> y position (farther = higher in frame)
    y_frac = 1.0 - (det_range / max_r)
    y_frac = np.clip(y_frac, 0.05, 0.95)

    return int(x_frac * img_w), int(y_frac * img_h)

# ============================================================
# STEP 11: Visualize — 4 panels
# ============================================================

fig = plt.figure(figsize=(24, 12))

# Panel 1: Range-Doppler map
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(rdm_2d, aspect="auto",
           extent=[0, max_range, -max_velocity, max_velocity],
           origin="lower", cmap="jet")
ax1.set_xlabel("Range (m)")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title("Range-Doppler Map")
for det in detections:
    ax1.plot(det["range"], det["velocity"], "w+", markersize=15, markeredgewidth=3)

# Panel 2: Range-Angle map (NEW!)
# collapse doppler to show range vs angle
range_angle_map = np.max(rdm_3d_db, axis=0)   # max across doppler
angle_axis = np.degrees(np.arcsin(np.linspace(-1, 1, N_angle_bins)))

ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(range_angle_map.T, aspect="auto",
           extent=[0, max_range, -90, 90],
           origin="lower", cmap="jet")
ax2.set_xlabel("Range (m)")
ax2.set_ylabel("Angle (degrees)")
ax2.set_title("Range-Angle Map (Angle of Arrival)")
ax2.set_ylim(-60, 60)
for det in detections:
    ax2.plot(det["range"], det["angle"], "w+", markersize=15, markeredgewidth=3)
    ax2.annotate(f'{det["angle"]:.0f}°',
                 (det["range"], det["angle"]),
                 textcoords="offset points", xytext=(10, 10),
                 color="white", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

# Panel 3: Angle spectrum for each detection
ax3 = fig.add_subplot(2, 2, 3)
angle_sweep = np.linspace(-90, 90, N_angle_bins)
for det in detections:
    range_idx = int(det["range"] / max_range * (N_samples // 2))
    doppler_idx = int(det["velocity"] / (2 * max_velocity) * N_chirps + N_chirps // 2)
    range_idx = np.clip(range_idx, 0, N_samples // 2 - 1)
    doppler_idx = np.clip(doppler_idx, 0, N_chirps - 1)
    spectrum = rdm_3d_db[doppler_idx, range_idx, :]
    ax3.plot(angle_sweep, spectrum, linewidth=2, label=f"{det['class']} ({det['angle']:.0f}°)")
    ax3.axvline(x=det["angle"], linestyle="--", alpha=0.5)

ax3.set_xlabel("Angle (degrees)")
ax3.set_ylabel("Power (dB)")
ax3.set_title("Angle Spectrum per Detection")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-60, 60)

# Panel 4: Camera view with PROPERLY placed detections
ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(img_array)
ax4.set_title("Camera View — Angle-Based Detection Overlay")
ax4.axis("off")

for det in detections:
    color = det["color"]
    speed = abs(det["velocity"])

    if speed > 5:
        box_size = 100
    elif speed > 0.5:
        box_size = 70
    else:
        box_size = 50

    # use the REAL angle-based mapping now
    px, py = radar_to_image_with_angle(
        det["range"], det["angle"],
        max_range, max_angle, img_width, img_height
    )

    # bounding box
    rect = patches.Rectangle(
        (px - box_size, py - box_size), box_size * 2, box_size * 2,
        linewidth=3, edgecolor=color, facecolor="none"
    )
    ax4.add_patch(rect)

    # corner brackets
    bracket_len = 20
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        bx = px + dx * box_size
        by = py + dy * box_size
        ax4.plot([bx, bx + dx * bracket_len], [by, by], color=color, linewidth=2)
        ax4.plot([bx, bx], [by, by + dy * bracket_len], color=color, linewidth=2)

    # label
    label_text = (f"{det['class']}\n"
                  f"R:{det['range']:.1f}m  V:{abs(det['velocity']):.1f}m/s\n"
                  f"Angle: {det['angle']:.1f}°  Threat: {det['threat']}")
    ax4.annotate(label_text, (px, py - box_size - 10),
                 ha="center", va="bottom",
                 color=color, fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.85))

# radar HUD
info_text = (f"RADAR: IWR6843ISK 60GHz\n"
             f"MODE: 3D FFT + CA-CFAR\n"
             f"MIMO: {N_tx}TX x {N_rx}RX = {N_virtual} virtual elements\n"
             f"Targets: {len(detections)}\n"
             f"FOV: ±{max_angle}°")
ax4.text(10, img_height - 20, info_text,
         fontsize=9, color="lime", fontfamily="monospace",
         verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="black", alpha=0.8))

plt.tight_layout()
plt.savefig("angle_of_arrival_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n=== PIPELINE COMPLETE ===")
print(f"1. Range FFT     → distance to target")
print(f"2. Doppler FFT   → speed of target")
print(f"3. Angle FFT     → direction of target")
print(f"4. CFAR detection → find targets in noise")
print(f"5. Classification → label by threat level")
print(f"\nDone! Saved angle_of_arrival_overlay.png")
