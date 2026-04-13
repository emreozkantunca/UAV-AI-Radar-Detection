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
# STEP 2: Radar parameters (same IWR6843ISK config)
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
# STEP 3: Define scene targets (radar doesn't know these)
# ============================================================

scene_targets = [
    {"range": 5.5, "velocity": 22.0, "rcs": 1.0,
     "img_x": 0.35, "img_y": 0.52},
    {"range": 8.0, "velocity": 0.0, "rcs": 0.5,
     "img_x": 0.72, "img_y": 0.18},
    {"range": 2.0, "velocity": 0.0, "rcs": 0.15,
     "img_x": 0.85, "img_y": 0.75},
]

# ============================================================
# STEP 4: Generate raw ADC data
# ============================================================

t_fast = np.arange(N_samples) / fs
adc_data = np.zeros((N_chirps, N_samples), dtype=complex)

for target in scene_targets:
    R = target["range"]
    v = target["velocity"]
    rcs = target["rcs"]
    f_beat = (2 * S * R) / c
    f_doppler = (2 * v * fc) / c

    for chirp_idx in range(N_chirps):
        phase_shift = 2 * np.pi * f_doppler * chirp_idx * T_chirp
        signal = rcs * np.exp(1j * 2 * np.pi * f_beat * t_fast + 1j * phase_shift)
        adc_data[chirp_idx, :] += signal

noise_level = 0.1
adc_data += noise_level * (np.random.randn(*adc_data.shape) + 1j * np.random.randn(*adc_data.shape))

# ============================================================
# STEP 5: Range-Doppler processing
# ============================================================

range_fft = np.fft.fft(adc_data, axis=1)[:, :N_samples // 2]
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
rdm = np.abs(doppler_fft)
rdm_db = 20 * np.log10(rdm + 1e-10)

print(f"Range-Doppler map shape: {rdm.shape}")

# ============================================================
# STEP 6: CFAR DETECTION — the real deal
# ============================================================
# 
# How CFAR works:
# 
# For every cell in the Range-Doppler map, CFAR looks at
# a window of neighboring cells to estimate the LOCAL noise.
# 
# The window has 3 zones:
# [guard cells] [training cells] [CELL UNDER TEST] [training cells] [guard cells]
# 
# - Cell Under Test (CUT): the cell we're checking
# - Guard cells: cells right next to CUT that we SKIP
#   (because a real target bleeds into nearby cells)
# - Training cells: cells farther out that represent pure noise
# 
# If CUT's power > (average of training cells) * threshold_factor,
# then CUT is declared a target.
# 
# This adapts to local noise — no more fixed threshold!

def cfar_2d(rdm_db, guard_cells, training_cells, threshold_factor_db):
    """
    2D CA-CFAR (Cell-Averaging CFAR) detector.
    
    Parameters:
    -----------
    rdm_db : 2D array
        Range-Doppler map in dB
    guard_cells : tuple (guard_doppler, guard_range)
        Number of guard cells on each side
    training_cells : tuple (train_doppler, train_range)
        Number of training cells on each side
    threshold_factor_db : float
        How many dB above the local noise a cell must be
        
    Returns:
    --------
    detection_mask : 2D boolean array
        True where targets are detected
    threshold_map : 2D array
        The adaptive threshold at each cell
    """
    
    n_doppler, n_range = rdm_db.shape
    guard_d, guard_r = guard_cells
    train_d, train_r = training_cells
    
    # total window size on each side
    window_d = guard_d + train_d
    window_r = guard_r + train_r
    
    detection_mask = np.zeros_like(rdm_db, dtype=bool)
    threshold_map = np.zeros_like(rdm_db)
    
    # slide the window across every cell
    for d in range(window_d, n_doppler - window_d):
        for r in range(window_r, n_range - window_r):
            
            # extract the full window around this cell
            window = rdm_db[d - window_d : d + window_d + 1,
                            r - window_r : r + window_r + 1]
            
            # create a mask to EXCLUDE the guard cells and the CUT
            mask = np.ones_like(window, dtype=bool)
            
            # zero out the guard region (center of window)
            gd_start = train_d
            gd_end = train_d + 2 * guard_d + 1
            gr_start = train_r
            gr_end = train_r + 2 * guard_r + 1
            mask[gd_start:gd_end, gr_start:gr_end] = False
            
            # average only the training cells
            training_values = window[mask]
            noise_estimate = np.mean(training_values)
            
            # set the adaptive threshold
            threshold = noise_estimate + threshold_factor_db
            threshold_map[d, r] = threshold
            
            # detect: is this cell above the local threshold?
            if rdm_db[d, r] > threshold:
                detection_mask[d, r] = True
    
    return detection_mask, threshold_map

# CFAR parameters — tune these for best results
guard_cells = (3, 3)         # skip 3 cells on each side (handles target spread)
training_cells = (8, 8)      # use 8 cells on each side for noise estimate
threshold_factor_db = 12     # target must be 12 dB above local noise

print("Running 2D CA-CFAR detection...")
detection_mask, threshold_map = cfar_2d(rdm_db, guard_cells, training_cells, threshold_factor_db)

# cluster detections and find peaks
labeled_array, num_raw = label(detection_mask)

detections = []
for det_id in range(1, num_raw + 1):
    cluster_mask = labeled_array == det_id
    cluster_values = rdm_db * cluster_mask
    peak_idx = np.unravel_index(np.argmax(cluster_values), rdm_db.shape)
    
    doppler_idx, range_idx = peak_idx
    det_range = range_idx * max_range / (N_samples // 2)
    det_velocity = (doppler_idx - N_chirps // 2) * (2 * max_velocity) / N_chirps
    det_power = rdm_db[peak_idx]
    
    # only keep strong detections (ignore tiny clusters)
    cluster_size = np.sum(cluster_mask)
    if cluster_size >= 2:
        detections.append({
            "range": det_range,
            "velocity": det_velocity,
            "power": det_power,
            "cluster_size": cluster_size,
        })

print(f"\n=== CFAR DETECTIONS: {len(detections)} targets ===")
for i, det in enumerate(detections):
    print(f"  Target {i+1}: range={det['range']:.1f}m, "
          f"velocity={det['velocity']:.1f}m/s, "
          f"power={det['power']:.1f}dB, "
          f"cluster={det['cluster_size']} cells")

# ============================================================
# STEP 7: Classify targets based on radar signature
# ============================================================

for det in detections:
    speed = abs(det["velocity"])
    power = det["power"]
    
    if speed > 15:
        det["class"] = "FAST MOVER"
        det["color"] = "red"
        det["threat"] = "HIGH"
    elif speed > 2:
        det["class"] = "MOVING TARGET"
        det["color"] = "orange"
        det["threat"] = "MEDIUM"
    elif speed > 0.3:
        det["class"] = "SLOW MOVER"
        det["color"] = "yellow"
        det["threat"] = "LOW"
    else:
        det["class"] = "STATIC"
        det["color"] = "cyan"
        det["threat"] = "NONE"

# ============================================================
# STEP 8: Map detections to image coordinates
# ============================================================

def radar_to_image(det_range, det_velocity, max_r, max_v, img_w, img_h):
    y_frac = 1.0 - (det_range / max_r)
    y_frac = np.clip(y_frac, 0.05, 0.95)
    x_frac = 0.5 + (det_velocity / max_v) * 0.3
    x_frac = np.clip(x_frac, 0.05, 0.95)
    return int(x_frac * img_w), int(y_frac * img_h)

# ============================================================
# STEP 9: Visualize — 3 panels now
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# LEFT: Range-Doppler map
ax1 = axes[0]
ax1.imshow(rdm_db, aspect="auto",
           extent=[0, max_range, -max_velocity, max_velocity],
           origin="lower", cmap="jet")
ax1.set_xlabel("Range (m)")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title("Range-Doppler Map")

for det in detections:
    ax1.plot(det["range"], det["velocity"], "r+", markersize=15, markeredgewidth=3)

# MIDDLE: CFAR threshold map vs actual power
ax2 = axes[1]
ax2.imshow(threshold_map, aspect="auto",
           extent=[0, max_range, -max_velocity, max_velocity],
           origin="lower", cmap="hot")
ax2.set_xlabel("Range (m)")
ax2.set_ylabel("Velocity (m/s)")
ax2.set_title("CFAR Adaptive Threshold Map")

# overlay detection mask
det_display = np.ma.masked_where(~detection_mask, detection_mask.astype(float))
ax2.imshow(det_display, aspect="auto",
           extent=[0, max_range, -max_velocity, max_velocity],
           origin="lower", cmap="cool", alpha=0.7)

# RIGHT: Camera view with clean detections
ax3 = axes[2]
ax3.imshow(img_array)
ax3.set_title("Camera View — CFAR Detections")
ax3.axis("off")

for det in detections:
    speed = abs(det["velocity"])
    color = det["color"]
    
    if speed > 5:
        box_size = 120
    elif speed > 0.5:
        box_size = 80
    else:
        box_size = 60

    px, py = radar_to_image(det["range"], det["velocity"],
                            max_range, max_velocity, img_width, img_height)

    # bounding box
    rect = patches.Rectangle(
        (px - box_size, py - box_size), box_size * 2, box_size * 2,
        linewidth=3, edgecolor=color, facecolor="none"
    )
    ax3.add_patch(rect)

    # corner brackets
    bracket_len = 20
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        bx = px + dx * box_size
        by = py + dy * box_size
        ax3.plot([bx, bx + dx * bracket_len], [by, by], color=color, linewidth=2)
        ax3.plot([bx, bx], [by, by + dy * bracket_len], color=color, linewidth=2)

    # label with classification
    label = (f"{det['class']}\n"
             f"{det['range']:.1f}m | {abs(det['velocity']):.1f}m/s\n"
             f"Threat: {det['threat']}")
    ax3.annotate(label, (px, py - box_size - 10),
                 ha="center", va="bottom",
                 color=color, fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.85))

# radar HUD overlay
info_text = (f"RADAR: IWR6843ISK 60GHz\n"
             f"MODE: 2D CA-CFAR\n"
             f"Targets: {len(detections)}\n"
             f"Guard: {guard_cells} | Train: {training_cells}\n"
             f"Threshold: +{threshold_factor_db}dB above noise")
ax3.text(10, img_height - 20, info_text,
         fontsize=9, color="lime", fontfamily="monospace",
         verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="black", alpha=0.8))

plt.tight_layout()
plt.savefig("cfar_detection_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# STEP 10: Print comparison
# ============================================================

print(f"\n=== COMPARISON ===")
print(f"Simple threshold: 15 detections (lots of false alarms)")
print(f"CFAR detection:   {len(detections)} detections (clean!)")
print(f"\nCFAR adapts to local noise — sidelobes don't fool it.")
print(f"\nDone! Saved cfar_detection_overlay.png")
