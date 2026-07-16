import cv2
import numpy as np
import sys
import time
 
# ============================================================
# MODEL LOADING — torch MiDaS first, pseudo-depth fallback
# ============================================================
 
MODEL_TYPE = None
depth_model = None
depth_transform = None
 
 
def load_midas_torch():
    global depth_model, depth_transform, MODEL_TYPE
    try:
        import torch
        print("Loading MiDaS-small via torch hub (~80 MB on first run)...")
        depth_model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        )
        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        depth_transform = transforms.small_transform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_model.to(device)
        depth_model.eval()
        MODEL_TYPE = "torch"
        print(f"✓ MiDaS-small loaded (device: {device})")
        return True
    except Exception as e:
        print(f"torch/MiDaS failed: {e}")
        return False
 
 
def load_pseudo_fallback():
    """
    Gradient-based pseudo-depth when torch is unavailable.
    Not real depth estimation, but produces a plausible visualization.
    Install torch for accurate results: pip install torch torchvision
    """
    global MODEL_TYPE
    MODEL_TYPE = "pseudo"
    print("⚠️  Using pseudo-depth fallback (gradient/sharpness-based).")
    print("   Install torch for real depth: pip install torch torchvision")
    return True
 
 
print("\n" + "=" * 55)
print(" 📡  DepthMapper — Day 18")
print("=" * 55)
 
if not load_midas_torch():
    load_pseudo_fallback()
 
 
# ============================================================
# DEPTH INFERENCE
# ============================================================
 
def estimate_depth(frame_bgr):
    """
    Run depth estimation on a BGR frame.
    Returns a raw depth array (not yet normalized to 0-1).
    MiDaS output: higher value = farther away.
    """
    if MODEL_TYPE == "torch":
        import torch
        device = next(depth_model.parameters()).device
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = depth_transform(rgb).to(device)
 
        with torch.no_grad():
            prediction = depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
 
        return prediction.cpu().numpy()
 
    else:
        # Pseudo-depth: local sharpness via Laplacian
        # Sharp regions → near; blurry regions → far
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        depth = cv2.GaussianBlur(np.abs(laplacian), (51, 51), 0)
        return depth
 
 
def normalize_depth(depth):
    """Normalize any depth array to float 0-1."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth)
 
 
# ============================================================
# COLORMAP — TURBO chosen for best perceptual uniformity
# ============================================================
# TURBO: improved JET, perceptually uniform, near=blue, far=red
# Alternatives worth trying:
#   cv2.COLORMAP_PLASMA  — purple to yellow, elegant
#   cv2.COLORMAP_MAGMA   — dark/moody, good for presentation
#   cv2.COLORMAP_INFERNO — high contrast for nearby objects
#   cv2.COLORMAP_JET     — classic but has perceptual issues
 
COLORMAP = cv2.COLORMAP_TURBO
 
 
def colorize_depth(depth_normalized):
    """Convert normalized depth array to a colorized heatmap."""
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, COLORMAP)
 
 
# ============================================================
# CENTER-REGION DISTANCE ESTIMATION  (TODO #1 — complete)
# ============================================================
 
def estimate_center_depth(depth_normalized):
    """
    Average depth in the center 20% of the frame.
    Returns float 0.0 (near) → 1.0 (far).
 
    Mimics a single-zone rangefinder or camera autofocus sensor:
    reports distance for whatever is in the center of the FOV.
    In real ToF sensors this is one measurement cell.
    In LiDAR it would be the center beam cluster.
    """
    h, w = depth_normalized.shape
    cy, cx = h // 2, w // 2
    rh, rw = h // 10, w // 10  # ±10% → 20% total region each axis
 
    region = depth_normalized[cy - rh: cy + rh, cx - rw: cx + rw]
    return float(np.mean(region)) if region.size > 0 else 0.5
 
 
def center_depth_label(value):
    """Map 0-1 depth to a human-readable range label + color."""
    if value < 0.33:
        return "NEAR", (0, 255, 80)
    elif value < 0.66:
        return "MID", (0, 220, 255)
    else:
        return "FAR", (0, 80, 255)
 
 
# ============================================================
# POINT CLOUD CSV EXPORT
# ============================================================
 
def export_point_cloud(depth_normalized, frame_shape, filename="point_cloud.csv"):
    """
    Export depth map as a point cloud CSV.
    Columns: x (pixel col), y (pixel row), depth (0=near, 1=far)
 
    Real LiDAR gives XYZ in meters with reflectance.
    Here we use pixel coordinates + normalized depth — same data structure,
    different units. Ready to drop into Open3D or CloudCompare.
    """
    import csv
    h, w = frame_shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
 
    # Subsample every 4th pixel — full res makes a huge file
    step = 4
    xs_flat = xs.flatten()[::step]
    ys_flat = ys.flatten()[::step]
    ds_flat = depth_normalized.flatten()[::step]
 
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "depth"])
        for x, y, d in zip(xs_flat, ys_flat, ds_flat):
            writer.writerow([int(x), int(y), f"{d:.4f}"])
 
    print(f"✓ Saved {filename} ({len(xs_flat):,} points)")
    return filename
 
 
# ============================================================
# DEPTH HISTOGRAM
# ============================================================
 
def show_histogram(depth_normalized):
    """Plot depth value distribution across the frame."""
    import matplotlib.pyplot as plt
 
    plt.figure(figsize=(8, 4))
    plt.hist(depth_normalized.flatten(), bins=100,
             color="#1a7fbf", edgecolor="none", alpha=0.85)
    plt.axvline(x=0.33, color="#ff6b35", linestyle="--",
                linewidth=1.2, label="near/mid boundary")
    plt.axvline(x=0.66, color="#ff6b35", linestyle="--",
                linewidth=1.2, label="mid/far boundary")
    plt.xlabel("Depth (0 = near, 1 = far)")
    plt.ylabel("Pixel count")
    plt.title("Depth Distribution — Day 18 DepthMapper")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
 
 
# ============================================================
# MAIN LOOP
# ============================================================
 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)
 
print("\nDepthMapper running!")
print("Controls:  s = save CSV  |  h = histogram  |  q = quit")
print("Move your hand closer/further — watch the heatmap shift.\n")
 
last_depth = None
frame_count = 0
fps_start = time.time()
fps = 0.0
csv_saved_msg = ""
csv_msg_timer = 0
 
# Run depth every N frames — MiDaS-small is ~2-5 FPS on CPU, this keeps display smooth
PROCESS_EVERY = 3
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame_count += 1
 
    if frame_count % PROCESS_EVERY == 0:
        raw_depth = estimate_depth(frame)
        last_depth = normalize_depth(raw_depth)
 
        elapsed = time.time() - fps_start
        fps = PROCESS_EVERY / elapsed if elapsed > 0 else 0
        fps_start = time.time()
 
    if last_depth is None:
        continue
 
    # Build display
    heatmap = colorize_depth(last_depth)
    blended = cv2.addWeighted(frame, 0.25, heatmap, 0.75, 0)
 
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rw_box, rh_box = w // 10, h // 10
 
    # Center zone rectangle
    cv2.rectangle(
        blended,
        (cx - rw_box, cy - rh_box),
        (cx + rw_box, cy + rh_box),
        (255, 255, 255), 2
    )
    # Crosshair
    cv2.line(blended, (cx - 8, cy), (cx + 8, cy), (255, 255, 255), 1)
    cv2.line(blended, (cx, cy - 8), (cx, cy + 8), (255, 255, 255), 1)
 
    # Center depth readout
    center_val = estimate_center_depth(last_depth)
    label, color = center_depth_label(center_val)
 
    cv2.putText(blended, f"Center: {label}  ({center_val:.2f})",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(blended, f"FPS: {fps:.1f}  Model: {MODEL_TYPE}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1)
    cv2.putText(blended, "s=save CSV   h=histogram   q=quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)
 
    # Temporary CSV saved notification
    if csv_msg_timer > 0:
        cv2.putText(blended, csv_saved_msg, (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
        csv_msg_timer -= 1
 
    cv2.imshow("DepthMapper — Day 18", blended)
 
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break
    elif key == ord("s") and last_depth is not None:
        fname = export_point_cloud(last_depth, frame.shape)
        csv_saved_msg = f"✓ Saved: {fname}"
        csv_msg_timer = 60  # show for ~60 frames
    elif key == ord("h") and last_depth is not None:
        show_histogram(last_depth)
 
cap.release()
cv2.destroyAllWindows()