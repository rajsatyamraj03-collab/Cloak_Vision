import cv2
import numpy as np
import time


def warmup_camera(cap, num_frames=80):
    print(f"[Camera] Warming up — discarding {num_frames} frames...")
    for _ in range(num_frames):
        cap.read()
        cv2.waitKey(10)
    print("[Camera] Warmup complete.")


def check_camera_moved(frame, background, threshold=30.0):
    small_frame = cv2.resize(frame, (160, 120))
    small_bg = cv2.resize(background, (160, 120))
    diff = cv2.absdiff(small_frame, small_bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray_diff)) > threshold


def capture_background(cap, window_name="Invisible Cloak Output", countdown_secs=5, num_frames=60):
    print(f"[Background] Countdown starting ({countdown_secs}s)...")
    start = time.time()

    while True:
        elapsed = time.time() - start
        remaining = max(0, countdown_secs - int(elapsed))
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        display = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        text = str(remaining) if remaining > 0 else "GO!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)
        cv2.putText(display, text, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 220, 255), 8, cv2.LINE_AA)

        instruction = "Step completely out of frame!"
        (iw, _), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(display, instruction, ((w - iw) // 2, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        cv2.waitKey(1)
        if elapsed >= countdown_secs:
            break

    print(f"[Background] Capturing {num_frames} frames...")
    frames = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frames.append(frame.astype(np.float32))

        h, w = frame.shape[:2]
        pct = (i + 1) / num_frames
        bar_w = int(w * pct)
        canvas = frame.copy()
        cv2.rectangle(canvas, (0, h - 24), (w, h), (40, 40, 40), -1)
        cv2.rectangle(canvas, (0, h - 24), (bar_w, h), (0, 200, 0), -1)
        cv2.putText(canvas, f"Capturing background... {int(pct * 100)}%", (10, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)

    if not frames:
        raise RuntimeError("[Background] No frames captured — check your webcam.")

    background = np.mean(frames, axis=0).astype(np.uint8)
    print("[Background] Capture complete.")
    return background


def create_red_mask(hsv_frame, params):
    s_lo = params["s_lo"]
    v_lo = params["v_lo"]
    lower1 = np.array([params["h1_lo"], s_lo, v_lo], dtype=np.uint8)
    upper1 = np.array([params["h1_hi"], 255, 255], dtype=np.uint8)
    lower2 = np.array([params["h2_lo"], s_lo, v_lo], dtype=np.uint8)
    upper2 = np.array([params["h2_hi"], 255, 255], dtype=np.uint8)
    return cv2.bitwise_or(
        cv2.inRange(hsv_frame, lower1, upper1),
        cv2.inRange(hsv_frame, lower2, upper2)
    )


def fill_mask_holes(mask):
    h, w = mask.shape
    flood_seed = np.zeros((h + 2, w + 2), dtype=np.uint8)
    inverted = cv2.bitwise_not(mask)
    cv2.floodFill(inverted, flood_seed, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(inverted))


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    result = np.zeros_like(mask)
    result[labels == largest_idx] = 255
    return result


def fill_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, cv2.FILLED)
    return filled


def clean_mask(raw_mask):
    k5 = np.ones((5, 5), np.uint8)
    k7 = np.ones((7, 7), np.uint8)

    mask = cv2.GaussianBlur(raw_mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=4)
    mask = cv2.medianBlur(mask, 7)
    mask = fill_mask_holes(mask)
    mask = keep_largest_component(mask)

    if not np.any(mask):
        return mask

    mask = fill_largest_contour(mask)
    mask = cv2.dilate(mask, k5, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)


def feather_mask(mask, blur_radius=15):
    if blur_radius % 2 == 0:
        blur_radius += 1
    return cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)


def replace_background(frame, background, mask):
    feathered = feather_mask(mask, 15)
    inv_feathered = cv2.bitwise_not(feathered)
    bg_part = cv2.bitwise_and(background, background, mask=feathered)
    fg_part = cv2.bitwise_and(frame, frame, mask=inv_feathered)
    output = cv2.add(fg_part, bg_part)

    if np.any(mask):
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        margin = 10
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2 = min(output.shape[1], x + w + margin)
        y2 = min(output.shape[0], y + h + margin)

        roi_out = output[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi_out, cv2.COLOR_BGR2HSV)
        residual1 = cv2.inRange(hsv_roi, np.array([0, 100, 60], np.uint8), np.array([12, 255, 255], np.uint8))
        residual2 = cv2.inRange(hsv_roi, np.array([158, 100, 60], np.uint8), np.array([180, 255, 255], np.uint8))
        residual_mask = cv2.bitwise_or(residual1, residual2)
        roi_bg = background[y1:y2, x1:x2]
        roi_out[residual_mask > 0] = roi_bg[residual_mask > 0]
        output[y1:y2, x1:x2] = roi_out

    return output
