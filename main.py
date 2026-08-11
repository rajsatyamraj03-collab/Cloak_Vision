import cv2
import numpy as np

from utils import (
    warmup_camera,
    capture_background,
    create_red_mask,
    clean_mask,
    replace_background,
    check_camera_moved,
)

OUTPUT_WINDOW   = "Invisible Cloak Output"
CALIB_WINDOW    = "HSV Calibration"
MASK_WINDOW     = "Mask Preview  [M to close]"

CAM_INDEX       = 0
FRAME_W, FRAME_H = 640, 480

CAM_CHECK_INTERVAL = 90
CAM_MOVED_AREA_LIMIT = 500


def _noop(_: int) -> None:
    """Empty callback required by cv2.createTrackbar."""
    pass


def setup_trackbars() -> None:
    """
    Create the HSV Calibration window with 6 trackbars.

    The window contains a tiny 1-pixel canvas — it exists only to host
    the trackbars, not to display any image content.

    Trackbar layout:
      H1 Lo / H1 Hi — hue bounds for the lower red band (H ≈ 0-10)
      H2 Lo / H2 Hi — hue bounds for the upper red band (H ≈ 170-180)
      S Min          — minimum saturation (rejects skin, beige, pastel)
      V Min          — minimum value      (rejects very dark / shadowed pixels)
    """
    cv2.namedWindow(CALIB_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CALIB_WINDOW, 400, 220)

    cv2.createTrackbar("H1 Lo",  CALIB_WINDOW,   0,  30, _noop)
    cv2.createTrackbar("H1 Hi",  CALIB_WINDOW,  10,  30, _noop)

    cv2.createTrackbar("H2 Lo",  CALIB_WINDOW, 162, 180, _noop)
    cv2.createTrackbar("H2 Hi",  CALIB_WINDOW, 180, 180, _noop)

    cv2.createTrackbar("S Min",  CALIB_WINDOW, 120, 255, _noop)
    cv2.createTrackbar("V Min",  CALIB_WINDOW,  80, 255, _noop)


def read_trackbar_params() -> dict:
    """
    Read current trackbar positions and return them as a parameter dict
    consumed by create_red_mask().

    Returns:
        dict with keys: h1_lo, h1_hi, h2_lo, h2_hi, s_lo, v_lo
    """
    return {
        "h1_lo": cv2.getTrackbarPos("H1 Lo", CALIB_WINDOW),
        "h1_hi": cv2.getTrackbarPos("H1 Hi", CALIB_WINDOW),
        "h2_lo": cv2.getTrackbarPos("H2 Lo", CALIB_WINDOW),
        "h2_hi": cv2.getTrackbarPos("H2 Hi", CALIB_WINDOW),
        "s_lo" : cv2.getTrackbarPos("S Min", CALIB_WINDOW),
        "v_lo" : cv2.getTrackbarPos("V Min", CALIB_WINDOW),
    }


def draw_hud(frame: np.ndarray,
             fps: float,
             mask_preview_on: bool,
             cam_moved_warning: bool,
             mask_px: int = 0) -> np.ndarray:
    """
    Draw a minimal Heads-Up Display on top of the composited output frame.

    Includes:
      - FPS counter (top-left)
      - Mask pixel count — confirms how much of the cloak is being replaced
      - Hotkey reminder strip (bottom)
      - Camera-moved warning (top-right, red text) when detected

    Args:
        frame            : Composited output frame (will NOT be mutated —
                           a copy is made internally).
        fps              : Measured frames per second.
        mask_preview_on  : Whether the mask debug window is currently visible.
        cam_moved_warning: Whether the camera-shift warning should be shown.
        mask_px          : Number of white pixels in the clean mask.

    Returns:
        A copy of the frame with HUD text rendered on it.
    """
    out   = frame.copy()
    h, w  = out.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(out, fps_text, (10, 28), font, 0.75,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, fps_text, (10, 28), font, 0.75,
                (0, 255, 120), 2, cv2.LINE_AA)

    if mask_px > 0:
        px_text = f"Cloak: {mask_px:,} px"
        cv2.putText(out, px_text, (10, 52), font, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, px_text, (10, 52), font, 0.6,
                    (0, 200, 255), 2, cv2.LINE_AA)


    bar_h = 26
    cv2.rectangle(out, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    mask_label = "M: Hide mask" if mask_preview_on else "M: Show mask"
    hint = f"  Q: Quit    R: Recapture bg    {mask_label}"
    cv2.putText(out, hint, (6, h - 8), font, 0.52,
                (180, 180, 180), 1, cv2.LINE_AA)

    return out


def main() -> None:
    """
    Application entry point.

    Lifecycle
    ---------
    1.  Open webcam and warm it up.
    2.  Create output + calibration windows with trackbars.
    3.  Capture averaged background (5-second countdown).
    4.  Enter the real-time processing loop:
          a. Read frame, flip for mirror view.
          b. Light Gaussian blur to denoise before colour detection.
          c. BGR → HSV conversion.
          d. Red detection via dual-range inRange (trackbar-driven params).
          e. Full morphological cleaning pipeline (open, close, fill holes,
             keep largest blob, final smooth).
          f. Soft alpha blend to replace cloak pixels with stored background.
          g. Draw HUD (FPS, hints, warnings).
          h. Display output; optionally show mask debug window.
          i. Handle keystrokes (Q / R / M).
          j. Every CAM_CHECK_INTERVAL frames (when cloak is absent) check
             for camera movement and set a warning flag.
    5.  Release resources.
    """

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Error] Cannot open webcam (index {CAM_INDEX}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    warmup_camera(cap, num_frames=80)

    cv2.namedWindow(OUTPUT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(OUTPUT_WINDOW, FRAME_W, FRAME_H)

    setup_trackbars()

    print("\n" + "="*50)
    print("  Invisible Cloak — Controls")
    print("="*50)
    print("  Q  — Quit")
    print("  R  — Recapture background")
    print("  M  — Toggle mask debug window")
    print("="*50 + "\n")

    background = capture_background(
        cap,
        window_name    = OUTPUT_WINDOW,
        countdown_secs = 5,
        num_frames     = 60,
    )

    mask_preview_on    = False
    cam_moved_warning  = False
    frame_count        = 0

    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps       = 0.0

    while True:

        ret, frame = cap.read()
        if not ret:
            print("[Warning] Frame read failed — retrying...")
            continue

        frame = cv2.flip(frame, 1)

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        params  = read_trackbar_params()
        raw_mask = create_red_mask(hsv, params)

        clean = clean_mask(raw_mask)

        output = replace_background(frame, background, clean)

        cur_tick = cv2.getTickCount()
        elapsed  = (cur_tick - prev_tick) / tick_freq
        fps      = 0.9 * fps + 0.1 * (1.0 / elapsed if elapsed > 0 else fps)
        prev_tick = cur_tick

        frame_count += 1
        mask_area    = int(np.sum(clean > 0))

        if frame_count % CAM_CHECK_INTERVAL == 0 and mask_area < CAM_MOVED_AREA_LIMIT:
            cam_moved_warning = check_camera_moved(frame, background)

        if cam_moved_warning and mask_area < CAM_MOVED_AREA_LIMIT:
            if not check_camera_moved(frame, background):
                cam_moved_warning = False

        output_hud = draw_hud(output, fps, mask_preview_on, cam_moved_warning,
                              mask_px=mask_area)

        cv2.imshow(OUTPUT_WINDOW, output_hud)

        if mask_preview_on:
            raw_bgr   = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
            clean_bgr = cv2.cvtColor(clean,    cv2.COLOR_GRAY2BGR)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(raw_bgr,   (0, 0), (raw_bgr.shape[1],   20), (30, 30, 30), -1)
            cv2.rectangle(clean_bgr, (0, 0), (clean_bgr.shape[1], 20), (30, 30, 30), -1)
            cv2.putText(raw_bgr,   "RAW MASK",
                        (5, 15), font, 0.55, (100, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(clean_bgr, f"CLEAN MASK  ({mask_area:,} px)",
                        (5, 15), font, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

            separator = np.zeros((raw_bgr.shape[0], 3, 3), dtype=np.uint8)
            separator[:] = (80, 80, 80)
            panel = np.hstack([raw_bgr, separator, clean_bgr])
            cv2.imshow(MASK_WINDOW, panel)
        else:
            try:
                cv2.destroyWindow(MASK_WINDOW)
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            print("[App] Quitting...")
            break

        elif key in (ord('r'), ord('R')):
            print("[App] Recapturing background...")
            cam_moved_warning = False
            background = capture_background(
                cap,
                window_name    = OUTPUT_WINDOW,
                countdown_secs = 5,
                num_frames     = 60,
            )

        elif key in (ord('m'), ord('M')):
            mask_preview_on = not mask_preview_on
            state = "ON" if mask_preview_on else "OFF"
            print(f"[App] Mask preview {state}")

    cap.release()
    cv2.destroyAllWindows()
    print("[App] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
