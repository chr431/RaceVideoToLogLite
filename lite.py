"""RaceVideoToLog Lite — CLI-only, CPU-only video speed extraction.

Usage:
    python lite.py video.mp4 --roi X1 Y1 X2 Y2 [--div N] [--frame-start N] [--frame-end N] [-o output.csv]
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR

from ocr_engine import (SpeedObservation, _get_model_kwargs,
                        extract_speed_value, ocr_digital_fallback,
                        clamp_region,
                        auto_select_anchors, compute_video_hash)
from correction import correct_with_anchors

# ── Fixed defaults (not exposed as CLI args) ──
_MAX_SPEED = 400.0
_MAX_ACCEL = 50.0
_TARGET_H = 24.0  # px
_PAD = 0.0        # px


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RaceVideoToLog Lite — CLI video speed extraction")
    p.add_argument("video", help="Video file path")
    p.add_argument("--roi", nargs=4, type=int, required=True,
                   metavar=("X1", "Y1", "X2", "Y2"), help="Speedometer region (pixels)")
    p.add_argument("--div", type=int, default=2, choices=range(1, 11),
                   help="Frame sampling divisor 1/N (default: 2)")
    p.add_argument("--frame-start", type=int, default=None, metavar="N",
                   help="Start frame number")
    p.add_argument("--frame-end", type=int, default=None, metavar="N",
                   help="End frame number")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output CSV path (default: video_stem.csv)")
    return p.parse_args()


def extract_frames(video_path: Path, region, div: int,
                   frame_start: int | None, frame_end: int | None):
    """Extract cropped frames from video at div interval.

    Uses grab/retrieve pattern: grab() skips frames without decoding (fast),
    retrieve() decodes only the frames we keep.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1, x2, y2 = clamp_region(region, width, height)

    frames = []
    fi = 0
    while True:
        if frame_end is not None and fi >= frame_end:
            break

        # grab() reads raw compressed data – fast, no decode
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_start is not None and fi < frame_start:
            fi += 1; continue
        if fi % div != 0:
            fi += 1; continue

        # retrieve() decodes the last grabbed frame
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            fi += 1; continue

        ts = fi / fps if fps > 0 else 0.0
        crop = frame[y1:y2 + 1, x1:x2 + 1].copy()
        frames.append((ts, crop))
        fi += 1

        if len(frames) % 100 == 0 and total_frames > 0:
            print(f"\r  Extract: {fi}/{total_frames}", end="", flush=True)

    if total_frames > 0 and len(frames) >= 100:
        print(f"\r  Extract: {total_frames}/{total_frames}")
    cap.release()
    return frames, fps


def preprocess(crop, target_h, pad_px):
    """Grayscale + resize to target_h."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    th = max(8.0, float(target_h))
    scale = th / h if h > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)))
    if pad_px > 0:
        gray = cv2.copyMakeBorder(gray, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _finish(gray, target_h, pad_px):
    """Resize + pad + BGR conversion helper."""
    h, w = gray.shape[:2]
    th = max(8.0, float(target_h))
    scale = th / h if h > 0 else 1.0
    if abs(scale - 1.0) > 0.02:
        gray = cv2.resize(gray, (max(1, int(w * scale)), int(th)))
    pad_int = int(pad_px)
    if pad_int > 0:
        gray = cv2.copyMakeBorder(gray, pad_int, pad_int, pad_int, pad_int, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_ocr(frames, ocr, target_h, pad_px, max_speed):
    """OCR pipeline with 3-step fallback chain."""
    observations = []
    for idx, (ts, crop) in enumerate(frames):
        proc = preprocess(crop, target_h, pad_px)
        ocr_result, _ = ocr(proc)
        sv, rt = extract_speed_value(ocr_result)

        if sv is None:
            proc2 = _finish(
                cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                target_h, pad_px)
            ocr_result, _ = ocr(proc2)
            sv, rt = extract_speed_value(ocr_result)

        if sv is None:
            sv, rt = ocr_digital_fallback(ocr, crop, max_speed)

        if sv is not None and rt is not None:
            observations.append(SpeedObservation(
                ts, sv, rt))
        else:
            observations.append(SpeedObservation(ts, -1.0, ""))

        if (idx + 1) % 5 == 0:
            print(f"\r  OCR: {idx + 1}/{len(frames)}", end="", flush=True)

    recognized = len([o for o in observations if o.raw_speed_kmh >= 0])
    print(f"\r  OCR: {len(frames)} frames, {recognized} recognized")
    return observations


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else video_path.with_suffix(".csv")
    region = tuple(args.roi)

    print(f"Video: {video_path}")
    print(f"ROI: {region}  div=1/{args.div}")
    print(f"Limits: {_MAX_SPEED} km/h, {_MAX_ACCEL} m/s^2")

    # ── Extract frames ──
    t0 = time.perf_counter()
    frames, fps = extract_frames(video_path, region, args.div,
                                  args.frame_start, args.frame_end)
    print(f"Frames: {len(frames)} ({fps:.1f} fps, {time.perf_counter()-t0:.1f}s)")

    if not frames:
        print("Error: no frames extracted")
        sys.exit(1)

    # ── OCR ──
    # CPU-only backend
    print("Initializing OCR engine...")
    kwargs = _get_model_kwargs("v5_mobile")
    ocr = RapidOCR(**(kwargs or {}))
    t_ocr_start = time.perf_counter()
    observations = run_ocr(frames, ocr, _TARGET_H, _PAD, _MAX_SPEED)
    t_ocr = time.perf_counter() - t_ocr_start

    if not observations:
        print("Error: no speed data recognized")
        sys.exit(1)

    # ── Auto-anchor + Correction ──
    print(f"Selecting anchors...")
    anchors = auto_select_anchors(observations, _MAX_SPEED)
    print(f"  Anchors: {len(anchors)} ({100*len(anchors)/len(observations):.1f}%)")

    rows = []
    for i, obs in enumerate(observations):
        if i in anchors:
            rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 2])
        else:
            rows.append([obs.timestamp, 0.0, obs.raw_speed_kmh, 0])

    # Correction with scrolling progress
    total_frames = len(rows)
    print("Running correction...")
    t_corr_start = time.perf_counter()

    def _corr_progress(done, total):
        print(f"\r  Correction: {done}/{total}", end="", flush=True)

    rows = correct_with_anchors(
        rows, observations, frames, ocr,
        _MAX_SPEED, _MAX_ACCEL, anchors,
        progress_fn=_corr_progress,
    )
    print(f"\r  Correction: {total_frames} frames processed")
    t_corr = time.perf_counter() - t_corr_start

    # ── Integrate distance ──
    dist = 0.0; prev_t = prev_v = None
    for r in rows:
        v = r[2] / 3.6
        if prev_t is not None and prev_v is not None:
            dt = r[0] - prev_t
            if dt > 0: dist += (prev_v + v) * 0.5 * dt
        prev_t, prev_v = r[0], v
        r[1] = dist

    # ── Write CSV ──
    vhash = compute_video_hash(video_path)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        fh.write("# RaceVideoToLog Lite\n")
        fh.write(f"# video_hash={vhash}, video={video_path.name}\n")
        fh.write(f"# roi={region[0]},{region[1]},{region[2]},{region[3]}, format=km/h\n")
        fh.write(f"# max_speed={_MAX_SPEED}, max_accel={_MAX_ACCEL}, div={args.div}\n")
        w = csv.writer(fh)
        for r in rows:
            w.writerow([f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.2f}", str(r[3])])

    elapsed = time.perf_counter() - t0
    corrected = sum(1 for r in rows if r[3] == 1)
    print(f"Done: {output_path}")
    print(f"  Rows: {len(rows)}  Corrected: {corrected}")
    print(f"  OCR: {t_ocr:.1f}s  Correction: {t_corr:.1f}s  Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
