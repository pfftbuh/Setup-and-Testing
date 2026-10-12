import argparse
import glob
import os

import cv2
import numpy as np

from heatmap_feature_extractor import recover_intensity_map

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES = ("cheating", "non_cheating")


def find_heatmaps(class_dir):
    return sorted(glob.glob(os.path.join(class_dir, "session_*", "session_heatmap_*.png")))


def overlay_class(paths, size=None):
    """Averages the recovered intensity maps of all heatmaps, then re-applies the JET colormap."""
    accumulator = None
    count = 0
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] unreadable: {path}")
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        intensity = recover_intensity_map(img).astype(np.float32)
        accumulator = intensity if accumulator is None else accumulator + intensity
        count += 1
        print(f"  Processed {count}/{len(paths)}: {os.path.basename(path)}")

    if count == 0:
        return None, 0

    mean = accumulator / count
    peak = mean.max()
    normalized = np.uint8(mean / peak * 255) if peak > 0 else np.uint8(mean)
    overlay = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay[normalized == 0] = (0, 0, 0)
    return overlay, count


def main():
    parser = argparse.ArgumentParser(description="Overlay all session heatmaps per class.")
    parser.add_argument("--sessions-dir", default=os.path.join(BASE_DIR, "sessions"))
    parser.add_argument("--output-dir", default=BASE_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for cls in CLASSES:
        paths = find_heatmaps(os.path.join(args.sessions_dir, cls))
        print(f"{cls}: {len(paths)} heatmaps found")
        overlay, used = overlay_class(paths)
        if overlay is None:
            print(f"  No heatmaps to overlay for {cls}.")
            continue
        out_path = os.path.join(args.output_dir, f"overlay_{cls}.png")
        cv2.imwrite(out_path, overlay)
        print(f"  Saved overlay of {used} heatmaps -> {out_path}")


if __name__ == "__main__":
    main()
