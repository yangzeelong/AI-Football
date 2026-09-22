#!/usr/bin/env bash
#
# Run the three demo configs one after another, render each, and stitch the
# results into a single side-by-side video so they can be compared in one
# playback.
#
#   ./tools/run_three_configs.sh [frames] [output_root]
#
# Defaults to 300 frames into output/three_configs. Each config writes its own
# directory plus a run log; the comparison video lands in <output_root>.
#
# OpenCV is needed for the stitching step (cv2 in any environment that has it);
# point PYTHON at that interpreter if the default does not.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

frames="${1:-300}"
out_root="${2:-output/three_configs}"
app="build-release/bin/aifootball_app"
python="${PYTHON:-python3}"

configs=(
    examples/aifootball_app/config.yaml
    examples/aifootball_app/config.rfdetr960_hrnetw32_256.yaml
    examples/aifootball_app/config.ref_rfdetr960_hrnetw48d384.yaml
)

if [[ ! -x "$app" ]]; then
    echo "missing $app, build first: cmake --build build-release" >&2
    exit 1
fi
if ! command -v "$python" >/dev/null 2>&1; then
    echo "$python not found; set PYTHON=/path/to/python with cv2" >&2
    exit 1
fi

mkdir -p "$out_root"

# Serial on purpose: the three runs share one GPU, so running them at the same
# time would make both the timings and the comparison meaningless.
for config in "${configs[@]}"; do
    name="$(basename "$config" .yaml)"
    echo
    echo "=== ${name} (frames=${frames}) ==="
    "$app" "$config" --max_frames "$frames" --output_dir "$out_root/$name" --render \
        2>&1 | tee "$out_root/$name.log" | sed -n '/Throughput/,/observations/p'
done

# Stitch the renders into one video: one column per config, with the config name
# burned into each, each column scaled to the same width so the three frames
# line up.
"$python" - "$out_root" "${configs[@]}" <<'PY'
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError as error:
    sys.exit("stitching needs cv2 and numpy: %s" % error)

out_root, *configs = sys.argv[1:]
names = [os.path.splitext(os.path.basename(path))[0] for path in configs]
paths = [os.path.join(out_root, name, "rendered.mp4") for name in names]

caps = [cv2.VideoCapture(path) for path in paths]
for name, cap, path in zip(names, caps, paths):
    if not cap.isOpened():
        sys.exit("cannot open %s" % path)

column_width = 1280
fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
columns = []
for cap in caps:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = column_width / float(width) if width > column_width else 1.0
    columns.append((int(width * scale), int(height * scale), scale))

width = sum(column[0] for column in columns)
height = max(column[1] for column in columns)
target = os.path.join(out_root, "side_by_side.mp4")
writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (width, height))
if not writer.isOpened():
    sys.exit("cannot open %s for writing" % target)

frames = 0
while True:
    tiles = []
    produced = False
    for name, cap, (tile_width, tile_height, scale) in zip(names, caps, columns):
        ok, frame = cap.read()
        if ok:
            produced = True
            if scale != 1.0:
                frame = cv2.resize(frame, (tile_width, tile_height))
        else:
            frame = np.zeros((tile_height, tile_width, 3), np.uint8)
        # Label the column with the config it came from.
        cv2.putText(frame, name, (24, tile_height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(frame)
    if not produced:
        break
    writer.write(np.hstack(tiles))
    frames += 1

writer.release()
for cap in caps:
    cap.release()
print("wrote %s (%d frames, %dx%d)" % (target, frames, width, height))
PY

echo
echo "renders      : $out_root/<config>/rendered.mp4"
echo "run logs     : $out_root/<config>.log"
echo "side by side : $out_root/side_by_side.mp4"
