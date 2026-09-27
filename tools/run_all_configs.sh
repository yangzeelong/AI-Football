#!/usr/bin/env bash
#
# Run every config under examples/aifootball_app/config one after another,
# render each, and stitch the results into a single side-by-side video so they
# can be compared in one playback. The directory decides what runs: drop a file
# in and it is picked up, remove one and it is skipped.
#
#   ./tools/run_all_configs.sh [frames] [output_root]
#
# Defaults to 300 frames into output/configs. Each config writes its own
# directory plus a run log; the comparison video lands in <output_root>.
#
# OpenCV is needed for the stitching step (cv2 in any environment that has it);
# point PYTHON at that interpreter if the default does not.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

frames="${1:-300}"
out_root="${2:-output/configs}"
app="build/bin/aifootball_app"
python="${PYTHON:-python3}"
config_dir="examples/aifootball_app/config"

if [[ ! -x "$app" ]]; then
    echo "missing $app, build first: cmake --build build" >&2
    exit 1
fi
if ! command -v "$python" >/dev/null 2>&1; then
    echo "$python not found; set PYTHON=/path/to/python with cv2" >&2
    exit 1
fi

# Everything the config directory holds, in a stable order.
configs=()
while IFS= read -r file; do
    configs+=("$file")
done < <(find "$config_dir" -maxdepth 1 -name '*.yaml' | sort)
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "no *.yaml under $config_dir" >&2
    exit 1
fi

mkdir -p "$out_root"
echo "configs: ${#configs[@]}"
for config in "${configs[@]}"; do
    echo "  $(basename "$config")"
done

# Serial on purpose: the runs share one GPU, so running them at the same time
# would make both the timings and the comparison meaningless.
for config in "${configs[@]}"; do
    name="$(basename "$config" .yaml)"
    echo
    echo "=== ${name} (frames=${frames}) ==="
    "$app" "$config" --max_frames "$frames" --output_dir "$out_root/$name" --render \
        2>&1 | tee "$out_root/$name.log" | sed -n '/Throughput/,/observations/p'
done

# Stitch the renders into one video: one column per config, with the config name
# burned into each. Columns are scaled to a common width so the frames line up.
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
for cap, path in zip(caps, paths):
    if not cap.isOpened():
        sys.exit("cannot open %s" % path)

# Keep the stitched frame a reasonable size however many configs there are.
column_width = max(480, 3840 // len(caps))
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
