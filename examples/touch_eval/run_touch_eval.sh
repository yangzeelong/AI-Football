#!/usr/bin/env bash
#
# Full touch evaluation: run the demo app on the video, dump the ground truth
# of the annotations (v2 one-touch-per-row CSV or the legacy .label) and score
# the detected touches offline.
#
#   ./examples/touch_eval/run_touch_eval.sh [video] [annotations] [output_dir]
#
# The annotations argument may be a .csv (v2), a .label (legacy) or a .json
# (dumped ground truth). Without it, data/<video>.touches.csv is used when it
# exists, otherwise data/<video>.label.
#
# Environment:
#   MAX_FRAMES=0    0 = whole video (16037 frames for 移动传球.mov)
#   RENDER=1        also render the annotated video (adds ~2 ms/frame at 720p)
#   CONTACT_RATIO=  override the touch radius ratio for both runs
#   APP=            app binary (default build/bin/aifootball_app)
#   TOUCH_EVAL=     offline evaluator binary (default build/bin/touch_eval)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIDEO="${1:-$ROOT/data/移动传球.mov}"
OUT="${3:-$ROOT/output/touch_eval_moving_pass}"

VIDEO_STEM="$(basename "$VIDEO")"
VIDEO_STEM="${VIDEO_STEM%.*}"
if [[ -n "${2:-}" ]]; then
    ANNOTATIONS="$2"
elif [[ -f "$ROOT/data/$VIDEO_STEM.touches.csv" ]]; then
    ANNOTATIONS="$ROOT/data/$VIDEO_STEM.touches.csv"
elif [[ -f "$ROOT/data/$VIDEO_STEM.label" ]]; then
    ANNOTATIONS="$ROOT/data/$VIDEO_STEM.label"
else
    echo "no annotations found for '$VIDEO_STEM' in $ROOT/data" >&2
    exit 1
fi

CONFIG="$ROOT/examples/touch_eval/config/moving_pass_720p.yaml"
APP="${APP:-$ROOT/build/bin/aifootball_app}"
TOUCH_EVAL="${TOUCH_EVAL:-$ROOT/build/bin/touch_eval}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RENDER="${RENDER:-1}"
CONTACT_RATIO="${CONTACT_RATIO:-}"

RATIO_ARGS=()
if [[ -n "$CONTACT_RATIO" ]]; then
    RATIO_ARGS=(--contact_ratio "$CONTACT_RATIO")
fi

mkdir -p "$OUT"

echo "== 1/3 running the app on $VIDEO (max_frames=$MAX_FRAMES, render=$RENDER) =="
APP_ARGS=("$CONFIG" --video_path "$VIDEO" --output_dir "$OUT" \
          --max_frames "$MAX_FRAMES" --detect_touches "${RATIO_ARGS[@]+"${RATIO_ARGS[@]}"}")
if [[ "$RENDER" == "1" ]]; then
    APP_ARGS+=(--render)
fi
"$APP" "${APP_ARGS[@]}" 2>&1 | tee "$OUT/app_run.log"

echo
echo "== 2/3 dumping the ground truth from $ANNOTATIONS =="
"$TOUCH_EVAL" --label "$ANNOTATIONS" --dump_gt "$OUT/ground_truth.json" \
    --dump_gt_csv "$OUT/ground_truth.csv" --quiet

echo "== 3/3 scoring (report: $OUT/touch_report.txt) =="
"$TOUCH_EVAL" --observations "$OUT/observations.jsonl" --label "$ANNOTATIONS" \
    --output_dir "$OUT" "${RATIO_ARGS[@]+"${RATIO_ARGS[@]}"}" \
    | tee "$OUT/touch_report.txt"
