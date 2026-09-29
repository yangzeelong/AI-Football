#!/usr/bin/env bash
#
# Full touch evaluation: run the demo app to produce observations.jsonl, then do
# everything else - reading the annotations, detecting the touches, scoring them
# and rendering the review material - in Python.
#
#   ./examples/touch_eval/run_touch_eval.sh [video] [annotations] [output_dir]
#
# The annotations argument may be a .csv (v2), a .label (legacy) or a .json
# (dumped ground truth). Without it, data/<video>.touches.csv is used when it
# exists, otherwise data/<video>.label.
#
# Environment:
#   MAX_FRAMES=0    0 = whole video (16037 frames for 移动传球.mov)
#   RENDER=1        also render the review mp4 (Python, after the score)
#   CONTACT_RATIO=  override the touch radius ratio for the Python run
#   CONFIG=         pipeline config (default moving_pass_720p.yaml = fp16,
#                   the only path the touch evaluation runs)
#   APP=            app binary (default build/bin/aifootball_app)
#   TOUCH_EVAL_CMD= the evaluator command (default: uv run in the py project)
#
# The app no longer judges touches: a threshold tuned here cannot change the
# observations, only how they are read, so the knobs below are passed to the
# Python evaluation and the app gets a plain inference run.
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

CONFIG="${CONFIG:-$ROOT/examples/touch_eval/config/moving_pass_720p.yaml}"
APP="${APP:-$ROOT/build/bin/aifootball_app}"
TOUCH_EVAL_CMD="${TOUCH_EVAL_CMD:-uv run --project $ROOT/examples/touch_eval/py touch-eval}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RENDER="${RENDER:-1}"
CONTACT_RATIO="${CONTACT_RATIO:-}"

RATIO_ARGS=()
if [[ -n "$CONTACT_RATIO" ]]; then
    RATIO_ARGS=(--contact_distance_ratio "$CONTACT_RATIO")
fi

mkdir -p "$OUT"

echo "== 1/4 running the app on $VIDEO (max_frames=$MAX_FRAMES) =="
APP_ARGS=("$CONFIG" --video_path "$VIDEO" --output_dir "$OUT" --max_frames "$MAX_FRAMES")
if [[ "$RENDER" == "1" ]]; then
    APP_ARGS+=(--render)
fi
"$APP" "${APP_ARGS[@]}" 2>&1 | tee "$OUT/app_run.log"

echo
echo "== 2/4 dumping the ground truth from $ANNOTATIONS =="
$TOUCH_EVAL_CMD --touches_csv "$ANNOTATIONS" --dump_gt "$OUT/ground_truth.json" \
    --quiet 2>/dev/null || $TOUCH_EVAL_CMD --label "$ANNOTATIONS" \
    --dump_gt "$OUT/ground_truth.json" --quiet

echo "== 3/4 scoring (report: $OUT/touch_report.json) =="
# `--label` and `--touches_csv` say the same thing to the parser, which picks by
# extension; the label flag is simply the one that always exists.
$TOUCH_EVAL_CMD --observations "$OUT/observations.jsonl" --label "$ANNOTATIONS" \
    --output_dir "$OUT" "${RATIO_ARGS[@]+"${RATIO_ARGS[@]}"}"

if [[ "$RENDER" == "1" ]]; then
    echo
    echo "== 4/4 rendering the review video (touches burned in, Python) =="
    uv run --project "$ROOT/examples/touch_eval/py" --extra tools \
        python "$ROOT/examples/touch_eval/tools/render_touch_review.py" \
        --output-dir "$OUT" --video "$VIDEO" --out "$OUT/review_pred_vs_gt.mp4"
fi

echo
echo "done: $OUT/touch_report.json, $OUT/touch_events.jsonl"
