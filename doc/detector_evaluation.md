# Detector Evaluation

Use `scripts/evaluate_detectors.py` to compare detector choices with the same
task-1 pipeline: detection, required MMPose 26-keypoint estimation, football
tracking, JSONL output, and quality report generation.

## Quick Smoke

```powershell
python scripts/evaluate_detectors.py `
  --videos "data\素材\传球-720p60.mov" `
  --detectors yolo `
  --max-frames 1 `
  --outdir tmp\detector_eval_smoke
```

## Recommended First Pass

```powershell
python scripts/evaluate_detectors.py `
  --videos "data\素材\传球-720p60.mov" "data\素材\带球-720p60.mov" `
  --detectors yolo rfdetr:nano rfdetr:small rfdetr:base `
  --max-frames 500 `
  --use-roi `
  --outdir tmp\detector_eval
```

Outputs:

- `tmp/detector_eval/<video>/<detector>/observations.jsonl`
- `tmp/detector_eval/<video>/<detector>/quality_report.json`
- `tmp/detector_eval/<video>/<detector>/quality_report.md`
- `tmp/detector_eval/summary.csv`
- `tmp/detector_eval/summary.md`

Read `summary.md` first, then open the per-run Markdown reports for failure
frames and detailed ball/keypoint statistics.

## Notes

- JSONL export requires MMPose by default because task 1 requires 26 keypoints.
- YOLO keeps Ultralytics BoT-SORT person `track_id`.
- RF-DETR person detections are tracked with ByteTrack before MMPose.
- `ball_jump_count` and `ball_jump_px_max` are important because a detector can
  report a ball every frame while still switching between wrong candidates.
- RF-DETR first run may be slow because the model is initialized from local
  weights and CUDA warms up.
