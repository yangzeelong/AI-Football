# Single-view app MVP

This MVP generates per-frame 2D observations for task 1:

- YOLO detects people and footballs.
- Ultralytics BoT-SORT assigns single-camera person track IDs.
- MMPose RTMPose/WholeBody projects body keypoints into the project 26-point schema.
- A lightweight football tracker keeps one ball track and short missing-frame predictions.
- JSONL output stores camera ID, timestamp, person observations, ball observations, confidence, and state.

## Command

```bash
python code/app.py ^
  --video data/sample.mov ^
  --model yolov8n.pt ^
  --classes person "sports ball" ^
  --output-observations tmp/C1_observations.jsonl ^
  --camera-id C1 ^
  --tracker botsort.yaml ^
  --pose-config path/to/rtmpose_wholebody_config.py ^
  --pose-checkpoint path/to/rtmpose_wholebody_checkpoint.pth ^
  --show
```

If `--pose-config` is omitted, the MVP still writes tracked ball observations and the frame-level schema, but person keypoints are not estimated.

## Project 26 Keypoints

Order:

1. nose
2. left_eye
3. right_eye
4. left_ear
5. right_ear
6. left_shoulder
7. right_shoulder
8. left_elbow
9. right_elbow
10. left_wrist
11. right_wrist
12. left_hip
13. right_hip
14. left_knee
15. right_knee
16. left_ankle
17. right_ankle
18. left_big_toe
19. left_small_toe
20. left_heel
21. right_big_toe
22. right_small_toe
23. right_heel
24. neck
25. pelvis
26. thorax

`neck`, `pelvis`, and `thorax` are virtual keypoints derived from shoulder and hip midpoints.
