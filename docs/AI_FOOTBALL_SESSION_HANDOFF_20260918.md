# AI-Football Alignment Session Handoff

Updated: 2026-09-18
Branch: `sdk/tensorrt`

This file records the current Python/TensorRT alignment work so the next
session can continue without repeating the failed experiments.

## Current Worktree

The worktree contains uncommitted changes in the detector, CUDA preprocess,
pose, tracker, message, graph, demo configuration, and timing-related code.
There is also an existing untracked progress document:

```text
docs/AI_FOOTBALL_PROGRESS_SUMMARY_20260913.md
```

Do not reset or discard these changes. This handoff document is the only file
added by the current session.

## Verified Baseline

Build command:

```bash
cmake --build build --target aifootball_sdk aifootball_demo --parallel 4
```

Result: passed.

The comparison video is:

```text
data/射门1-1080p60.mov
```

Python reference:

```text
/tmp/nexus_compare_python_full/observations.jsonl
```

The latest C++ run was executed for 1000 frames with output at:

```text
/tmp/nexus_compare_cpp_rescuefix_1000/observations.jsonl
```

Comparison after the ball-rescue fix:

```text
common_frames             1000
person_count_mismatch        3
ball_count_mismatch          0
raw_count_mismatch          13
ball_presence_mismatch       0
ball_track_set_mismatch      0
person_bbox_mae              0.0255256131
person_bbox_p95_abs_coord    0.0682830811
person mismatch frames       294, 344, 663
```

The C++ ball presence, ball track IDs, and ball boxes are aligned for this
sample. Person boxes are also stable; only three frames differ in visible
person count.

## Change Already Applied

`FootballTracker::IsRescuableBall()` was adjusted to match the Python rescue
rule. When a valid foot keypoint is close enough to a rejected ball, Python
uses that result and skips the person-bounding-box fallback. C++ now does the
same. This reduced raw-count mismatches from 31 frames to 13 without changing
person boxes or ball tracks.

The earlier large ByteTrack state-machine rewrite was reverted because it
made the Python alignment worse. The current tracker remains the known-good
simplified implementation with the existing ROI, threshold, rejected-ball,
and tracking changes.

## Main Remaining Issue

The remaining raw-count mismatch is primarily a data-semantics problem:

- Python `raw_detection_counts` counts detections after ROI and application
  filters, plus any rescued ball detections.
- C++ currently overwrites `TrackedDetectionMessage::rawCounts` using the
  visible tracker output. A track can be temporarily hidden or emitted at a
  different lifecycle point, so this count is not equivalent to Python's
  filtered detector count.

Relevant current code:

```text
sdk/aifootball/src/tracker/ByteTracker.cpp
  Process(): output rawCounts is rebuilt from tracked persons and accepted balls

sdk/aifootball/src/pose/HRNetPoseEstimator.cpp
  Process(): copies rawCounts from TrackedDetectionMessage

sdk/aifootball/src/tracker/FootballTracker.cpp
  Process(): starts from smoothed rawCounts and adds rescued balls
```

## Recommended Next Change

Keep detector counts and Python-equivalent filtered counts separate instead of
changing the meaning of `rawCounts` mid-pipeline.

Suggested internal field:

```cpp
DetectionCounts filteredCounts;
```

Add it to `TrackedDetectionMessage`, `PoseMessage`, and
`SmoothedPoseMessage`. Then:

1. In `ByteTracker::Process()`, count persons after ROI and person threshold
   checks directly from the filtered detector/tracker input, not from emitted
   tracks. Count accepted balls at the same stage.
2. Copy `filteredCounts` through `HRNetPoseEstimator` and
   `KeypointSmoother`.
3. In `FootballTracker::Process()`, initialize the public output count from
   `smMsg->filteredCounts` and increment it only for rescued balls.
4. Leave the detector's original `rawCounts` available for diagnostics.
5. Ensure `AIFootballPipeline` serializes the final Python-equivalent count in
   the public `SdkOutput`/observation result.

Use `rg -n "rawCounts\\s*=" sdk/aifootball/src` to find every propagation site
before editing. After the change, rerun the 1000-frame comparison and expect
raw-count mismatches to reach zero or to be limited to any remaining rescue
score differences.

## Do Not Do Yet

- Do not replace the simplified ByteTrack implementation with a full tracker
  rewrite before measuring the count-semantics fix.
- Do not change the public SDK API for this issue; the separate count should be
  internal until the comparison is stable.
- Do not treat module timing values as additive FPS. Actors execute in
  parallel, and detector/pose timings are service times.
- Do not overwrite the existing `/tmp` comparison outputs; create a new
  directory such as `/tmp/nexus_compare_cpp_countfix_1000`.

## Verification Commands

```bash
cmake --build build --target aifootball_sdk aifootball_demo --parallel 4
git diff --check

./build/bin/aifootball_demo \
  examples/aifootball_demo/config.yaml \
  --video_path data/射门1-1080p60.mov \
  --output_dir /tmp/nexus_compare_cpp_countfix_1000 \
  --max_frames 1000 --quiet
```

The comparison should use the existing Python reference and compare at least:
person count, ball count, raw count, ball presence, ball track set, and person
box coordinates.
