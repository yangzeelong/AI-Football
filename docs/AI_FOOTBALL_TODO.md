# AI-Football TODO

Updated: 2026-09-13

## Immediate

1. Re-run full performance tests with the latest SDK code in Release mode.
   - Current Debug build is much slower and should not be used for performance numbers.
   - Measured sample: Debug 600 frames = 44.224s, about 13.57 FPS.
   - Measured sample: Release 600 frames = 23.323s, about 25.73 FPS.
   - Measure at least: no-render + ROI, render + ROI, and no-ROI if needed.

2. Verify the demo renderer on a longer clip.
   - The short smoke test generated valid JSONL and MP4.
   - Need a longer run to confirm frame pacing and visual stability.

3. Re-check Python vs TensorRT result differences.
   - Align frame count, stride, warm-up, ROI coordinates, and thresholds.
   - Inspect tracker lifecycle and ROI filtering order.
   - Previous C++ ROI result had fewer persons per frame than Python; this still needs investigation.

## SDK

4. Add focused tests for `QueuePolicy`.
   - `Block`: `Process()` waits when the input queue is full.
   - `DropOldest`: drops the oldest queued frame and accepts the new frame.
   - `DropNew`: rejects the current frame and returns `nexusflow::FAILURE`.

5. Add a callback-based integration example.
   - Current demo uses `PollResult()`.
   - Online integration may prefer `SetResultCallback()`.

6. Decide whether more fields belong in `AIFootballContext`.
   - Current context includes config path, device, input queue policy, and ROI.
   - Model paths, batch size, and instance count are still read from YAML.

## Performance

7. Profile Detector in Release mode.
   - Re-check `Detector.Preprocess`, `Detector.TensorRT`, and postprocess timings.
   - Confirm GPU preprocess does not introduce unexpected sync points.

8. Tune multi-instance and batch settings.
   - Detector: compare `instanceCount=1/2` and batch size `1/2/4`.
   - Pose: compare `instanceCount=1/2` and batch size `8/16`.
   - Record FPS, latency, GPU utilization, and memory usage together.

9. Continue reducing data copies.
   - Demo `VideoReader` now passes owned RGB frames through `DecodedFrameView::dataOwner`, so SDK input does not need an extra copy.
   - Future online inputs may need NV12, BGR, plane/stride views, or GPU buffer input to avoid RGB conversion and host copies.

## Useful Commands

```bash
cmake -S . -B build_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_AIFOOTBALL_SDK=ON \
  -DWITH_EXAMPLES=ON \
  -DWITH_TESTING=OFF \
  -DWITH_BENCHMARK=OFF

cmake --build build_release --target aifootball_demo --parallel 4

./build_release/bin/aifootball_demo \
  examples/aifootball_demo/config.yaml \
  --video_path data/射门1-1080p60.mov \
  --output_dir output/sdk_demo_release \
  --quiet
```
