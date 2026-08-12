# MMPose models

Task 1 requires 26 project keypoints, so JSONL observation export loads MMPose
by default. Put the RTMPose WholeBody config and checkpoint under this directory.

Expected layout:

```text
models/mmpose/
  configs/
    _base_/default_runtime.py
    wholebody_2d_keypoint/rtmpose/coco-wholebody/
      rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
  rtmpose-wholebody/
    rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth
```

Current default app paths:

```powershell
--pose-config models/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
--pose-checkpoint models/mmpose/rtmpose-wholebody/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth
```
