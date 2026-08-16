# MMPose models

Task 1 requires 26 project keypoints, so JSONL observation export loads MMPose
by default. Put the WholeBody configs and checkpoints under this directory.

Expected layout:

```text
models/mmpose/
  configs/
    _base_/default_runtime.py
    wholebody_2d_keypoint/rtmpose/coco-wholebody/
      rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
    wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/
      td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py
      td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py
  rtmpose-wholebody/
    rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth
  hrnet/
    hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth
    hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
```

Current default app paths:

```powershell
--pose-config models/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py
--pose-checkpoint models/mmpose/rtmpose-wholebody/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth
```

Download HRNet checkpoints:

```powershell
python scripts/download_hrnet_pose_models.py --model hrnet-w32
python scripts/download_hrnet_pose_models.py --model hrnet-w48-dark
```
