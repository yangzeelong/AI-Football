# RF-DETR weights

Put RF-DETR checkpoint files in this directory. The app reads weights from
`models/rfdetr` by default.

| Size | Save as | Download URL |
| --- | --- | --- |
| nano | `rf-detr-nano.pth` | https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth |
| small | `rf-detr-small.pth` | https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth |
| base | `rf-detr-base.pth` | https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth |
| medium | `rf-detr-medium.pth` | https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth |
| large | `rf-detr-large.pth` | https://storage.googleapis.com/rfdetr/rf-detr-large.pth |

Example:

```powershell
python code/app.py --video "data\素材\传球-720p60.mov" --output-dir tmp\rfdetr_smoke --config config\app.yaml
```

RF-DETR size and weight directory are configured in `config/app.yaml`.
