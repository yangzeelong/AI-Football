#!/bin/bash

input_video="./data/素材/射门1-1080p60.mov"
tmp_file="C1_observations"
output_observations="tmp/$tmp_file.jsonl"
output_report="reports/$tmp_file_quality_report.json"
output_report_md="reports/$tmp_file_quality_report.md"

# 运行视频获取分析结果
python code/app.py --video "$input_video" --detector yolo --model models/yolo/yolov8n.pt \
    --device cuda:0 --pose-device cuda:0 --output-observations "$output_observations" \
    --pose-config models/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py \
    --pose-checkpoint models/mmpose/rtmpose-wholebody/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth \
    --use-roi --roi-config config/roi.json --app-config config/app.yaml

# 根据结果出粗版评估报告
python code/evaluate_observations.py --input "$output_observations" --output "$output_report" --markdown "$output_report_md"

# 根据结果replay视频并生成可视化结果
python code/replay_observations.py --observations "$output_observations" --show --app-config config/app.yaml
