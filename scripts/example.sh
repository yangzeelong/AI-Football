#!/bin/bash

input_video="./data/素材/射门1-1080p60.mov"
output_dir="tmp/C1_observations"
output_report="reports/C1_observations_quality_report.json"
output_report_md="reports/C1_observations_quality_report.md"

# 运行视频获取分析结果
python code/app.py --video "$input_video" --output-dir "$output_dir" \
    --device cuda:0 --use-roi --roi-config config/roi.json --config config/app.yaml

# 根据结果出粗版评估报告
python code/evaluate_observations.py --input "$output_dir/observations.jsonl" --output "$output_report" --markdown "$output_report_md"

# 根据结果replay视频并生成可视化结果
python code/replay_observations.py --observations "$output_dir/observations.jsonl" --show --app-config config/app.yaml
