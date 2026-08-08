# 任务 1 Roadmap：单视角感知质量

## 目标

任务 1 的目标是把单视角视频稳定产出可被任务 2 使用的 2D Observation，包括人体框、足球框、26 点骨骼、单镜头 ID、置信度和状态。

整体链路：

```text
视频输入 -> YOLO 人/球检测 -> BoT-SORT 人体跟踪 -> RTMPose/WholeBody 26点骨骼 -> 足球专用跟踪 -> 时序稳定 -> JSONL输出 -> 质量评估
```

## 第 1 阶段：MVP 跑通，1 周

目标：现有代码能稳定跑一段视频，输出可用 JSONL。

已完成：

- YOLO 检测
- BoT-SORT track id
- RTMPose WholeBody 权重
- 26 keypoints schema
- 足球简单跟踪
- JSONL 输出

继续补充：

- 固定运行命令模板
- 跑 3-5 个素材视频
- 保存可视化视频
- 统计每帧人数、足球检测、关键点数量

交付物：

- `tmp/C1_observations.jsonl`
- 可视化视频
- smoke test 结果

## 第 2 阶段：26 点骨骼质量确认，1 周

目标：确认 26 点定义和输出顺序能服务后续动作分析。

重点：

- 核对 26 点顺序
- 检查左右脚脚尖、脚跟是否稳定
- 检查 `neck / pelvis / thorax` 虚拟点是否合理
- 给低置信度点加 `missing / low_confidence` 状态
- 输出关键点可视化编号图，便于业务人员确认

验收：

- 每个人固定输出 26 点
- 点位顺序稳定
- 脚部关键点能区分左右脚
- 低质量点不会被误当成高置信观测

## 第 3 阶段：人体检测与跟踪稳定性，1-2 周

目标：球员在单镜头内 ID 尽量连续。

重点：

- BoT-SORT 参数调优
- 人体框置信度阈值调优
- ROI 过滤联调
- 遮挡后 ID 恢复测试
- 统计 ID switch、track fragment

建议指标：

- 人体检测漏检率
- 单人 ID switch 次数
- track 连续率
- 遮挡恢复帧数
- 每条 track 平均持续时间

交付物：

- 单镜头 tracking 评估脚本
- 每个视频的 tracking 报告

## 第 4 阶段：足球检测与专用跟踪，1-2 周

目标：足球小目标、高速、遮挡情况下尽量不断轨。

重点：

- YOLO 足球检测效果评估
- 足球 tracker 参数调优
- 丢球后短时预测
- 排除误检，比如鞋、白线、反光点
- 判断是否需要标注少量足球数据做微调

建议指标：

- 足球检测召回率
- 足球误检率
- 连续丢失最长帧数
- 丢失后恢复时间
- 预测轨迹跳变距离

交付物：

- 足球轨迹 JSONL
- 足球轨迹可视化
- 是否需要微调的判断报告

## 第 5 阶段：时序稳定与抗抖，1 周

目标：解决静止抖动、关键点跳变、短时漏检。

重点：

- 对关键点加 One Euro Filter 或 Kalman 平滑
- 人体框中心平滑
- 足球轨迹平滑，保留高速变化
- 增加状态：`observed / predicted / low_confidence / occluded / lost`

验收：

- 静止人体关键点抖动下降
- 快速运动不过度滞后
- 短时遮挡有合理预测
- 状态字段能反映观测质量

## 第 6 阶段：三类场景验证，1 周

目标：专门验证任务需求里的三种场景。

场景：

- 静止抖动
- 快速运动
- 遮挡

每类输出：

- 原视频
- 可视化结果
- JSONL
- 指标统计
- 问题帧截图

建议验收表：

| 场景 | 主要指标 | 目标 |
| --- | --- | --- |
| 静止 | 关键点像素标准差 | 越低越好，先建立基线 |
| 快速运动 | 漏检率、轨迹断裂 | 不出现长时间断轨 |
| 遮挡 | ID switch、恢复帧数 | 遮挡后尽量恢复原 ID |

## 第 7 阶段：接口冻结与任务 2 对接，1 周

目标：让任务 2 可以直接消费任务 1 输出。

需要冻结：

- JSONL 字段
- 26 keypoints 顺序
- `track_id` 语义
- `confidence` 语义
- `state` 枚举
- `camera_id`
- `timestamp_sec`
- bbox 坐标格式

建议输出格式：

```json
{
  "frame_index": 0,
  "timestamp_sec": 0.0,
  "camera_id": "C1",
  "persons": [],
  "balls": []
}
```

## 第 8 阶段：整理交付，1 周

目标：形成可复现交付包。

交付物：

- 运行文档
- 模型权重路径说明
- 依赖安装说明
- 2D 输出 schema
- 26 点定义
- 验证报告
- 已知问题列表
- 后续优化建议

## 后续 commit 拆分建议

- `feat: add keypoint confidence states`
- `feat: add observation metrics script`
- `feat: add keypoint smoothing`
- `feat: tune football tracker states`
- `feat: export observation report`
- `docs: add task1 validation roadmap`

## 下一步建议

优先做质量评估脚本。

现在 MVP 已经能跑，接下来最重要的是知道它哪里好、哪里坏。建议先输出这些指标：

- 每帧人数
- 每帧足球数量
- 每人 26 点平均置信度
- 低置信关键点比例
- track_id 数量
- ID 断裂情况
- 足球丢失帧数

有了这个报告，后面调模型、调 tracker、加平滑才不会只靠肉眼判断。
