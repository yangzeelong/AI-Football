# 骨架平滑说明

这份说明对应当前 `config/app.yaml` 里的 `method: skeleton` 分支。

## 目标

减少关键点抖动，同时避免关键点直接“钉死”在上一帧。当前实现重点照顾最不稳定的点：

- 手腕
- 脚踝
- 脚尖 / 脚跟

躯干点会更保守一些，虚拟点会根据平滑后的骨架重新计算。

## 运行流程

对每个跟踪中的人：

1. 通过 `track_id` 取历史状态。
2. 根据 bbox 中心在滑窗内的运动判断这个人是否静止。
3. 按关键点角色做平滑。
4. 只有静止时才叠加额外约束。
5. 重新生成虚拟点：`neck`、`pelvis`、`thorax`。

## 静止判定

静止不是看单帧位移，而是看最近 `static_window` 帧的 bbox 中心序列。

先算每帧 bbox 中心：

```text
center = ((x1 + x2) / 2, (y1 + y2) / 2)
```

然后取最近 `static_window` 个中心，计算相邻帧位移：

```text
step_i = distance(center[i], center[i-1])
max_step = max(step_i)
```

如果：

```text
max_step <= static_motion_px
```

就认为这个人是静止的。

这意味着：

- 只要最近几帧里 bbox 中心有一帧跳得明显，就不会进静止分支
- 人在走动时，`is_static = false`
- 人停住后，要等满 `static_window` 帧才会开始按静止处理

## 关键点分组

当前 `skeleton` 分支把关键点分成几类：

- 锚点：`left_shoulder`、`right_shoulder`、`left_hip`、`right_hip`
- 快速末端点：`left_wrist`、`right_wrist`、`left_ankle`、`right_ankle`
- 普通末端点：`left_big_toe`、`left_small_toe`、`left_heel`、`right_big_toe`、`right_small_toe`、`right_heel`
- 其他点：剩余可见身体点

## 平滑规则

当前按角色做插值：

- 锚点使用 `skeleton_anchor_alpha`
- 快速末端点使用：
  - 静止时 `skeleton_fast_endpoint_alpha`
  - 运动时 `skeleton_fast_moving_endpoint_alpha`
- 普通末端点使用：
  - 静止时 `skeleton_endpoint_alpha`
  - 运动时 `skeleton_moving_endpoint_alpha`
- 其他点使用：
  - 静止时 `static_alpha`
  - 运动时 `moving_alpha`

这里的“运动”并没有单独做复杂判别，实际就是 `is_static = false` 时走运动分支。

## 静止约束

当人物被判定为静止时，再叠加两层约束：

1. 单点步长限制
   - 快速末端点使用 `skeleton_fast_static_step_px`
   - 其他末端点使用 `skeleton_max_static_step_px`

2. 肢段长度检查
   - 比较当前帧和上一帧的肢段长度：
     - 肩 - 肘 - 腕
     - 髋 - 膝 - 踝
   - 如果相对变化太大，就把末端点往上一帧拉回一点。

## 当前实现的边界

- 静止判定只看 bbox 中心，不看关键点自身速度
- `skeleton` 不是全局骨架求解，只是局部启发式约束
- 现在对腕/踝单独放宽了步长上限，脚尖/脚跟仍然更保守
- `sensitive_keypoints` 主要还是 `one_euro` 分支的配置遗留，对 `skeleton` 分支影响不大

## 当前配置值

```yaml
keypoint_smoothing:
  enabled: true
  method: skeleton
  skeleton_anchor_alpha: 0.35
  skeleton_endpoint_alpha: 0.12
  skeleton_moving_endpoint_alpha: 0.45
  skeleton_fast_endpoint_alpha: 0.22
  skeleton_fast_moving_endpoint_alpha: 0.62
  skeleton_fast_static_step_px: 5
  skeleton_max_static_step_px: 2.5
  skeleton_limb_tolerance: 0.25
```

## 注意点

- 这条分支是偏启发式的本地规则，不是 IK，也不是全局姿态优化。
- `sensitive_keypoints` 目前主要还是给 `one_euro` 分支用，`skeleton` 分支里不直接依赖它。
- 如果手腕或脚踝还是偏慢，优先只调快速末端点的 alpha，不要先动躯干参数。
