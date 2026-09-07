# Roboflow 与 AI-Football-DEV 实测对比

工作目录：`C:\Proj\AI-Football-DEV`

视频：`C:\Proj\sports-main\examples\soccer\data\2e57b9_0.mp4`

## 结论先行

- `AI-Football-DEV` 已经可以在 `ai-football` conda 环境中稳定跑通完整视频。
- 当前这条视频上，`AI-Football-DEV` 的完整跑分是 `750` 帧用 `42.27s`，约 `17.8 FPS`。
- 同一条视频上，Roboflow 方案完整跑分是 `272.27s`，约 `2.75 FPS`。
- 这次对比是同一输入、同一全量视频，更适合做业务汇报。

## 实测结果

### AI-Football-DEV

- 环境：`ai-football`
- GPU：`cuda:0`
- 输入：`2e57b9_0.mp4`
- 结果：`750/750` 帧全部处理完成
- 耗时：`42.27s`
- 吞吐：约 `17.8 FPS`
- 约合 `00:42.27`
- 结果视频：[rendered.mp4](C:/Proj/AI-Football-DEV/tmp/full_run_2e57b9/rendered.mp4)

### Roboflow 方案

- 同一条视频全量跑分，耗时 `272.27s`
- 吞吐约 `2.75 FPS`
- 约合 `04:32.27`
- 这是和 `AI-Football-DEV` 同条件的完整结果
- 结果视频：[2e57b9_0_player_tracking_full.mp4](C:/Proj/sports-main/outputs/2e57b9_0_player_tracking_full.mp4)

## 业务层对比

### 1. 精度

- Roboflow 方案的优点是检测链路成熟，开箱能力强，适合快速验证。
- 但在足球场景里，远距离小球、遮挡球、贴地球、球员脚边球，仍然会漏。
- `AI-Football-DEV` 这边更适合做业务定制：
  - 可以换 detector
  - 可以调阈值
  - 可以加球恢复逻辑
  - 可以把 `hrnet-w48-dark` 作为 landmark 主干继续增强

### 2. 性能

- `AI-Football-DEV` 当前完整视频实测吞吐已经可用，GPU 下能稳定跑起来。
- Roboflow 方案完整视频实测吞吐明显更低。
- 这类业务里，真正的瓶颈往往不是单个 detector，而是：
  - 检测 + 跟踪 + 姿态 + 渲染的串行开销
  - 大分辨率输入
  - 球这种小目标的反复重检

## 当前看到的问题

- Roboflow 风格方案适合快速拼装，但一旦进入业务长期维护，版本漂移会比较烦。
- 足球检测本身仍有漏检，尤其是小球、快速运动、遮挡场景。

## 建议

1. 先把 `AI-Football-DEV` 作为主线，保留本地推理和可控性。
2. Roboflow 只借鉴“检测 + 跟踪 + 难例回收”的思路，不把平台本身当核心依赖。
3. 后续重点补：
   - 球专用检测稳定性
   - 人体跟踪连续性
   - `hrnet-w48-dark` landmark 的时序稳定性
   - 运行耗时 OSD 和失败样本导出
