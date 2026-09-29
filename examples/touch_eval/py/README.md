# touch_eval (Python)

触球业务逻辑：读标注、判定触球、打分、渲染复盘视频。C++ 侧只负责基模推理
（检测 + 姿态），它写出的 `observations.jsonl` 就是这里唯一的输入。

## 环境

用 [uv](https://docs.astral.sh/uv/)，不需要手工建 venv：

```bash
cd examples/touch_eval/py
uv sync              # 建 .venv 并装 numpy / pillow（+ pytest）
uv run touch-eval --help
uv run pytest -q                       # 全部测试
uv run pytest -q -m parity             # 只跑与 C++ 的差分测试
```

`uv sync` 之后所有命令都用 `uv run ...` 跑，解释器与依赖由 `uv.lock` 固定。
旧的 `../tools/*.py`（画扫描曲线、渲染复盘）用系统 python 即可，它们只依赖
matplotlib / opencv；这两条路的边界会在迁移完成后合并。

## 与 C++ 的差分

`tests/test_parity_cpp.py` 在整段 `data/移动传球.mov`（16037 帧）上把这份 Python
实现和 C++ 实现逐事件、逐字段对拍：

* 判定：`replay(observations.jsonl)` 与 `output/.../touch_events.jsonl` 的
  `frame_index / timestamp / person / foot / distance / threshold / speed` 必须一致；
* 标注解析：与 `touch_eval --dump_gt` 的 JSON 一致；
* 打分：与 `touch_report.json` 的主口径、容差、计数表、误检分类一致。

差分测试就是"可以放心用 Python 跑业务逻辑"的判据：C++ 仍是参考实现，直到这些
测试在数据上稳定通过。
