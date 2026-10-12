# _legacy —— 归档的重复评测（不参与台账）

这里放的是**同一份权重、同一 split 评了两遍**的旧指标文件（内容与主目录里那份**逐项相同**）。
2026-10-12 统一模型命名时移到这里：保留证据，但避免"一个模型两个文件/两个名字"。

`tools/detection_finetune/metrics_ledger.py` 会跳过本目录（`LEGACY_DIRNAME = "_legacy"`）。

| 文件 | 内容 | 等价于（规范名） |
| --- | --- | --- |
| `metrics_v1-frozen.json` | 旧 tag `finetune-v1-frozen`（已改为 `v1-frozen-step1`，旧名存 `tag_legacy`） | `../test_960/metrics_v1-frozen-step1.json` |

主目录里同一模型的规范文件才是"唯一入口"；本目录只作留档，别引用。
