#!/usr/bin/env bash
# 用法： source env.sh
# 作用： 设好 $PY（ai-football 环境的 python）并把该环境 bin 前置到 PATH，
#        之后文档里的 `$PY xxx.py` 与 `label-studio ...` 都能直接粘。
#
# 为什么需要它：本机默认激活的是 conda base（base 里没有 rfdetr），
# 裸 `python` / `pip` 都会走错环境。

ENV_DIR=${AI_FOOTBALL_ENV:-/root/miniconda3/envs/ai-football}

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "env.sh: 找不到 $ENV_DIR/bin/python —— 请检查 conda 环境名或设置 AI_FOOTBALL_ENV" >&2
  return 1 2>/dev/null || exit 1
fi

export PY="$ENV_DIR/bin/python"
case ":$PATH:" in
  *":$ENV_DIR/bin:"*) ;;
  *) export PATH="$ENV_DIR/bin:$PATH" ;;
esac

# 把工具根目录加进 PYTHONPATH，让各子目录的脚本能 `import base_layout`。
# 用标准机制而不是在每个 .py 里写 `sys.path.insert(...)`——源码里不留任何路径 hack。
# 本文件既可能被 `source env.sh`（CWD 就在本目录），也可能被
# `source ../detection_finetune/env.sh`（CWD 在别处），所以不能依赖 $PWD：
# bash 用 BASH_SOURCE，zsh 用 %x（zsh 独有的展开式，用 eval 包住避免 bash 解析报错）。
if [ -n "${BASH_VERSION:-}" ]; then
  _env_sh="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  eval '_env_sh="${(%):-%x}"'
else
  _env_sh="./env.sh"
fi
_TOOL_DIR=$(cd "$(dirname "$_env_sh")" 2>/dev/null && pwd)
if [ -n "${_TOOL_DIR:-}" ]; then
  case ":$PYTHONPATH:" in
    *":$_TOOL_DIR:"*) ;;
    *) export PYTHONPATH="$_TOOL_DIR${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
fi
unset _env_sh

# 根路径：全部从本文件位置推导，export 给所有子进程；Python 侧 base_layout.py 读同一批变量。
# 布局约定（三个目录相邻即可）：<WORKSPACE>/{AI-Football,Nexusflow,data}
# 要拆开布局就先自己设这些变量，本脚本只补缺省。
if [ -n "${_TOOL_DIR:-}" ]; then
  export AI_FOOTBALL_REPO_ROOT="${AI_FOOTBALL_REPO_ROOT:-$(cd "$_TOOL_DIR/../.." && pwd)}"
  export AI_FOOTBALL_WORKSPACE="${AI_FOOTBALL_WORKSPACE:-$(cd "$AI_FOOTBALL_REPO_ROOT/.." && pwd)}"
  export AI_FOOTBALL_NEXUSFLOW="${AI_FOOTBALL_NEXUSFLOW:-$AI_FOOTBALL_WORKSPACE/Nexusflow}"
  export AI_FOOTBALL_DATA_DIR="${AI_FOOTBALL_DATA_DIR:-$AI_FOOTBALL_WORKSPACE/data}"
  export AI_FOOTBALL_DATASET_DIR="${AI_FOOTBALL_DATASET_DIR:-$AI_FOOTBALL_DATA_DIR/ball_cone_v1}"
fi

echo "env.sh 就绪："
echo "  PY        = $PY"
echo "  TOOL_DIR  = ${_TOOL_DIR:-（没认出来，路径自检会失败）}"
echo "  which python -> $(command -v python)"
echo "  WORKSPACE = ${AI_FOOTBALL_WORKSPACE:-（没认出来）}"
echo "  DATA_DIR  = ${AI_FOOTBALL_DATA_DIR:-（没认出来）}"
"$PY" - <<'PYCHECK'
import sys
try:
    import torch, rfdetr, cv2                      # noqa: F401
    print("  自检 = OK（python %s | torch %s | cuda %s）"
          % (sys.version.split()[0], torch.__version__, torch.cuda.is_available()))
except Exception as exc:                            # noqa: BLE001
    print("  自检 = 失败：%s" % exc)
    print("  提示：装包要用 `$PY -m pip install <pkg>`，否则会装到 conda base")
try:
    import base_layout                                  # noqa: F401
    print("  路径自检 = OK（import base_layout → %s）" % base_layout.__file__)
except Exception as exc:                            # noqa: BLE001
    print("  路径自检 = 失败：%s" % exc)
    print("  提示：子目录脚本靠 PYTHONPATH 找到 base_layout.py，source 本文件才有")
PYCHECK
