#!/bin/bash
# 微调长跑启动器（**一个启动器跑所有版本**，版本是参数，避免 v1/v2 各一个脚本文件）
#
# 用法：
#   ./s4_train_launch.sh v1                 # 有 checkpoint 就续训，没有就从头
#   ./s4_train_launch.sh v1-frozen-step1    # 第一步：冻结主干只学分类层（--freeze 由配置的 EXTRA_ARGS 带）
#   ./s4_train_launch.sh v1-frozen-step2    # 第二步：解冻 encoder 精修（warm start 自 step1）
#   ./s4_train_launch.sh v1-frozen-step2 --fresh       # 忽略 checkpoint 从头
#   ./s4_train_launch.sh v1-frozen-step2 --epochs 30   # 临时覆盖（其余参数透传给 s4_train_finetune.py）
#
# 约定（与 README「模型管理（唯一命名）」一致）：
#   配置 train_configs/<版本>.yaml ｜ 产物 runs/<版本>/ ｜ 日志 runs/<版本>/train.log
#   pid 文件 /tmp/<版本>.pid  ｜ 停止：kill -- -"$(cat /tmp/<版本>.pid)"
#
# 吞吐实测（s4_train_throughput.py，16G 卡）：batch 4 + checkpointing 最快；加大 batch 反而更慢。
set -u
VERSION="${1:-v1}"
shift || true
cd "$(dirname "$0")" || exit 1
source ./env.sh >/dev/null 2>&1

CONFIG=train_configs/${VERSION}.yaml
OUT=runs/${VERSION}
LOG=$OUT/train.log
PIDF=/tmp/${VERSION}.pid
if [ ! -f "$CONFIG" ]; then
  echo "找不到配置 $CONFIG（版本名要对应 train_configs/<版本>.yaml）"; exit 1
fi
FRESH=0; EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --fresh) FRESH=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

# 版本自带的固定参数：配置里写一行 `# EXTRA_ARGS: --freeze encoder`（step1 就是这么声明冻结的），
# 这样"重跑某一步"只有一条命令，不靠人记参数。
CFG_EXTRA=$(grep -m1 -E "^#\s*EXTRA_ARGS:" "$CONFIG" 2>/dev/null | sed 's/^#\s*EXTRA_ARGS:\s*//' || true)
if [ -n "$CFG_EXTRA" ]; then
  case " ${EXTRA[*]} " in
    *" --freeze "*) : ;;                       # 命令行已显式给过，尊重用户
    *) EXTRA+=($CFG_EXTRA); echo "  配置声明的额外参数：$CFG_EXTRA" ;;
  esac
fi

mkdir -p "$OUT"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "$VERSION 已在跑（pid $(cat "$PIDF")）；先 kill -- -\$(cat $PIDF) 再重启"
  exit 1
fi

# 自动续训：checkpoint.pth 优先，其次编号最大的
RESUME=()
if [ "$FRESH" = "0" ]; then
  if [ -f "$OUT/checkpoint.pth" ]; then
    RESUME=(--resume "$OUT/checkpoint.pth")
  else
    LATEST=$(ls -1 "$OUT"/checkpoint[0-9]*.pth 2>/dev/null | sort | tail -1 || true)
    [ -n "$LATEST" ] && RESUME=(--resume "$LATEST")
  fi
fi

# 分配器：multi_scale 每步输入尺寸都变，开这个降低显存抖动（实测 8657→7873 MiB）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "==== $(date '+%F %T') 启动 ===="
  echo "version=$VERSION  config=$CONFIG  resume=${RESUME[*]:-（从头）}  extra=${EXTRA[*]:-}"
} >> "$LOG"

setsid bash -c '"$0" -u s4_train_finetune.py --config "$2" "${@:3}" >> "$1" 2>&1' \
  "$PY" "$LOG" "$CONFIG" "${RESUME[@]}" "${EXTRA[@]}" > /dev/null 2>&1 &
echo $! > "$PIDF"
sleep 12

if ! pgrep -f "s4_train_finetune.py" > /dev/null; then
  echo "!! 启动失败（12 秒内没看到训练进程），日志尾部："
  tail -12 "$LOG"
  exit 1
fi
echo "已启动 $VERSION pid=$(cat $PIDF)（进程组）  日志：$LOG"
echo "监视：  $PY s4_train_watch.py --log $LOG"
echo "停止：  kill -- -\$(cat $PIDF)"
grep -E "Total batch size|LENGTH OF DATA LOADER" "$LOG" | tail -2
tail -1 "$LOG" | cut -c1-120
