#!/bin/bash
# 复核界面启动/重启：用 pidfile 停旧实例。
# 注意：**不要**用 `pkill -f s3_label_review_ui.py` 手工重启——那会把"命令行里含这个词"的调用方 shell
# 一起杀掉（踩过两次）。这个脚本自身 cmdline 是脚本路径，因此安全。
set -u
cd "$(dirname "$0")" || exit 1
# 解释器一律取自 env.sh（$PY），不在本脚本里硬编码路径
source ./env.sh >/dev/null || { echo "env.sh 失败：设不出 \$PY"; exit 1; }
[ -x "${PY:-}" ] || { echo "env.sh 没设好 \$PY"; exit 1; }
PIDF=/tmp/review_ui.pid
PORT="${PORT:-8090}"

if [ -f "$PIDF" ]; then
  if kill "$(cat "$PIDF")" 2>/dev/null; then echo "已停止旧实例 pid $(cat "$PIDF")"; fi
  rm -f "$PIDF"
fi
# 兜底：清掉本轮遗留的 python 界面进程。
# 判定必须看**可执行文件**（/proc/<pid>/exe 指向 python）而不是看命令行文本——
# 只看文本会把"命令行里恰好含 s3_label_review_ui.py 的调用方 shell"一起杀掉（踩过两次）
"$PY" - <<'PYEOF'
import os, signal
me = os.getpid()
for pid in filter(str.isdigit, os.listdir('/proc')):
    if int(pid) == me:
        continue
    try:
        exe = os.path.basename(os.readlink('/proc/%s/exe' % pid))
        cmd = open('/proc/%s/cmdline' % pid, 'rb').read().decode('utf-8', 'ignore')
    except OSError:
        continue
    if not exe.startswith('python'):
        continue
    if 'review_ui' in cmd:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print('清掉残留界面进程 pid', pid)
        except OSError:
            pass
PYEOF
sleep 1

setsid "$PY" -u s3_label_review_ui.py --port "$PORT" "$@" > /tmp/review_ui.log 2>&1 < /dev/null &
echo $! > "$PIDF"
sleep 5
echo "=== 启动日志 ==="
cat /tmp/review_ui.log
echo "=== 自检 ==="
curl -s -o /dev/null -w "http://127.0.0.1:$PORT/ -> %{http_code}\n" "http://127.0.0.1:$PORT/"

# 前端冒烟：用 node + 最小 DOM 桩把页面 JS 跑一遍，抓"函数用了没定义/绑定不存在元素"这类静默错误
# node 一律从 PATH 取（需要 ≥18：脚本用了 fetch + 顶层 await）；要指定别的解释器就设 $NODE。
NODE_BIN="${NODE:-$(command -v node 2>/dev/null)}"
if [ -n "$NODE_BIN" ] && [ -f tests/ui_smoke.js ]; then
  "$NODE_BIN" tests/ui_smoke.js "http://127.0.0.1:$PORT/" || {
    echo "!! 前端冒烟失败：页面可能有脚本错误，浏览器里会白屏或按钮无反应（看上面 ✗ 行）"; }
else
  echo "（跳过前端冒烟：没找到 node，装一个即可，如 apt install nodejs / nvm）"
fi
echo "本地终端执行：ssh -L $PORT:localhost:$PORT <user>@<host>  → 浏览器打开 http://localhost:$PORT"
echo "停止：kill \$(cat $PIDF)"
