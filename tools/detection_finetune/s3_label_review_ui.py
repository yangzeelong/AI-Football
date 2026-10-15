#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 人工校验界面（**零依赖**：只用 Python 标准库，不装 Label Studio/CVAT）。

为什么自带界面：本机 headless（浏览器靠 ssh 端口转发），且 Label Studio 的依赖
`python-json-logger` 在本机装不上——安装时创建 `pythonjsonlogger` 目录被拒（EACCES，root 也一样；
换 conda / 换目录 / `/tmp` 同样，已实测）。本界面只需浏览器 + 端口转发，且直接读写本项目的
COCO 结构，类别名与 `classes.json` 天生一致。

**一次启动覆盖全部批次**（批次在界面里下拉切换，各批结果写各自的 `<batch>_edited.json`，
差异报告仍按批分，便于 `s3_label_review_roundtrip.py --import --batch <批>` 分批回收）：

* 显示预标注（person 蓝 / ball 绿 / cone 橙）与来源、split、批次进度；
* 点框选中 → `1/2/3` 改类，`Delete` 删框，画布拖拽新增框（默认 cone）；
* `←/→` 翻图，`Enter` 标记"本图已核对"并前进，`S` 保存；
* 编辑进度存 localStorage（刷新不丢），保存写服务端。

用法::

    $PY s3_label_review_ui.py                                  # 载入全部批次，127.0.0.1:8090
    $PY s3_label_review_ui.py --batch train_video_01,valid_01  # 只载指定批次
    $PY s3_label_review_ui.py --batch synth_spot_01            # 只做合成抽检
    # 本地另开终端：ssh -L 8090:localhost:8090 <user>@<host> → http://localhost:8090
    # 复核完（点“保存”）回到命令行：
    $PY s3_label_review_roundtrip.py --import reports/review_batches/train_video_01/train_video_01_edited.json \\
        --batch train_video_01
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import base_layout

#: 类别配色（与界面一致）
CLASS_STYLE = {1: ("person", "#4C8BF5"), 2: ("ball", "#3CC47C"), 3: ("cone", "#F5A623")}
DEFAULT_PORT = 8090
#: 页面版本号（改了前端就 +1；工具条上会显示，用来确认"你看的是不是新代码"）
BUILD_TAG = "overlay-1"
#: 批次展示顺序（train 在前，合成抽检最后）
BATCH_ORDER = ("train_video", "valid", "test", "synth")


def _norm_boxes(boxes: Iterable[dict[str, Any]]) -> list[tuple[int, tuple[float, ...]]]:
    """把框列表归一成可比较的形式（判断"这张图到底改没改"）。"""
    out = []
    for box in boxes:
        out.append((int(box["cls"]), tuple(round(float(v), 1) for v in box["bbox"])))
    return sorted(out)


class Batch:
    """一个批次的标注状态。"""

    def __init__(self, name: str, batch_dir: Path) -> None:
        self.name = name
        self.dir = batch_dir
        self.preannot = batch_dir / "preannot.json"
        self.out_path = batch_dir / ("%s_edited.json" % name)
        #: 原始预标注（人工改造的**基准**：判断"哪些框是人工新增/改过的"必须用它，
        #: 而不是"上次保存的状态"，否则保存一次之后 diff 恒为空）
        self.pre = json.loads(self.preannot.read_text(encoding="utf-8"))
        self.pre_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in self.pre["annotations"]:
            self.pre_by_image.setdefault(ann["image_id"], []).append(ann)
        self.coco = json.loads((self.out_path if self.out_path.exists() else self.preannot)
                               .read_text(encoding="utf-8"))
        self.id_to_name = {c["id"]: c["name"] for c in self.coco["categories"]}
        self.by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in self.coco["annotations"]:
            self.by_image.setdefault(ann["image_id"], []).append(ann)

    def orig_of(self, image_id: int) -> list[dict[str, Any]]:
        """该图的原始预标注框（类别 + xywh）。"""
        return [{"cls": a["category_id"], "bbox": [round(float(v), 1) for v in a["bbox"]]}
                for a in self.pre_by_image.get(image_id, [])]

    def boxes_of(self, image_id: int) -> list[dict[str, Any]]:
        return [{"id": a["id"], "cls": a["category_id"],
                 "bbox": [round(float(v), 1) for v in a["bbox"]]}
                for a in self.by_image.get(image_id, [])]

    def write(self, edited: dict[int, list[dict[str, Any]]]) -> tuple[int, dict[str, int], int]:
        """按前端结果写回本批 `<batch>_edited.json`；返回 (框数, 类别计数, **真正改动**的图数)。"""
        anns: list[dict[str, Any]] = []
        counts: dict[str, int] = {"person": 0, "ball": 0, "cone": 0, "empty": 0}
        changed = 0
        for img in self.coco["images"]:
            original = [{"cls": a["category_id"], "bbox": [round(float(v), 2) for v in a["bbox"]]}
                        for a in self.by_image.get(img["id"], [])]
            boxes = edited.get(img["id"])
            if boxes is None:                      # 前端没动过这张 → 保留原标
                boxes = original
            elif _norm_boxes(boxes) != _norm_boxes(original):
                changed += 1
            if not boxes:
                counts["empty"] += 1
            for box in boxes:
                cls = int(box["cls"])
                x, y, w, h = (float(v) for v in box["bbox"])
                anns.append({"id": len(anns) + 1, "image_id": img["id"], "category_id": cls,
                             "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                             "area": round(w * h, 2), "iscrowd": 0,
                             "attributes": {"source": "human_review", "review": "human"}})
                key = self.id_to_name.get(cls, str(cls))
                counts[key] = counts.get(key, 0) + 1
        self.coco["annotations"] = anns
        self.out_path.write_text(json.dumps(self.coco, ensure_ascii=False), encoding="utf-8")
        return len(anns), counts, changed


class State:
    """一次复核会话：多批次 + 全局图片 id 映射 + 保存路由。"""

    def __init__(self, names: list[str], root: Path) -> None:
        self.root = root
        self.batches: list[Batch] = []
        for name in names:
            batch_dir = base_layout.REVIEW_BATCH_DIR / name
            if not (batch_dir / "preannot.json").exists():
                print("跳过 %s（缺 preannot.json，先 s3_label_review_roundtrip.py --export）" % name)
                continue
            self.batches.append(Batch(name, batch_dir))
        if not self.batches:
            raise SystemExit("没有可用的批次")
        # 全局 id：gid → (批次名, 批次内 image_id)；前端只认 gid
        self.gid_to: list[tuple[str, int]] = []
        self.tasks: list[dict[str, Any]] = []
        for batch in self.batches:
            for img in batch.coco["images"]:
                gid = len(self.gid_to) + 1
                self.gid_to.append((batch.name, img["id"]))
                self.tasks.append({
                    "gid": gid, "batch": batch.name,
                    "file_name": img["file_name"],
                    "width": img.get("width") or 0, "height": img.get("height") or 0,
                    # 路径里有空格/中文，必须 URL 编码
                    "url": "/img?p=%s" % quote(img["file_name"]),
                    "boxes": batch.boxes_of(img["id"]),
                    #: 原始预标注：前端用它算"上一张新增/改过了哪些框"（叠加复用的基准）
                    "orig": batch.orig_of(img["id"]),
                })
        self.spec = (Path(__file__).parent / "docs" / "annotation_spec.md").read_text(
            encoding="utf-8") if (Path(__file__).parent / "docs" / "annotation_spec.md").exists() else ""

    def image_path(self, rel: str) -> Path | None:
        """解析图片路径，且**只允许数据根之内**（防目录穿越）。"""
        cand = (self.root / rel).resolve()
        if str(cand).startswith(str(self.root.resolve())) and cand.is_file():
            return cand
        for split in base_layout.SPLITS:                 # 兜底：dataset/<split>/images/<名>
            alt = (base_layout.split_images(split) / Path(rel).name).resolve()
            if alt.is_file():
                return alt
        return None

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        """按批次路由前端结果，逐批写 `<batch>_edited.json`。"""
        boxes_by_gid = {int(item["gid"]): item.get("boxes", []) for item in payload.get("images", [])}
        per_batch: dict[str, dict[int, list[dict[str, Any]]]] = {b.name: {} for b in self.batches}
        for gid, boxes in boxes_by_gid.items():
            if 1 <= gid <= len(self.gid_to):
                name, image_id = self.gid_to[gid - 1]
                per_batch[name][image_id] = boxes
        result: dict[str, Any] = {"batches": {}}
        for batch in self.batches:
            edits = per_batch.get(batch.name, {})
            n_ann, counts, changed = batch.write(edits)
            result["batches"][batch.name] = {
                "edited_images": changed, "annotations": n_ann, "counts": counts,
                "out": str(batch.out_path.relative_to(base_layout.DATASET_ROOT))}
        return result


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>ball_cone_v1 人工校验</title>
<style>
 :root{--person:#4C8BF5;--ball:#3CC47C;--cone:#F5A623;}
 *{box-sizing:border-box}
 body{margin:0;background:#14161a;color:#e8eaed;font:13px/1.5 -apple-system,"PingFang SC",sans-serif;
      display:flex;height:100vh}
 #left{flex:1;display:flex;flex-direction:column;min-width:0}
 #bar{padding:8px 12px;background:#1d2026;display:flex;gap:10px;align-items:center;flex-wrap:wrap;
      border-bottom:1px solid #2b2f36}
 #bar b{color:#fff}
 .k{background:#2b2f36;border-radius:4px;padding:1px 6px;font-family:ui-monospace,monospace;font-size:12px}
 .pill{padding:2px 8px;border-radius:10px;font-size:12px;background:#242a31}
 .mode{padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600}
 .mode.sel{background:#2b3a4d;color:#9dc1f0}
 .mode.new{background:#7a4a00;color:#ffd79a}
 select{background:#2b2f36;color:#e8eaed;border:1px solid #3c4149;border-radius:6px;padding:4px 8px}
 #wrap{flex:1;overflow:auto;display:flex;align-items:center;justify-content:center;padding:10px}
 canvas{background:#000;box-shadow:0 0 0 1px #2b2f36;max-width:100%}
 canvas.select{cursor:crosshair}
 canvas.new{cursor:copy}
 #right{width:340px;border-left:1px solid #2b2f36;padding:12px;overflow:auto;background:#191c21;font-size:12px}
 #right h3{margin:0 0 6px;font-size:13px;color:#fff}
 #list div{padding:2px 4px;border-radius:4px;cursor:pointer;font-family:ui-monospace,monospace}
 #list div.sel{background:#2b3a4d;outline:1px solid #4C8BF5}
 .muted{color:#9aa0a6}
 button{background:#2b2f36;color:#e8eaed;border:1px solid #3c4149;border-radius:6px;padding:5px 10px;cursor:pointer}
 button:hover{background:#343a43}
 button.pri{background:#2e7d32;border-color:#2e7d32}
 table{width:100%;border-collapse:collapse}
 td{padding:2px 0}
 td.n{text-align:right;font-family:ui-monospace,monospace}
 #toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);background:#2e7d32;color:#fff;
        padding:8px 16px;border-radius:8px;opacity:0;transition:.25s;pointer-events:none}
 #toast.show{opacity:1}
</style></head><body>
<div id="left">
  <div id="bar">
    <select id="bsel"></select>
    <span id="pos" class="muted">-/-</span>
    <span id="mode" class="mode sel">选择模式（拖拽=框选）</span>
    <span class="pill">选中 <b id="nsel">0</b></span>
    <span class="pill">本图框 <b id="nbox">0</b></span>
    <span class="pill">来源 <b id="src">-</b> · split <b id="spl">-</b></span>
    <span class="pill">进度 <b id="prog">-</b></span>
    <span class="k">N 新建框</span><span class="k">拖拽 框选</span><span class="k">1/2/3 改类</span>
    <span class="k">Del 删除选中</span><span class="k">Ctrl+Z 撤销</span><span class="k">Esc 清空选择</span><span class="k">C 叠加新增框</span><span class="k">Shift+C 整体覆盖</span>
    <span class="k">←/→ 翻图</span><span class="k">Enter 已核对并下一张</span><span class="k">S 保存</span>
    新建框类别 <select id="ncls"></select>
    <button id="save" class="pri">保存</button>
    <button id="mark">标记已核对 (Enter)</button>
    <button id="reuse">叠加上一张新增框 (C)</button>
    <label class="muted" title="勾上后按 C 叠加完会自动标记已核对、跳下一张（适合几乎一样的连帧）">
      <input type="checkbox" id="autoNext"> 叠加后自动下一张</label>
    <button id="reload">放弃改动重载</button>
  </div>
  <div id="wrap"><canvas id="cv" class="select"></canvas></div>
</div>
<div id="right">
  <h3>本图框（<span id="nbox2">0</span>，点=选中，Shift+点=加选）</h3>
  <div id="list"></div>
  <h3 style="margin-top:14px">批次进度</h3>
  <table id="batches"></table>
  <h3 style="margin-top:14px">保存结果</h3>
  <div id="stat" class="muted">-</div>
  <h3 style="margin-top:14px">标注规范（摘要）</h3>
  <div id="spec" class="muted" style="white-space:pre-wrap;max-height:28vh;overflow:auto"></div>
</div>
<div id="toast"></div>
<script>
const TASKS = __TASKS__, STYLE = __STYLE__, SPEC = __SPEC__, BATCHES = __BATCHES__;
// 页面脚本一有异常就弹出来（这类内嵌 JS 出错时浏览器默认静默，容易误判成"点了没反应"）
window.onerror = function(msg, src, line, col){
  try{ const el=document.getElementById('toast');
       if(el){ el.textContent='页面脚本错误：'+msg+'（行'+(line||0)+'）';
               el.style.background='#b3261e'; el.classList.add('show'); } }catch(_){}
  return false;
};
const EDIT='review_ui_edit', DONE='review_ui_done', UNDOMAX=40;
let idx=0, mode='select', sel=new Set(), newCls=3, img=null, scale=1, drag=null, undoStack=[];
const cv=document.getElementById('cv'), ctx=cv.getContext('2d'), wrap=document.getElementById('wrap');
const norm = bs => JSON.stringify(bs.map(b=>[b.cls, ...b.bbox.map(v=>Math.round(v))]).sort());
const edits = JSON.parse(localStorage.getItem(EDIT) || '{}');
const done  = JSON.parse(localStorage.getItem(DONE)  || '{}');
TASKS.forEach(t=>{ t.orig = (t.orig || []).slice(); if(edits[t.gid]) t.boxes = edits[t.gid]; });
const cur = () => TASKS[idx];
const touched = t => norm(t.boxes) !== norm(t.orig);
//: 当前图的框：t.boxes 是权威值（edits[gid] 只是 localStorage 镜像）
//: 注意：这个函数之前漏定义了，导致 render/draw/hit/applyClass/delSelected 全部抛异常
const boxesOf = t => (t && Array.isArray(t.boxes)) ? t.boxes : [];
function boxCount(t){ return boxesOf(t).length; }
function iouXYWH(a, b){
  const ax2=a[0]+a[2], ay2=a[1]+a[3], bx2=b[0]+b[2], by2=b[1]+b[3];
  const ix=Math.max(0, Math.min(ax2,bx2)-Math.max(a[0],b[0]));
  const iy=Math.max(0, Math.min(ay2,by2)-Math.max(a[1],b[1]));
  const inter=ix*iy, uni=a[2]*a[3]+b[2]*b[3]-inter;
  return uni>0 ? inter/uni : 0;
}
//: 该图里"人工新增或改过"的框：在原始预标注里找不到同类且 IoU>=0.5 的对应框
function dirtyBoxesOf(t){
  const orig = t.orig || [], cur = boxesOf(t);
  return cur.filter(b => !orig.some(o => o.cls === b.cls && iouXYWH(b.bbox, o.bbox) >= 0.5));
}
function saveLocal(){
  localStorage.setItem(EDIT, JSON.stringify(edits));
  localStorage.setItem(DONE, JSON.stringify(done)); }
function commit(t){ if(touched(t)) edits[t.gid]=t.boxes; else delete edits[t.gid]; saveLocal(); }
function pushUndo(){ undoStack.push(JSON.stringify(TASKS.map(t=>({gid:t.gid, boxes:t.boxes}))));
  if(undoStack.length>UNDOMAX) undoStack.shift(); }
function popUndo(){
  if(!undoStack.length){ toast('没有可撤销的操作'); return; }
  const snap = JSON.parse(undoStack.pop());
  snap.forEach(o=>{ const t=TASKS.find(x=>x.gid===o.gid); if(t){ t.boxes=o.boxes; commit(t);} });
  sel = new Set([...sel].filter(i=>i<cur().boxes.length)); render(); toast('已撤销'); }
function setSel(indexes, add){ if(!add) sel.clear(); indexes.forEach(i=>sel.add(i)); render(); }
function counts(batch){ let tot=0, ok=0;
  TASKS.forEach(t=>{ if(t.batch!==batch) return; tot++; if(done[t.gid]||edits[t.gid]) ok++; });
  return [ok,tot]; }
function firstIdxOf(batch){ let i=0; for(const b of BATCHES){ if(b===batch) return i; i+=counts(b)[1]; } return 0; }
function renderBatches(){ const s=document.getElementById('bsel'); s.innerHTML='';
  BATCHES.forEach(b=>{ const [ok,tot]=counts(b); const o=document.createElement('option');
    o.value=b; o.textContent=b+'  '+ok+'/'+tot; s.appendChild(o); });
  s.value=cur().batch;
  const tb=document.getElementById('batches'); tb.innerHTML='';
  BATCHES.forEach(b=>{ const [ok,tot]=counts(b); const tr=document.createElement('tr');
    tr.innerHTML='<td>'+b+'</td><td class="n">'+ok+' / '+tot+'</td>';
    tr.style.cursor='pointer'; tr.onclick=()=>jump(firstIdxOf(b)); tb.appendChild(tr); }); }
function jump(i){ idx=Math.max(0,Math.min(i,TASKS.length-1)); sel.clear(); drag=null; render(); }
function render(){
  const t=cur(); if(!t) return;
  const [ok,tot]=counts(t.batch);
  document.getElementById('pos').textContent = t.batch+'  '+(idx-firstIdxOf(t.batch)+1)+'/'+tot+
      '   （总 '+(idx+1)+'/'+TASKS.length+'）';
  document.getElementById('src').textContent=t.source||'-';
  document.getElementById('spl').textContent=t.split||'-';
  document.getElementById('prog').textContent=ok+'/'+tot;
  document.getElementById('mode').className='mode '+(mode==='new'?'new':'sel');
  document.getElementById('mode').textContent = mode==='new'
      ? '新建模式（拖拽画新框，默认 '+STYLE[newCls][0]+'）' : '选择模式（拖拽=框选）';
  cv.className = mode==='new' ? 'new' : 'select';
  document.getElementById('nsel').textContent=sel.size;
  document.getElementById('nbox').textContent=t.boxes.length;
  document.getElementById('nbox2').textContent=t.boxes.length;
  const list=document.getElementById('list'); list.innerHTML='';
  t.boxes.forEach((b,i)=>{ const d=document.createElement('div');
    d.className=(sel.has(i)?'sel':''); d.style.color=STYLE[b.cls][1];
    d.textContent=(sel.has(i)?'● ':'○ ')+STYLE[b.cls][0]+'  '+b.bbox.map(v=>Math.round(v)).join(',');
    d.onclick=ev=>{ if(ev.shiftKey){ sel.has(i)?sel.delete(i):sel.add(i); render(); }
                    else setSel([i]); }; list.appendChild(d); });
  document.getElementById('spec').textContent=SPEC.slice(0,1200);
  renderBatches();
  if(!img) img=new Image();
  img.onload=draw; img.src=t.url+'&_='+Date.now();
}
function draw(){
  const sw=wrap.clientWidth-20, sh=wrap.clientHeight-20;
  scale=Math.min(sw/img.naturalWidth, sh/img.naturalHeight, 1);
  cv.width=img.naturalWidth*scale; cv.height=img.naturalHeight*scale;
  cv.style.width=cv.width+'px'; cv.style.height=cv.height+'px';
  ctx.drawImage(img,0,0,cv.width,cv.height);
  cur().boxes.forEach((b,i)=>{ const c=STYLE[b.cls][1]; const on=sel.has(i);
    ctx.lineWidth=on?3.5:1.5; ctx.strokeStyle=c;
    ctx.strokeRect(b.bbox[0]*scale,b.bbox[1]*scale,b.bbox[2]*scale,b.bbox[3]*scale);
    if(on){ ctx.setLineDash([3,3]); ctx.strokeStyle='#fff'; ctx.lineWidth=1;
      ctx.strokeRect(b.bbox[0]*scale-2,b.bbox[1]*scale-2,b.bbox[2]*scale+4,b.bbox[3]*scale+4);
      ctx.setLineDash([]); }
    ctx.fillStyle=c; ctx.font='bold 13px monospace';
    ctx.fillText(STYLE[b.cls][0], b.bbox[0]*scale+3, Math.max(12,b.bbox[1]*scale-3)); });
  if(!drag) return;
  if(mode==='new'){ ctx.setLineDash([4,3]); ctx.strokeStyle='#ffd79a';
    ctx.strokeRect(Math.min(drag.x0,drag.x1)*scale,Math.min(drag.y0,drag.y1)*scale,
                   Math.abs(drag.x1-drag.x0)*scale,Math.abs(drag.y1-drag.y0)*scale); ctx.setLineDash([]); }
  else { ctx.fillStyle='rgba(76,139,245,0.18)'; ctx.strokeStyle='#4C8BF5'; ctx.lineWidth=1;
    const x=Math.min(drag.x0,drag.x1)*scale, y=Math.min(drag.y0,drag.y1)*scale;
    const w=Math.abs(drag.x1-drag.x0)*scale, h=Math.abs(drag.y1-drag.y0)*scale;
    ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h); }
}
function hit(mx,my){ const b=cur().boxes;
  for(let i=b.length-1;i>=0;i--){ const [x,y,w,h]=b[i].bbox;
    if(mx>=x&&mx<=x+w&&my>=y&&my<=y+h) return i; } return -1; }
function rectHit(x1,y1,x2,y2){ const lo_x=Math.min(x1,x2), hi_x=Math.max(x1,x2);
  const lo_y=Math.min(y1,y2), hi_y=Math.max(y1,y2);
  const out=[]; cur().boxes.forEach((b,i)=>{ const [x,y,w,h]=b.bbox;
    if(x < hi_x && x+w > lo_x && y < hi_y && y+h > lo_y) out.push(i); }); return out; }
function xy(ev){ const r=cv.getBoundingClientRect();
  return [(ev.clientX-r.left)/scale,(ev.clientY-r.top)/scale]; }
cv.onmousedown = ev=>{ const [x,y]=xy(ev);
  if(mode==='new'){ drag={x0:x,y0:y,x1:x,y1:y}; return; }
  const h=hit(x,y);
  if(h>=0){ if(ev.shiftKey){ sel.has(h)?sel.delete(h):sel.add(h); render(); } else setSel([h]); return; }
  drag={x0:x,y0:y,x1:x,y1:y,marquee:true}; };
cv.onmousemove = ev=>{ if(!drag) return; const [x,y]=xy(ev); drag.x1=x; drag.y1=y; draw(); };
cv.onmouseup = ev=>{ if(!drag) return; const d=drag; drag=null;
  const box=[Math.min(d.x0,d.x1),Math.min(d.y0,d.y1),Math.abs(d.x1-d.x0),Math.abs(d.y1-d.y0)];
  if(mode==='new'){
    if(box[2]<6||box[3]<6){ draw(); return; }
    pushUndo(); const t=cur(); const nb=t.boxes.slice();
    nb.push({id:0,cls:newCls,bbox:box.map(v=>+v.toFixed(1))});
    t.boxes=nb; commit(t); setSel([nb.length-1]);
    toast('已新建 '+STYLE[newCls][0]+'（1/2/3 改类，N 继续新建）');
    return; }
  if(Math.abs(box[2])<5 && Math.abs(box[3])<5){ sel.clear(); render(); return; }
  const hits=rectHit(d.x0,d.y0,d.x1,d.y1);
  setSel(hits, ev.shiftKey); toast('框选到 '+hits.length+' 个框'); };
function applyClass(cls){ const t=cur();
  if(sel.size===0){ newCls=cls; document.getElementById('ncls').value=cls;
    toast('新建框类别 = '+STYLE[cls][0]); render(); return; }
  pushUndo(); const nb=t.boxes.slice(); [...sel].forEach(i=>{ nb[i]={...nb[i],cls:cls}; });
  t.boxes=nb; commit(t); render(); toast('已把 '+sel.size+' 个框改为 '+STYLE[cls][0]); }
function delSelected(){ const t=cur();
  if(sel.size===0){ toast('先选中框（拖拽框选，或点框）'); return; }
  pushUndo(); const kill=[...sel].sort((a,b)=>b-a); const nb=t.boxes.slice();
  kill.forEach(i=>nb.splice(i,1)); t.boxes=nb; sel.clear(); commit(t); render();
  toast('已删除 '+kill.length+' 个框'); }
function copyPrev(replaceAll){
  try{
    const t = cur();
    if(idx <= 0){ toast('已经是第一张，没有上一张可用（先往后翻一张）'); return; }
    const prev = TASKS[idx-1];
    if(prev.batch !== t.batch){ toast('上一张属于别的批次（'+prev.batch+'），已跳过以免误用'); return; }
    const srcAll = boxesOf(prev);
    if(!srcAll.length){ toast('上一张没有框可复用（先在上一张画几个框）'); return; }

    // Shift+C：两帧几乎完全一致时，用上一张的全部框覆盖本图
    if(replaceAll){
      const before = boxesOf(t);
      const same = norm(before) === norm(srcAll);
      pushUndo();
      t.boxes = srcAll.map(b=>({cls:b.cls, bbox:[...b.bbox], id:0}));
      commit(t); sel = new Set(t.boxes.map((_,i)=>i)); render();
      toast(same ? ('本图与上一张框完全相同（'+srcAll.length+' 个，已重新选中）')
                 : ('已用上一张的全部 '+srcAll.length+' 个框覆盖本图'));
      afterCopy();
      return;
    }

    // C（默认）：只叠加"上一张人工新增/改过的框"，本图其余框保留
    const dirty = dirtyBoxesOf(prev);
    if(!dirty.length){
      toast('上一张没有"新增/改过"的框可叠加（Shift+C 可用它全部框覆盖本图）'); return;
    }
    pushUndo();
    const nb = boxesOf(t).map(b=>({...b}));
    let added = 0, fixed = 0, kept = 0;
    dirty.forEach(src=>{
      const hit = nb.findIndex(b => iouXYWH(b.bbox, src.bbox) >= 0.5);
      if(hit < 0){ nb.push({cls:src.cls, bbox:[...src.bbox], id:0}); added++; }
      else if(nb[hit].cls !== src.cls){ nb[hit] = {...nb[hit], cls: src.cls}; fixed++; }
      else kept++;
    });
    if(!added && !fixed){ toast('上一张的 '+dirty.length+' 个改动在本图已存在，无需叠加'); return; }
    t.boxes = nb; commit(t); sel = new Set(nb.map((_,i)=>i)); render();
    toast('已叠加：新增 '+added+' 个 / 同步类别 '+fixed+' 个'+(kept?('（另有 '+kept+' 个已一致）'):'')
          +'；本图原有框保留');
    afterCopy();
  }catch(err){
    toast('复用失败：'+err.message);
    console.error('copyPrev error', err);
  }
}
function afterCopy(){
  const an = document.getElementById('autoNext');
  if(an && an.checked) markDone();
}
function markDone(){ done[cur().gid]=1; saveLocal(); toast('已标记，跳到下一张未核对');
  for(let i=1;i<=TASKS.length;i++){ const j=(idx+i)%TASKS.length; if(!done[TASKS[j].gid]){ jump(j); return; } } }
document.getElementById('ncls').onchange = ev=>{ newCls=+ev.target.value; render(); };
const on = (id, fn) => { const el = document.getElementById(id); if(el) el.onclick = fn; };
on('mark', markDone);
on('reuse', ev=>copyPrev(ev.shiftKey));          // 点=叠加新增框；Shift+点=整体覆盖
on('save', save);
on('reload', ()=>{ localStorage.removeItem(EDIT); localStorage.removeItem(DONE); location.reload(); });
document.onkeydown = ev=>{
  if(ev.ctrlKey && (ev.key==='z'||ev.key==='Z')){ ev.preventDefault(); return popUndo(); }
  if(ev.key==='ArrowRight') return jump(idx+1);
  if(ev.key==='ArrowLeft')  return jump(idx-1);
  if(ev.key==='Enter')      return markDone();
  if(ev.key==='Escape'){ mode='select'; sel.clear(); return render(); }
  if(ev.key==='n'||ev.key==='N'){ mode = mode==='new' ? 'select' : 'new'; sel.clear(); return render(); }
  if(['1','2','3'].includes(ev.key)) return applyClass(+ev.key);
  if(ev.key==='Delete'||ev.key==='Backspace') return delSelected();
  if((ev.key==='c'||ev.key==='C') && !ev.ctrlKey && !ev.metaKey) return copyPrev(ev.shiftKey);
  if(ev.key==='s'||ev.key==='S') return save(); };
function toast(m){ const t=document.getElementById('toast'); t.textContent=m; t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),1200); }
async function save(){
  const r = await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({images: TASKS.map(t=>({gid:t.gid, boxes:t.boxes}))})});
  const j = await r.json(); let html='';
  for (const [name,info] of Object.entries(j.batches))
    html += '<div>'+name+'：改 '+info.edited_images+' 图 / '+info.annotations+' 框 '+
            JSON.stringify(info.counts)+'</div>';
  document.getElementById('stat').innerHTML = html || '（无变化）';
  toast('已保存到各批次的 _edited.json'); }
(function init(){
  const bt = document.getElementById('build'); if(bt) bt.textContent = '__BUILD_TAG__';
  const an = document.getElementById('autoNext');
  if(an){
    an.checked = localStorage.getItem('review_ui_autonext') === '1';
    an.onchange = ()=>localStorage.setItem('review_ui_autonext', an.checked ? '1' : '0');
  }
  const s=document.getElementById('ncls');
  [3,1,2].forEach(cls=>{ const o=document.createElement('option'); o.value=cls;
    o.textContent=STYLE[cls][0]; o.selected = cls===newCls; s.appendChild(o); });
  try{ render(); }
  catch(err){
    const el=document.getElementById('toast');
    if(el){ el.textContent='界面初始化失败：'+err.message; el.style.background='#b3261e';
            el.classList.add('show'); }
    console.error('init failed', err);
  }
})();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    state: State = None  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:        # 静音
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                                  # noqa: N802
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = (PAGE.replace("__TASKS__", json.dumps(self.state.tasks, ensure_ascii=False))
                        .replace("__STYLE__", json.dumps(CLASS_STYLE, ensure_ascii=False))
                        .replace("__BATCHES__", json.dumps([b.name for b in self.state.batches],
                                                           ensure_ascii=False))
                        .replace("__SPEC__", json.dumps(self.state.spec[:2000], ensure_ascii=False))
                        .replace("__BUILD_TAG__", BUILD_TAG))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/tasks":
            self._send(200, json.dumps(self.state.tasks, ensure_ascii=False).encode("utf-8"),
                       "application/json")
            return
        if parsed.path == "/img":
            rel = parse_qs(parsed.query).get("p", [""])[0]
            path = self.state.image_path(rel)
            if path is None:
                self._send(404, b"image not found", "text/plain")
                return
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), ctype)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:                                 # noqa: N802
        if self.path != "/api/save":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        result = self.state.save(payload)
        for name, info in result["batches"].items():
            if info["edited_images"]:
                print("[已保存] %-16s 改 %3d 图 / %3d 框 %s"
                      % (name, info["edited_images"], info["annotations"], info["counts"]))
        self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json")


def resolve_batches(arg: str) -> list[str]:
    """`all` / 逗号列表 → 批次名列表（按 BATCH_ORDER 排序，train 在前）。"""
    available = sorted(p.name for p in base_layout.REVIEW_BATCH_DIR.iterdir()
                       if (p / "preannot.json").exists()) if base_layout.REVIEW_BATCH_DIR.exists() else []
    if not arg or arg == "all":
        wanted = available
    else:
        wanted = [name.strip() for name in arg.split(",") if name.strip()]
    def key(name: str) -> tuple[int, str]:
        for i, prefix in enumerate(BATCH_ORDER):
            if name.startswith(prefix):
                return (i, name)
        return (len(BATCH_ORDER), name)
    return sorted([w for w in wanted if w in available] or wanted, key=key)


def serve(names: list[str], port: int, host: str) -> int:
    Handler.state = State(names, base_layout.DATASET_ROOT)
    order = [b.name for b in Handler.state.batches]
    httpd = ThreadingHTTPServer((host, port), Handler)
    print("载入 %d 个批次 / %d 图：" % (len(order), len(Handler.state.tasks)))
    for batch in Handler.state.batches:
        print("  %-16s %3d 图 / %4d 框" % (batch.name, len(batch.coco["images"]),
                                          len(batch.coco["annotations"])))
    print("已启动：http://%s:%d/（本机 headless → 本地 ssh -L %d:localhost:%d <user>@<host>）"
          % (host, port, port, port))
    print("界面上可下拉切换批次；保存后各批写各自的 <batch>_edited.json，再按批 --import 回收")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S3 人工校验界面（零依赖，标准库实现）")
    parser.add_argument("--batch", default="all",
                        help="all（默认，全部批次）或逗号分隔批次名")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    base_layout.ensure_layout()
    return serve(resolve_batches(args.batch), args.port, args.host)


if __name__ == "__main__":
    sys.exit(main())
