#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""网络/公开数据入库：`sources/web/` + manifest 登记 + 许可台账。

按命名规则，本脚本产出的素材前缀是 `web_*`（`source=web`），命名规则见 README 与
`base_layout.py` 顶部表。做四件事：

1. **入库命名**：`web_<sha1 前 8>_<原名 slug>.<ext>`，写入 `sources/web/`（原图不缩放）；
2. **质量闸门**：短边 ≥ `--min-side`、长宽比 0.35–2.8、清晰度（拉普拉斯方差）≥ `--min-sharpness`；
3. **近重复去重**：16×16 dHash（与库内已有 + 本批互相），汉明距离 < `--dedupe` 判重（默认 6/256）；
4. **合规台账**：每批追加 `ledger.csv`（来源 URL / 获取时间 / 许可 / 可否再分发 / 条目数），
   manifest 每张图记 `license` 与 `note=采集自 <URL>`；`--license` 必填（**只入库许可明确的**）。

入库后的图登记为 `source=web`、`split=train`（可用 `--split` 改），因此可以**直接走 S3 流程标注**：

    $PY s3_label_review_roundtrip.py --export --source web --limit 100 --batch-name web_01

（web 图没有预标注框 → 批次里是空框，正好用界面里的 `N` 键新建。）

用法::

    $PY s1_data_collect_web.py --ping https://example.com/a.jpg            # 先测可达性
    $PY s1_data_collect_web.py --from-dir ~/Downloads/cones \\
        --license "CC-BY-4.0" --source-url https://example.com/set --redistributable no
    $PY s1_data_collect_web.py --from-list urls.txt --license "CC0" --source-url https://... --fetch
    $PY s1_data_collect_web.py --stats
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import base_layout
from s1_data_extract_frames import dhash, hamming

#: 入库目录
WEB_DIR = base_layout.SOURCES_DIR / "web"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
UA = "Mozilla/5.0 (X11; Linux x86_64) ai-football-dataset/1.0"


def slug(text: str, limit: int = 24) -> str:
    """文件名安全化（保留中文与字母数字，其余换下划线）。"""
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text, flags=re.UNICODE).strip("_")
    return text[:limit] or "img"


def web_name(path_or_url: str, data: bytes) -> str:
    """入库文件名：`web_<sha1前8>_<原文件名 slug>.<ext>`。"""
    digest = hashlib.sha1(data).hexdigest()[:8]
    stem = Path(path_or_url.split("?")[0]).stem
    ext = Path(path_or_url.split("?")[0]).suffix.lower()
    if ext not in IMAGE_EXT:
        ext = ".jpg"
    return "web_%s_%s%s" % (digest, slug(stem), ext)


def load_hashes() -> list[int]:
    """已有入库图的 dHash（用于跨批去重）。"""
    hashes = []
    if not WEB_DIR.exists():
        return hashes
    for path in sorted(WEB_DIR.glob("web_*")):
        if path.suffix.lower() not in IMAGE_EXT:
            continue
        image = cv2.imread(str(path))
        if image is not None:
            hashes.append(dhash(image))
    return hashes


def quality_ok(image: Any, args: argparse.Namespace) -> tuple[bool, str, float]:
    h, w = image.shape[:2]
    if min(h, w) < args.min_side:
        return False, "过小 %dx%d" % (w, h), 0.0
    if not (0.35 <= h / w <= 2.8):
        return False, "长宽比异常 %.2f" % (h / w), 0.0
    sharp = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharp < args.min_sharpness:
        return False, "糊图 tex%.0f" % sharp, sharp
    return True, "", sharp


def fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def ingest_one(name: str, data: bytes, origin: str, args: argparse.Namespace,
               seen: list[int]) -> tuple[bool, str, dict[str, Any] | None]:
    """单张入库：质量闸门 + 去重 + 写盘；返回 (是否入库, 原因, manifest 行)。"""
    import numpy as np

    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        return False, "解码失败", None
    ok, why, sharp = quality_ok(image, args)
    if not ok:
        return False, why, None
    digest = dhash(image)
    if args.dedupe and any(hamming(digest, prev) < args.dedupe for prev in seen):
        return False, "近重复（与库内或本批）", None
    rel = web_name(name, data)
    target = WEB_DIR / rel
    suffix = 1
    while target.exists():                       # 同 sha1 同名（理论上同图）→ 直接算重复
        if target.read_bytes() == data:
            return False, "与库内完全同文件", None
        target = WEB_DIR / ("%s_%d%s" % (Path(rel).stem, suffix, Path(rel).suffix))
        suffix += 1
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        return False, "写盘失败", None
    seen.append(digest)
    row = {
        "image": str(target.relative_to(base_layout.DATASET_ROOT)), "source": "web",
        "source_video": origin,                 # 采集来源 URL（可追溯）
        "frame_index": "", "time_sec": "",
        "width": image.shape[1], "height": image.shape[0],
        "scene": "web", "view": "unknown", "split": args.split,
        "is_synthetic": "no", "ai_background": "no", "is_pseudolabel": "no",
        "license": args.license, "note": "采集自 %s sharpness=%.0f" % (origin, sharp),
    }
    return True, "", row


def write_ledger(args: argparse.Namespace, kept: int, dropped: int, reasons: dict[str, int],
                 urls: Sequence[str]) -> None:
    """追加许可台账（列名必须与 `base_layout.LEDGER_FIELDS` 对齐，否则整行错位）。"""
    exists = base_layout.LEDGER_CSV.exists()
    label = "web_%s" % (args.source_url.split("/")[2] if "//" in args.source_url
                        else Path(args.from_dir or args.from_list or "batch").name)
    with base_layout.LEDGER_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_layout.LEDGER_FIELDS))
        if not exists:
            writer.writeheader()
        writer.writerow({
            "name": label, "url": args.source_url, "fetched_at":
                datetime.now().isoformat(timespec="seconds"), "license": args.license,
            "redistributable": args.redistributable, "n_images": kept,
            "note": "剔除 %d（%s）；来源 URL 数 %d" % (
                dropped, json.dumps(reasons, ensure_ascii=False), len(urls)),
        })


def cmd_ping(args: argparse.Namespace) -> int:
    """先测外网可达性（拿到 URL 后第一步就做这个）。"""
    urls = args.ping if isinstance(args.ping, list) else [args.ping]
    bad = 0
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
            with urllib.request.urlopen(req, timeout=15) as resp:
                print("  OK   %-70s %s" % (url[:70], resp.status))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            bad += 1
            print("  FAIL %-70s %s" % (url[:70], exc))
    return 1 if bad else 0


def cmd_collect(args: argparse.Namespace) -> int:
    if not args.license:
        print("!! 必须显式给 --license（只入库许可明确的物料）；不确定就先别入库")
        return 2
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    seen = [] if args.no_dedupe_against_library else load_hashes()
    print("库内已有 %d 张，载入 dHash %d 个" % (len(list(WEB_DIR.glob('web_*'))), len(seen)))

    items: list[tuple[str, bytes]] = []
    urls: list[str] = []
    if args.from_list:
        urls = [line.strip() for line in Path(args.from_list).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
        print("URL 列表 %d 条（%s）" % (len(urls), "下载" if args.fetch else "仅登记，不下载"))
    elif args.from_dir:
        for path in sorted(Path(args.from_dir).rglob("*")):
            if path.suffix.lower() in IMAGE_EXT and path.is_file():
                items.append((path.name, path.read_bytes()))
        print("目录 %s：找到 %d 张图" % (args.from_dir, len(items)))
    else:
        print("用 --from-dir（本地已下载）或 --from-list（URL 列表，配 --fetch）")
        return 2

    origin = args.source_url or (args.from_dir or args.from_list or "")
    rows, reasons = [], {}
    kept = 0
    if args.from_list and args.fetch:
        for url in urls:
            try:
                data = fetch(url)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                reasons["下载失败"] = reasons.get("下载失败", 0) + 1
                print("  下载失败 %s：%s" % (url[:70], exc))
                continue
            ok, why, row = ingest_one(url, data, url, args, seen)
            if ok:
                rows.append(row); kept += 1
                print("  + %s ← %s" % (row["image"], url[:60]))
            else:
                reasons[why] = reasons.get(why, 0) + 1
                print("  - %s (%s)" % (url[:60], why))
    else:
        for name, data in items:
            ok, why, row = ingest_one(name, data, origin, args, seen)
            if ok:
                rows.append(row); kept += 1
                print("  + %s" % row["image"])
            else:
                reasons[why] = reasons.get(why, 0) + 1
                print("  - %s (%s)" % (name, why))

    if rows:
        with base_layout.MANIFEST_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_layout.MANIFEST_FIELDS))
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in base_layout.MANIFEST_FIELDS})
    dropped = sum(reasons.values())
    write_ledger(args, kept, dropped, reasons, urls)
    print("\n入库 %d 张 → %s（剔除 %d：%s）" % (kept, WEB_DIR.relative_to(base_layout.DATASET_ROOT),
                                              dropped, json.dumps(reasons, ensure_ascii=False)))
    print("台账 → %s" % base_layout.LEDGER_CSV.relative_to(base_layout.DATASET_ROOT))
    if kept:
        print("下一步：$PY s3_label_review_roundtrip.py --export --source web --limit 100"
              " --batch-name web_01（web 图无预标注，用界面 N 键新建框）")
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    files = sorted(p for p in WEB_DIR.glob("web_*") if p.suffix.lower() in IMAGE_EXT) \
        if WEB_DIR.exists() else []
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print("sources/web：%d 张 / %.1f MB" % (len(files), total_mb))
    with base_layout.MANIFEST_CSV.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("source") == "web"]
    print("manifest 里 source=web 的行：%d" % len(rows))
    by_license: dict[str, int] = {}
    for row in rows:
        by_license[row.get("license", "")] = by_license.get(row.get("license", ""), 0) + 1
    for lic, count in sorted(by_license.items()):
        print("  %-24s %d" % (lic or "(未填)", count))
    if base_layout.LEDGER_CSV.exists():
        with base_layout.LEDGER_CSV.open(encoding="utf-8") as handle:
            ledger = [r for r in csv.DictReader(handle)
                      if str(r.get("name", "")).startswith("web_")]
        total = sum(int(r.get("n_images") or 0) for r in ledger)
        print("ledger 里 web 批次：%d（累计登记 %d 张）" % (len(ledger), total))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="网络/公开数据入库（web_* + 台账）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ping", nargs="+", metavar="URL", help="测 URL 可达性")
    group.add_argument("--from-dir", help="已下载图片所在目录")
    group.add_argument("--from-list", help="URL 列表文件（配 --fetch 才下载）")
    group.add_argument("--stats", action="store_true", help="看 web 入库概况")
    parser.add_argument("--fetch", action="store_true", help="真正下载 --from-list 里的 URL")
    parser.add_argument("--license", default="", help="许可（**必填**：CC0/CC-BY-4.0/自有授权…）")
    parser.add_argument("--source-url", default="", help="来源页 URL（可追溯）")
    parser.add_argument("--redistributable", default="no", choices=["yes", "no"], help="可否再分发")
    parser.add_argument("--split", default="train", choices=list(base_layout.SPLITS), help="归到哪个 split")
    parser.add_argument("--min-side", type=int, default=480, help="短边下限")
    parser.add_argument("--min-sharpness", type=float, default=60.0, help="清晰度下限（拉普拉斯方差）")
    parser.add_argument("--dedupe", type=int, default=6, help="dHash 汉明距离小于此值判重（0=不去重）")
    parser.add_argument("--no-dedupe-against-library", action="store_true", help="只与批内去重")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_layout.ensure_layout()
    if args.ping:
        return cmd_ping(args)
    if args.stats:
        return cmd_stats(args)
    return cmd_collect(args)


if __name__ == "__main__":
    sys.exit(main())
