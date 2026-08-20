from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen


MODELS = {
    "hrnet-w32": {
        "url": (
            "https://download.openmmlab.com/mmpose/top_down/hrnet/"
            "hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth"
        ),
        "path": (
            "models/mmpose/hrnet/"
            "hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth"
        ),
    },
    "hrnet-w48-dark": {
        "url": (
            "https://download.openmmlab.com/mmpose/top_down/hrnet/"
            "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
        ),
        "path": (
            "models/mmpose/hrnet/"
            "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HRNet COCO-WholeBody pose checkpoints.")
    parser.add_argument("--model",
                        choices=[*MODELS, "all"],
                        default="hrnet-w48-dark",
                        help="HRNet checkpoint to download.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print download targets without downloading.")
    parser.add_argument("--force",
                        action="store_true",
                        help="Download even if the target file exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        item = MODELS[name]
        target = Path(item["path"])
        url = str(item["url"])
        if args.dry_run:
            print(f"{name}: {url} -> {target}")
            continue
        download(url, target, force=args.force)


def download(url: str, target: Path, force: bool) -> None:
    if target.exists() and not force:
        print(f"[skip] exists: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".part")
    print(f"[download] {url}")
    print(f"[target] {target}")
    with urlopen(url) as response, tmp_path.open("wb") as file:
        shutil.copyfileobj(response, file)
    tmp_path.replace(target)
    print(f"[done] {target}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
