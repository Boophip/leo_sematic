"""Train the smoke MLP task-quality proxy from profiling CSV rows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.proxy.quality_proxy import MODEL_TYPES, MODEL_TYPE_HIST_GBDT, train_quality_proxy_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=(
            ROOT
            / "data"
            / "profiling"
            / "dota_demo_1024_o200"
            / "test_smoke"
            / "roi_profile_smoke.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "proxy" / "smoke",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default=MODEL_TYPE_HIST_GBDT)
    parser.add_argument("--include-image-features", action="store_true")
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument(
        "--hidden-layer-sizes",
        type=int,
        nargs="+",
        default=[32, 16],
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    _prepare_output_dir(args.output_dir, exist_ok=args.exist_ok)
    print("Quality proxy training estimate: about 1-5 minutes; not an RL run.")
    summary = train_quality_proxy_smoke(
        args.profile_csv,
        args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        max_iter=args.max_iter,
        hidden_layer_sizes=args.hidden_layer_sizes,
        model_type=args.model_type,
        include_image_features=args.include_image_features,
        quality_threshold=args.quality_threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
