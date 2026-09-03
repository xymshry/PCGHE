"""Command-line interface for download and plaintext experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig, FeatureConfig, ModelConfig
from .dataset import download_mitdb
from .experiment import run_baselines, run_dimension_sweep, run_experiment


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, default=Path("cache/mitdb_nv_beats.npz"))
    parser.add_argument("--force-cache", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcghe-ecg",
        description="Patient-independent plaintext ECG experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download MIT-BIH records")
    download.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))

    run = subparsers.add_parser("run", help="Run one plaintext configuration")
    _common_run_arguments(run)
    run.add_argument("--feature", choices=("raw", "dwt"), default="dwt")
    run.add_argument("--pca-components", type=int, default=32)
    run.add_argument("--no-pca", action="store_true")

    baselines = subparsers.add_parser("baselines", help="Run four plaintext baselines")
    _common_run_arguments(baselines)

    sweep = subparsers.add_parser("sweep", help="Run the DWT PCA dimension sweep")
    _common_run_arguments(sweep)
    sweep.add_argument(
        "--dimensions", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256]
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        download_mitdb(args.data_dir)
        print(f"MIT-BIH records are available in {args.data_dir.resolve()}")
        return

    if args.command == "run":
        components = None if args.no_pca else args.pca_components
        config = ExperimentConfig(
            feature=FeatureConfig(kind=args.feature),
            model=ModelConfig(pca_components=components),
        )
        result = run_experiment(
            args.data_dir,
            args.output_dir,
            args.cache_file,
            config,
            args.force_cache,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "baselines":
        frame = run_baselines(
            args.data_dir,
            args.output_dir,
            args.cache_file,
            args.force_cache,
        )
        print(frame.to_string(index=False))
        return

    frame = run_dimension_sweep(
        args.data_dir,
        args.output_dir,
        args.cache_file,
        tuple(args.dimensions),
        args.force_cache,
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
