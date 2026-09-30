"""Train and score CSV time series without repository-specific paths."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from . import DEFAULT_SEEDS, MSCAD


def load_series(path, train_length=None, label_column="Label"):
    """Read numeric features and optional binary labels; never drop rows."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(set(header)) != len(header):
            raise ValueError(f"{path.name}: CSV needs a unique, nonempty header")
        label_index = header.index(label_column) if label_column in header else None
        feature_indices = [i for i in range(len(header)) if i != label_index]
        if not feature_indices:
            raise ValueError(f"{path.name}: no feature columns")
        rows = []
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"{path.name}: row {line_number} has missing/extra columns")
            try:
                rows.append([float(value) for value in row])
            except ValueError as error:
                raise ValueError(f"{path.name}: row {line_number} must contain finite numbers") from error
    if not rows:
        raise ValueError(f"{path.name}: CSV has no data rows")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: all values must be finite; rows are never dropped")
    labels = None if label_index is None else values[:, label_index]
    if labels is not None:
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{path.name}: {label_column} must be binary (0/1)")
        labels = labels.astype(np.int64)
    if train_length is None:
        match = re.search(r"_tr_(\d+)(?:_|\.csv$)", path.name)
        if match is None:
            raise ValueError(f"{path.name}: supply --train-length or a filename containing _tr_N_")
        train_length = int(match.group(1))
    if not 0 < train_length < len(values):
        raise ValueError(f"{path.name}: train length must leave nonempty training and test segments")
    return values[:, feature_indices], labels, train_length


def select_files(data, data_dir, file_list):
    """Select files explicitly or from the official TSB-AD file_name list."""
    if data:
        paths = [Path(path).resolve() for path in data]
    else:
        root = Path(data_dir).resolve()
        with Path(file_list).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "file_name" not in reader.fieldnames:
                raise ValueError("File list must have a file_name column")
            paths = []
            for row in reader:
                name = row["file_name"]
                if not name:
                    raise ValueError("File list contains an empty file_name")
                path = (root / name).resolve()
                if root not in path.parents:
                    raise ValueError("File list entries must stay inside the data directory")
                paths.append(path)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Select a nonempty list of unique input files")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Input file does not exist: {path}")
    return paths


def evaluate_scores(scores, labels, series, metric_window=None):
    """Use the official TSB-AD implementation, retaining threshold-free metrics."""
    try:
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank
    except ImportError as error:
        raise ImportError("Evaluation needs the optional dependencies: pip install '.[evaluation]'") from error
    if labels is None or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both normal (0) and anomalous (1) labels")
    period = metric_window if metric_window is not None else int(find_length_rank(series[:, 0], rank=1))
    if period <= 0:
        raise ValueError("Metric window must be positive")
    # No min-max scaling, threshold selection, or label-based score adjustment.
    values = get_metrics(scores, labels, slidingWindow=period)
    result = {key: float(values[key]) for key in ("AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC")}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("TSB-AD returned nonfinite metrics for this series")
    return result, period


def build_parser():
    parser = argparse.ArgumentParser(description="Train MSCAD on a normal prefix and score CSV time series.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", nargs="+", type=Path, help="One or more numeric CSV files")
    group.add_argument("--file-list", type=Path, help="Official TSB-AD CSV list with a file_name column")
    parser.add_argument("--data-dir", type=Path, help="Data directory used with --file-list")
    parser.add_argument("--train-length", type=int, help="Override _tr_N_ in the input filename")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--score-region", choices=("test", "full"), default="test",
                        help="test: score only the held-out suffix; full: legacy full-series protocol")
    parser.add_argument("--evaluate", action="store_true", help="Compute official TSB-AD AUC/VUS metrics")
    parser.add_argument("--metric-window", type=int, help="VUS tolerance window; default: first-channel ACF estimate")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc.; default auto-selects CUDA/CPU")
    parser.add_argument("--win-size", type=int, default=128)
    parser.add_argument("--patch-sizes", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers-per-scale", type=int, default=2)
    parser.add_argument("--bridge-depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.file_list and not args.data_dir:
        parser.error("--file-list requires --data-dir")
    if args.data_dir and not args.file_list:
        parser.error("--data-dir is used with --file-list")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("--seeds must be unique integers between 0 and 2**32-1")
    if args.metric_window is not None and args.metric_window <= 0:
        parser.error("--metric-window must be positive")
    config_keys = ("win_size", "patch_sizes", "d_model", "n_heads", "n_layers_per_scale", "bridge_depth",
                   "epochs", "patience", "batch_size", "lr", "validation_size", "dropout")
    config = {key: getattr(args, key) for key in config_keys}
    try:
        paths = select_files(args.data, args.data_dir, args.file_list)
        if len({path.stem for path in paths}) != len(paths):
            raise ValueError("Input filenames must have unique stems to prevent output collisions")
        # Check all output names before starting expensive training.
        for path in paths:
            for seed in args.seeds:
                stem = f"{path.stem}_seed{seed}_{args.score_region}"
                for suffix in (".npy", ".json"):
                    if (args.output_dir / (stem + suffix)).exists():
                        raise ValueError(f"Output already exists: {stem + suffix}; choose a new --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            data, labels, train_length = load_series(path, args.train_length, args.label_column)
            start = train_length if args.score_region == "test" else 0
            scored_data = data[start:]
            scored_labels = None if labels is None else labels[start:]
            if args.evaluate:
                if scored_labels is None or set(np.unique(scored_labels)) != {0, 1}:
                    raise ValueError("Evaluation requires both normal (0) and anomalous (1) labels in the scored region")
                # Fail on a missing optional installation before training.
                from TSB_AD.evaluation.metrics import get_metrics  # noqa: F401
                from TSB_AD.utils.slidingWindows import find_length_rank  # noqa: F401
            for seed in args.seeds:
                print(f"{path.name}: seed={seed}, score_region={args.score_region}", flush=True)
                started = time.perf_counter()
                detector = MSCAD(**config, seed=seed, device=args.device).fit(data[:train_length])
                scores = detector.decision_function(scored_data)
                stem = f"{path.stem}_seed{seed}_{args.score_region}"
                report = {"dataset": path.name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "seed": seed, "score_region": args.score_region, "start_index": start,
                          "train_length": train_length, "length": len(data), "channels": data.shape[1],
                          "config": config, "device": str(detector.device),
                          "numpy_version": np.__version__, "torch_version": torch.__version__,
                          "scores_file": stem + ".npy", "elapsed_seconds": time.perf_counter() - started}
                if args.evaluate:
                    report["metrics"], report["metric_window"] = evaluate_scores(
                        scores, scored_labels, scored_data, args.metric_window)
                np.save(args.output_dir / report["scores_file"], scores)
                (args.output_dir / (stem + ".json")).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(f"  saved {len(scores)} scores in {report['elapsed_seconds']:.1f}s", flush=True)
    except ImportError as error:
        parser.error(f"{error}. For metrics, install optional dependencies: pip install '.[evaluation]'")
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
