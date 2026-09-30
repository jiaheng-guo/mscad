"""Exercise CSV boundaries and the installed command with real tiny inputs."""
import csv
import json
import subprocess
import sys

import numpy as np
import pytest


def make_csv(path, length=48, bad_row=None):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["value", "Label"])
        for index in range(length):
            writer.writerow(["nan" if index == bad_row else np.sin(index / 4),
                             int(index >= 40)])
    return path


def test_loader_preserves_filename_training_boundary(tmp_path):
    from mscad.cli import load_series
    path = make_csv(tmp_path / "example_tr_32_1st_40.csv")
    data, labels, train_length = load_series(path)
    assert data.shape == (48, 1)
    assert train_length == 32
    np.testing.assert_array_equal(labels, np.arange(48) >= 40)


def test_loader_rejects_missing_rows_instead_of_shifting_boundary(tmp_path):
    from mscad.cli import load_series
    path = make_csv(tmp_path / "example_tr_32_1st_40.csv", bad_row=4)
    with pytest.raises(ValueError, match="finite"):
        load_series(path)


@pytest.mark.parametrize("train_length", [0, 48, 49])
def test_loader_rejects_empty_train_or_test(tmp_path, train_length):
    from mscad.cli import load_series
    path = make_csv(tmp_path / "custom.csv")
    with pytest.raises(ValueError, match="train"):
        load_series(path, train_length=train_length)


def test_custom_csv_needs_training_length(tmp_path):
    from mscad.cli import load_series
    with pytest.raises(ValueError, match="train-length"):
        load_series(make_csv(tmp_path / "custom.csv"))


def test_default_seeds_are_one_through_five():
    from mscad.cli import build_parser
    args = build_parser().parse_args(["--data", "example.csv"])
    assert args.seeds == [1, 2, 3, 4, 5]


def test_real_cli_writes_only_requested_run_outputs(tmp_path):
    path = make_csv(tmp_path / "example_tr_32_1st_40.csv")
    output = tmp_path / "outputs"
    command = [sys.executable, "-m", "mscad", "--data", str(path),
               "--output-dir", str(output), "--seeds", "1", "2",
               "--win-size", "8", "--patch-sizes", "2", "4",
               "--d-model", "8", "--n-heads", "2", "--n-layers-per-scale", "1",
               "--bridge-depth", "2", "--epochs", "1", "--batch-size", "8",
               "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    reports = sorted(output.glob("*.json"))
    assert len(reports) == 2
    for path in reports:
        report = json.loads(path.read_text())
        assert report["seed"] in (1, 2)
        assert report["score_region"] == "test"
        assert report["start_index"] == 32
        score = np.load(output / report["scores_file"])
        assert score.shape == (16,)
        assert np.isfinite(score).all()
        assert (score >= 0).all()
    # Reruns must not silently overwrite a previous experiment.
    repeat = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert repeat.returncode != 0
    assert "already exists" in repeat.stderr


def test_file_list_selects_exact_files_and_rejects_escape(tmp_path):
    from mscad.cli import select_files
    path = make_csv(tmp_path / "example_tr_32_1st_40.csv")
    listing = tmp_path / "list.csv"
    listing.write_text("file_name\n" + path.name + "\n")
    assert select_files(None, tmp_path, listing) == [path.resolve()]
    listing.write_text("file_name\n../outside.csv\n")
    with pytest.raises(ValueError, match="data directory"):
        select_files(None, tmp_path, listing)


def test_official_metrics_on_separated_scores():
    pytest.importorskip("TSB_AD.evaluation.metrics")
    pytest.importorskip("statsmodels")
    from mscad.cli import evaluate_scores
    labels = np.zeros(100, dtype=int)
    labels[40:50] = 1
    scores = np.linspace(0, 0.1, 100) + labels
    metrics, window = evaluate_scores(scores, labels, scores[:, None], metric_window=5)
    assert window == 5
    assert metrics["AUC-ROC"] == pytest.approx(1)
    assert metrics["AUC-PR"] == pytest.approx(1)
    assert all(0 <= value <= 1 for value in metrics.values())
    assert set(metrics) == {"AUC-ROC", "AUC-PR", "VUS-ROC", "VUS-PR"}
