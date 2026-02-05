import json
from pathlib import Path

from churn_ml_decision.track import file_sha256, log_run


def test_file_sha256_deterministic(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_text("hello world")
    h1 = file_sha256(f)
    h2 = file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest length


def test_file_sha256_different_content(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content A")
    f2.write_text("content B")
    assert file_sha256(f1) != file_sha256(f2)


def test_log_run_creates_file(tmp_path: Path):
    log_path = tmp_path / "logs" / "experiments.jsonl"
    log_run(log_path, {"stage": "test", "metric": 0.9})

    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert "run_id" in record
    assert "timestamp" in record
    assert record["stage"] == "test"
    assert record["metric"] == 0.9


def test_log_run_appends(tmp_path: Path):
    log_path = tmp_path / "experiments.jsonl"
    log_run(log_path, {"run": 1})
    log_run(log_path, {"run": 2})

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2

    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["run"] == 1
    assert r2["run"] == 2
    assert r1["run_id"] != r2["run_id"]
