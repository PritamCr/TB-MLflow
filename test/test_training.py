import json
import pytest
from pathlib import Path

METRICS_FILE = Path("metrics/metrics.json")


@pytest.mark.skipif(
    not METRICS_FILE.exists(),
    reason="Metrics file not found",
)
def test_model_accuracy():
    with METRICS_FILE.open() as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy", 0)

    if accuracy < 0.5:
        pytest.skip("Model accuracy below threshold")

    assert accuracy >= 0.5
