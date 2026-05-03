"""
Unit tests for the ArtifactManager module.
"""
import pytest
import os
import json
from core import artifact_manager as am

pytestmark = pytest.mark.unit

def test_get_run_id_format():
    """Verify that the run ID has the correct format."""
    run_id = am.get_run_id()
    assert isinstance(run_id, str)
    # Simple regex to check YYYYMMDD_HHMMSS_ms format
    assert len(run_id.split('_')) == 3
    assert run_id.split('_')[0].isdigit() and len(run_id.split('_')[0]) == 8


def test_get_run_path_creates_directory(tmp_path):
    """Verify that get_run_path creates the necessary directories."""
    exp_name = "test_exp"
    model_name = "test_model"
    run_id = "20250101_120000_000"

    with pytest.MonkeyPatch.context() as m:
        m.setattr(am, "BASE_RESULTS_PATH", str(tmp_path))

        run_path = am.get_run_path(exp_name, model_name, run_id)

        assert os.path.isdir(run_path)

        expected_path = tmp_path / exp_name / f"{model_name}_{run_id}"
        assert run_path == str(expected_path)
        assert os.path.basename(run_path) == f"{model_name}_{run_id}"
        assert os.path.basename(os.path.dirname(run_path)) == exp_name


def test_save_and_load_json(tmp_path):
    """Verify that a dictionary can be saved to and loaded from a JSON file."""
    data = {"param1": 10, "param2": "value"}
    filename = "test.json"

    am.save_json(data, str(tmp_path), filename)

    loaded_data = am.load_json(str(tmp_path), filename)

    assert data == loaded_data


def test_load_json_file_not_found(tmp_path):
    """Verify that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        am.load_json(str(tmp_path), "non_existent.json")


def test_find_latest_run_id(tmp_path):
    """Verify that the function correctly identifies the latest run ID."""
    exp_name = "test_exp"
    model_name = "test_model"

    base_path = tmp_path / exp_name

    # Create some dummy run directories
    (base_path / f"{model_name}_20250101_110000_000").mkdir(parents=True)
    (base_path / f"{model_name}_20250101_120000_000").mkdir()
    (base_path / "other_model_20250101_130000_000").mkdir() # Should be ignored

    # Patch BASE_RESULTS_PATH to point to our temp directory
    with pytest.MonkeyPatch.context() as m:
        m.setattr(am, "BASE_RESULTS_PATH", str(tmp_path))

        latest_id = am.find_latest_run_id(exp_name, model_name)

    assert latest_id == "20250101_120000_000"


def test_find_latest_run_id_no_runs(tmp_path):
    """Verify it returns None when no matching runs are found."""
    with pytest.MonkeyPatch.context() as m:
        # Point to a non-existent base path
        m.setattr(am, "BASE_RESULTS_PATH", str(tmp_path / "non_existent"))

        assert am.find_latest_run_id("non_existent_exp", "any_model") is None