"""Tests for the fail-closed job-spec validator (AAA: Arrange, Act, Assert)."""
import copy

import pytest

yaml = pytest.importorskip("yaml")

from jobspec import REQUIRED_HYPERPARAMS, JobSpecError, from_dict, load  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SHA256 = "ab" * 32

BASE = {
    "name": "banking77-modernbert",
    "recipe": "classifier_ft",
    "backend": "kaggle",
    "base_model": {"repo": "answerdotai/ModernBERT-base", "revision": COMMIT},
    "dataset": {"source": "hf", "ref": "PolyAI/banking77", "revision": COMMIT},
    "data_policy": "public",
    "hyperparams": {"learning_rate": 5e-5, "epochs": 3},
    "max_hours": 2,
    "eval": {"task": "holdout", "metric": "macro_f1"},
}

SFT_PARAMS = {
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    "max_steps": 300,
}


def spec_with(**changes):
    data = copy.deepcopy(BASE)
    data.update(changes)
    return data


def field_of(data):
    with pytest.raises(JobSpecError) as caught:
        from_dict(data)
    return caught.value.field


def test_valid_spec_is_frozen_and_hours_normalised():
    # Arrange
    data = spec_with()
    # Act
    spec = from_dict(data)
    # Assert
    assert spec.max_hours == 2.0 and isinstance(spec.max_hours, float)
    with pytest.raises(AttributeError):
        spec.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.hyperparams["epochs"] = 9  # type: ignore[index]


def test_unknown_top_level_key_is_refused():
    # Arrange
    data = spec_with(gpu="t4")
    # Act / Assert
    assert field_of(data) == "gpu"


def test_branch_name_instead_of_commit_is_refused():
    # Arrange
    data = spec_with(base_model={"repo": "answerdotai/ModernBERT-base", "revision": "main"})
    # Act / Assert
    assert field_of(data) == "base_model.revision"


def test_qlora_recipe_requires_its_hyperparameters():
    # Arrange
    params = dict(SFT_PARAMS)
    del params["lora_r"]
    data = spec_with(recipe="llm_sft_unsloth", hyperparams=params)
    # Act
    with pytest.raises(JobSpecError) as caught:
        from_dict(data)
    # Assert
    assert caught.value.field == "hyperparams"
    assert "lora_r" in str(caught.value)


def test_qlora_recipe_with_all_required_hyperparameters_passes():
    # Arrange
    data = spec_with(recipe="llm_sft_unsloth", hyperparams=SFT_PARAMS)
    # Act
    spec = from_dict(data)
    # Assert
    assert set(REQUIRED_HYPERPARAMS["llm_sft_unsloth"]) <= set(spec.hyperparams)


def test_qlora_recipe_cannot_run_locally():
    # Arrange
    data = spec_with(recipe="llm_sft_unsloth", backend="local", hyperparams=SFT_PARAMS)
    # Act / Assert
    assert field_of(data) == "backend"


@pytest.mark.parametrize(
    ("key", "value"),
    [("epochs", True), ("epochs", 2.5), ("learning_rate", 1.0), ("learning_rate", float("nan"))],
)
def test_hyperparameters_are_type_and_bound_checked(key, value):
    # Arrange
    data = spec_with(hyperparams={key: value})
    # Act / Assert
    assert field_of(data) == f"hyperparams.{key}"


def test_hyperparameter_from_another_recipe_is_refused():
    # Arrange
    data = spec_with(hyperparams={"lora_r": 8})
    # Act / Assert
    assert field_of(data) == "hyperparams.lora_r"


def test_raw_private_data_never_leaves_the_lab_machine():
    # Arrange
    data = spec_with(
        dataset={"source": "local-raw", "ref": "inbox_raw", "revision": SHA256},
        data_policy="raw-local-only",
    )
    # Act / Assert
    assert field_of(data) == "backend"


def test_redacted_private_data_cannot_be_labelled_public():
    # Arrange
    data = spec_with(
        backend="local",
        dataset={"source": "local-redacted", "ref": "inbox_redacted", "revision": SHA256},
        data_policy="public",
    )
    # Act / Assert
    assert field_of(data) == "data_policy"


def test_max_hours_over_backend_cap_is_refused():
    # Arrange
    data = spec_with(backend="local", max_hours=7)
    # Act / Assert
    assert field_of(data) == "max_hours"


def test_kaggle_dataset_needs_integer_version():
    # Arrange
    data = spec_with(dataset={"source": "kaggle", "ref": "owner/some-data", "version": "1"})
    # Act / Assert
    assert field_of(data) == "dataset.version"


def test_load_reads_yaml_and_reports_bad_yaml_as_root_error(tmp_path):
    # Arrange
    good, bad = tmp_path / "good.yaml", tmp_path / "bad.yaml"
    good.write_text(yaml.safe_dump(BASE))
    bad.write_text("name: [unclosed")
    # Act
    spec = load(good)
    # Assert
    assert spec.name == "banking77-modernbert"
    with pytest.raises(JobSpecError) as caught:
        load(bad)
    assert caught.value.field == "<root>"
