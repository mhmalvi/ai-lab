"""Tests for the pure parts of the QLoRA SFT recipe (AAA: Arrange, Act, Assert).

No GPU, Unsloth, TRL or transformers is needed; the training call is not tested.
"""
import subprocess

import pytest

from sft_recipe import (
    DEFAULTS,
    GGUF_QUANT,
    PIP_STEPS,
    REQUIRED,
    RecipeError,
    TimeBudget,
    export_artifacts,
    install_stack,
    merge_hyperparams,
    pick_chat_column,
    throughput,
    to_chat,
)

GIVEN = {
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "learning_rate": 2e-4,
    "max_seq_length": 1024,
    "max_steps": 300,
}


def test_merge_keeps_spec_values_over_defaults():
    # Arrange
    given = GIVEN | {"batch_size": 8}
    # Act
    params = merge_hyperparams(given)
    # Assert
    assert params["batch_size"] == 8
    assert params["seed"] == DEFAULTS["seed"]
    assert set(REQUIRED) <= set(params)


def test_merge_refuses_missing_required_key():
    # Arrange
    given = {k: v for k, v in GIVEN.items() if k != "max_steps"}
    # Act / Assert
    with pytest.raises(RecipeError, match="max_steps"):
        merge_hyperparams(given)


def test_merge_refuses_unknown_key():
    # Arrange
    given = GIVEN | {"num_iterations": 5}
    # Act / Assert
    with pytest.raises(RecipeError, match="num_iterations"):
        merge_hyperparams(given)


def test_sharegpt_turns_become_chat_roles():
    # Arrange
    conversation = [
        {"from": "system", "value": "Be brief."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello"},
    ]
    # Act
    turns = to_chat(conversation)
    # Assert
    assert [t["role"] for t in turns] == ["system", "user", "assistant"]
    assert turns[2]["content"] == "Hello"


def test_openai_turns_pass_through():
    # Arrange
    conversation = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]
    # Act / Assert
    assert to_chat(conversation) == conversation


@pytest.mark.parametrize(
    "conversation",
    [
        [],
        "not a list",
        [{"role": "user", "content": "only a question"}],
        [{"role": "robot", "content": "x"}, {"role": "assistant", "content": "y"}],
        [{"role": "user", "content": 3}, {"role": "assistant", "content": "y"}],
    ],
)
def test_bad_conversations_are_refused(conversation):
    # Arrange: conversation from the parameter list
    # Act / Assert
    with pytest.raises(RecipeError):
        to_chat(conversation)


def test_chat_column_is_picked_or_refused():
    # Arrange
    with_chat, without = ["id", "conversations"], ["text", "label"]
    # Act / Assert
    assert pick_chat_column(with_chat) == "conversations"
    with pytest.raises(RecipeError, match="no chat column"):
        pick_chat_column(without)


def test_throughput_prefers_logged_tokens():
    # Arrange
    history = [{"num_tokens": 1000}, {"loss": 1.2}, {"num_tokens": 4000}]
    # Act
    result = throughput(history, train_runtime=20.0, estimate=99)
    # Assert
    assert result == {"num_tokens": 4000, "tokens_per_second": 200.0, "tokens_estimated": False}


def test_throughput_falls_back_to_flagged_estimate_and_survives_zero_runtime():
    # Arrange
    history = [{"loss": 1.0}]
    # Act
    result = throughput(history, train_runtime=0.0, estimate=500)
    # Assert
    assert result == {"num_tokens": 500, "tokens_per_second": 0.0, "tokens_estimated": True}


def test_time_budget_stops_after_its_fraction_of_the_timeout():
    # Arrange
    now = [0.0]
    budget = TimeBudget(1000, fraction=0.6, clock=lambda: now[0])
    # Act
    now[0] = 599.0
    early = budget.exhausted()
    now[0] = 601.0
    late = budget.exhausted()
    # Assert
    assert early is False
    assert late is True


@pytest.mark.parametrize("timeout", [None, 0, -5, True, "3600"])
def test_time_budget_without_a_valid_timeout_never_stops(timeout):
    # Arrange
    now = [0.0]
    budget = TimeBudget(timeout, clock=lambda: now[0])
    # Act
    now[0] = 10**9
    # Assert
    assert budget.exhausted() is False


def test_install_stack_runs_every_pinned_step_in_order():
    # Arrange
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    # Act
    install_stack(run=run, python="python3")
    # Assert
    assert [c[5:] for c in calls] == [list(step) for step in PIP_STEPS]
    assert any("unsloth==" in arg for c in calls for arg in c)


def test_install_stack_stops_at_first_failure():
    # Arrange
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "no matching distribution")

    # Act / Assert
    with pytest.raises(RecipeError, match="no matching distribution"):
        install_stack(run=run, python="python3")
    assert len(calls) == 1


class FakeModel:
    """Records saves; the GGUF export can be made to fail."""

    def __init__(self, gguf_error=None):
        self.saved = []
        self.gguf_error = gguf_error

    def save_pretrained(self, path):
        self.saved.append(("adapter", path))

    def save_pretrained_merged(self, path, tokenizer, save_method):
        self.saved.append((save_method, path))

    def save_pretrained_gguf(self, path, tokenizer, quantization_method):
        if self.gguf_error:
            raise self.gguf_error
        self.saved.append((quantization_method, path))


class FakeTokenizer:
    def save_pretrained(self, path):
        pass


def test_export_saves_adapter_then_merged_then_gguf(tmp_path):
    # Arrange
    model = FakeModel()
    # Act
    saved, errors = export_artifacts(model, FakeTokenizer(), tmp_path)
    # Assert
    assert [kind for kind, _ in model.saved] == ["adapter", "merged_16bit", GGUF_QUANT]
    assert saved == {"adapter": "adapter", "merged_16bit": "merged_16bit", "gguf": "gguf"}
    assert errors == {}


def test_failed_gguf_export_is_recorded_and_adapter_is_kept(tmp_path):
    # Arrange
    model = FakeModel(gguf_error=RuntimeError("llama.cpp build failed"))
    # Act
    saved, errors = export_artifacts(model, FakeTokenizer(), tmp_path)
    # Assert
    assert "adapter" in saved and "gguf" not in saved
    assert errors == {"gguf": "RuntimeError: llama.cpp build failed"}
