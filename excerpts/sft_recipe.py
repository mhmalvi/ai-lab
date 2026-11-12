"""Recipe: QLoRA supervised fine-tune of a chat LLM with Unsloth + TRL + PEFT, for a T4.

Excerpt from the private lab's ``llm_sft_unsloth`` recipe. Sanitized and trimmed:
  * kept: the required-hyperparameter contract, the defaults merge, chat-column
    and role normalisation (OpenAI and ShareGPT turns), the token-throughput
    metric, the session time budget, the pinned runtime install and the
    adapter-first export order;
  * removed: the shared recipe helpers (job loading, CLI flags, output dir,
    dataset loading, metrics writing) and the kernel bundling step.

Status: this recipe is built and its pure parts are unit-tested. It has NOT been
run on a GPU yet: the first run is pending GPU access. ``train_and_export`` is
shown for completeness and is not exercised by the tests.

Plan of a run: load the base model in 4-bit with Unsloth's ``FastLanguageModel``,
attach a LoRA adapter (r/alpha/dropout from the spec), train with TRL's
``SFTTrainer`` on a chat-template dataset, then save three artifacts: the LoRA
adapter, a merged 16-bit model and a q4_k_m GGUF. The T4 has no bf16, so training
runs in fp16.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

RECIPE = "llm_sft_unsloth"
# The spec must state these (jobspec.REQUIRED_HYPERPARAMS enforces the same set at
# submit time). They decide the adapter's shape and the GPU-hours spent.
REQUIRED = ("lora_r", "lora_alpha", "lora_dropout", "learning_rate", "max_seq_length", "max_steps")
DEFAULTS = {"epochs": 1, "batch_size": 2, "grad_accum": 4, "seed": 3407, "warmup_ratio": 0.05}
GGUF_QUANT = "q4_k_m"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CHAT_COLUMNS = ("messages", "conversations")
# ShareGPT speaker names -> chat-template roles.
ROLE_ALIASES = {
    "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
    "model": "assistant", "system": "system",
}  # fmt: skip
# Saving the merged 16-bit model and building the GGUF (Unsloth compiles llama.cpp)
# can take a large share of the session, so training stops at this fraction.
TIME_BUDGET_FRACTION = 0.6

# Runtime install, following Unsloth's own Kaggle notebook. Unsloth is not in the
# Kaggle image, so the recipe installs it inside the kernel. The notebook leaves
# unsloth unpinned; the lab pins it so that a rerun installs the same code.
PIP_STEPS: tuple[tuple[str, ...], ...] = (
    ("torch", "torchvision", "torchaudio", "xformers",
     "--index-url", "https://download.pytorch.org/whl/cu128"),
    ("unsloth==2026.10.2", "unsloth_zoo==2026.10.2"),
    ("--no-deps", "--upgrade", "torchao>=0.16.0"),
    ("transformers==4.57.6",),
    ("--no-deps", "trl==0.22.2"),
)  # fmt: skip


class RecipeError(RuntimeError):
    """The job, its data or the runtime cannot support this recipe (exit code 2)."""


def merge_hyperparams(given: Mapping[str, object]) -> dict:
    """Return the spec's hyperparams merged over ``DEFAULTS``, with ``REQUIRED`` enforced.

    Args:
        given: the ``hyperparams`` mapping from the job spec.

    Returns:
        A new dict holding every key in ``REQUIRED`` and ``DEFAULTS``.

    Raises:
        RecipeError: a required key is missing, or a key is unknown to the recipe.
    """
    missing = [key for key in REQUIRED if key not in given]
    if missing:
        raise RecipeError(f"{RECIPE} requires hyperparams {missing}")
    allowed = set(DEFAULTS) | set(REQUIRED)
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise RecipeError(f"unknown hyperparams for this recipe: {unknown}")
    return {**DEFAULTS, **given}


def to_chat(conversation: object) -> list[dict]:
    """Normalise one conversation to ``[{"role", "content"}, ...]``.

    Both OpenAI-style ``role``/``content`` turns and ShareGPT ``from``/``value``
    turns are accepted.

    Args:
        conversation: a list of turn dicts.

    Returns:
        The turns as role/content dicts, with roles in user/assistant/system.

    Raises:
        RecipeError: the conversation is empty or not a list, a turn has an unknown
            role, the content is not a string, or no assistant turn is present.
    """
    if not isinstance(conversation, list) or not conversation:
        raise RecipeError("a conversation must be a non-empty list of turns")
    turns = []
    for turn in conversation:
        if not isinstance(turn, Mapping):
            raise RecipeError(f"conversation turn is not an object: {type(turn).__name__}")
        speaker = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value"))
        role = ROLE_ALIASES.get(str(speaker).lower())
        if role is None or not isinstance(content, str):
            raise RecipeError(f"bad turn: role {speaker!r}, content type {type(content).__name__}")
        turns.append({"role": role, "content": content})
    if not any(t["role"] == "assistant" for t in turns):
        raise RecipeError("conversation has no assistant turn to learn from")
    return turns


def pick_chat_column(column_names: Sequence[str]) -> str:
    """Return the dataset column that holds conversations.

    Args:
        column_names: the dataset's columns.

    Returns:
        The first of ``CHAT_COLUMNS`` that is present.

    Raises:
        RecipeError: if none of ``CHAT_COLUMNS`` is present. We never guess.
    """
    column = next((c for c in CHAT_COLUMNS if c in column_names), None)
    if column is None:
        raise RecipeError(f"no chat column ({CHAT_COLUMNS}) in {list(column_names)}")
    return column


def throughput(log_history: Sequence[Mapping], train_runtime: float, estimate: int) -> dict:
    """Tokens trained per second.

    TRL logs a cumulative ``num_tokens``, and we use it when it is there. Without it,
    ``estimate`` is used and flagged as such.

    Args:
        log_history: ``trainer.state.log_history``.
        train_runtime: seconds spent in ``trainer.train()``.
        estimate: fallback token count (rows seen x mean tokens per row).

    Returns:
        ``{"num_tokens", "tokens_per_second", "tokens_estimated"}``.
    """
    logged = [e["num_tokens"] for e in log_history if isinstance(e.get("num_tokens"), int | float)]
    tokens = int(max(logged)) if logged else int(estimate)
    rate = tokens / train_runtime if train_runtime > 0 else 0.0
    return {
        "num_tokens": tokens,
        "tokens_per_second": round(rate, 1),
        "tokens_estimated": not logged,
    }


class TimeBudget:
    """Decides when training must stop so the exports still fit in the session."""

    def __init__(
        self,
        timeout_seconds: object,
        fraction: float = TIME_BUDGET_FRACTION,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Start the budget clock.

        Args:
            timeout_seconds: the kernel's session timeout. A missing or non-positive
                value means no budget (never stop early).
            fraction: share of the timeout that training may use.
            clock: monotonic seconds (injectable for tests).
        """
        self.budget = None
        valid = isinstance(timeout_seconds, int | float) and not isinstance(timeout_seconds, bool)
        if valid and timeout_seconds > 0:
            self.budget = timeout_seconds * fraction
        self.clock = clock
        self.started = clock()

    def exhausted(self) -> bool:
        """Return whether training has used up its share of the session."""
        return self.budget is not None and self.clock() - self.started > self.budget


def time_budget_callback(timeout_seconds: object):
    """Return a ``transformers.TrainerCallback`` that stops training when the budget is spent.

    Args:
        timeout_seconds: the kernel's session timeout (see ``TimeBudget``).

    Returns:
        A callback instance for ``SFTTrainer(callbacks=[...])``.
    """
    from transformers import TrainerCallback

    budget = TimeBudget(timeout_seconds)

    class StopAtBudget(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if budget.exhausted():
                control.should_training_stop = True
            return control

    return StopAtBudget()


def install_stack(
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    python: str = sys.executable,
) -> None:
    """Pip-install the Unsloth stack into the running kernel, one ``PIP_STEPS`` line at a time.

    Args:
        run: ``subprocess.run``-compatible callable (injectable for tests).
        python: the interpreter whose site-packages receive the install.

    Raises:
        RecipeError: on the first step that exits non-zero.
    """
    for step in PIP_STEPS:
        argv = [python, "-m", "pip", "install", "-q", *step]
        result = run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-1500:]
            raise RecipeError(
                f"pip install {' '.join(step)} failed (rc={result.returncode}): {tail}"
            )


def export_artifacts(model, tokenizer, out: Path) -> tuple[dict, dict]:
    """Save the adapter, then the merged 16-bit model, then the q4_k_m GGUF.

    The adapter is the irreplaceable output, so it is saved first and a failure
    there propagates. If a later export fails, the error is recorded, not raised,
    so that the adapter and metrics still reach the lab.

    Args:
        model: the trained PEFT model (Unsloth-patched).
        tokenizer: its tokenizer.
        out: output dir.

    Returns:
        ``(saved, errors)``, both keyed by ``adapter``/``merged_16bit``/``gguf``.
        ``saved`` maps each to a path relative to ``out``, ``errors`` to a message.
    """
    adapter = out / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    saved, errors = {"adapter": "adapter"}, {}
    exports = (
        ("merged_16bit", model.save_pretrained_merged, {"save_method": "merged_16bit"}),
        ("gguf", model.save_pretrained_gguf, {"quantization_method": GGUF_QUANT}),
    )
    for name, save, options in exports:
        try:
            save(str(out / name), tokenizer, **options)
            saved[name] = name
        except Exception as exc:  # noqa: BLE001 - keep the adapter even if an export breaks
            errors[name] = f"{type(exc).__name__}: {exc}"[:1000]
    return saved, errors


def train_and_export(spec: Mapping, params: Mapping, train, evaluation, out: Path) -> dict:
    """Train the adapter, evaluate it and export the artifacts (NOT run yet; untested).

    Args:
        spec: the validated job spec as a dict (``base_model.repo``/``revision``,
            ``timeout_seconds``).
        params: hyperparams from ``merge_hyperparams``.
        train: training split with a rendered ``text`` column.
        evaluation: evaluation split with a rendered ``text`` column.
        out: output dir.

    Returns:
        Losses, step count, peak VRAM and the export results.

    Raises:
        RecipeError: unsloth missing or no CUDA device.
    """
    import torch

    # Unsloth patches transformers/TRL/PEFT on import, so it must load before any of them.
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RecipeError(f"unsloth is not importable ({exc}); this recipe needs a GPU kernel")
    if not torch.cuda.is_available():
        raise RecipeError("Unsloth needs a CUDA GPU; none is visible")
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["base_model"]["repo"],
        revision=spec["base_model"]["revision"],
        max_seq_length=params["max_seq_length"],
        dtype=torch.float16,  # T4 (sm_75) has no bf16
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=list(LORA_TARGETS),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=params["seed"],
    )
    config = SFTConfig(
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum"],
        warmup_ratio=params["warmup_ratio"],
        num_train_epochs=params["epochs"],
        max_steps=params["max_steps"],  # wins over epochs; it bounds the GPU-hours
        learning_rate=params["learning_rate"],
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        seed=params["seed"],
        output_dir=str(out / "checkpoints"),
        save_strategy="no",
        report_to="none",  # no tracking service phoning home from a kernel
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=evaluation,
        dataset_text_field="text",
        max_seq_length=params["max_seq_length"],
        args=config,
        callbacks=[time_budget_callback(spec.get("timeout_seconds"))],
    )
    result = trainer.train()
    eval_loss = trainer.evaluate()["eval_loss"]
    saved, errors = export_artifacts(model, tokenizer, out)
    return {
        "train_loss": float(result.training_loss),
        "eval_loss": float(eval_loss),
        "global_step": trainer.state.global_step,
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "artifacts": saved,
        "artifact_errors": errors,
    }
