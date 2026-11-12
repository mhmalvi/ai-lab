"""Job specs: a YAML file describing one training job -> a validated, frozen ``JobSpec``.

Excerpt from the private lab's job-spec module. Sanitized and trimmed:
  * kept: the frozen spec records, every field and cross-field rule, per-recipe
    hyperparameter bounds, the required-hyperparameter table and the YAML loader;
  * changed: the evaluation task names are generic here.

The spec is the contract between the operator (or an agent) and every backend, so
it fails closed: unknown keys, unpinned revisions, unknown recipes, out-of-bounds
hyperparameters and policy contradictions all raise ``JobSpecError`` naming the
offending field (dotted path, e.g. ``base_model.revision``). Nothing is defaulted
silently except ``hyperparams``, which may be empty (recipes own their defaults).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

# Kaggle kernel slugs are lowercase-hyphen; the job name becomes part of one.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$")
HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
KAGGLE_REF_RE = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*$")
LOCAL_REF_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# A git commit is the only HF revision that cannot move under us; branches and tags can.
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
# Local redacted datasets are pinned by the sha256 of their content.
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

BACKENDS = frozenset({"kaggle", "local"})
DATASET_SOURCES = frozenset({"hf", "kaggle", "kaggle-redacted", "local-redacted", "local-raw"})
DATA_POLICIES = frozenset({"public", "redacted-private", "raw-local-only"})
# Our own UNREDACTED data: it may only be read by a job running on the lab machine.
LOCAL_ONLY_SOURCES = frozenset({"local-raw"})
# Our own data, redacted and audited, uploaded to a PRIVATE Kaggle dataset: only a
# Kaggle kernel can mount it, and it is pinned like local-redacted.
KAGGLE_ONLY_SOURCES = frozenset({"kaggle-redacted"})
REDACTED_SOURCES = frozenset({"local-redacted", "kaggle-redacted"})
# inbox = the lab's own labelled email benchmark; holdout = the dataset's own test split.
EVAL_TASKS = frozenset({"inbox", "holdout"})
EVAL_METRICS = frozenset({"accuracy", "macro_f1", "ece"})
# Kaggle sessions die at 12 h; 11.5 leaves time to save artifacts before the kill.
# Local runs share the box with live services; 6 h keeps one from hogging it overnight.
MAX_HOURS = {"kaggle": 11.5, "local": 6.0}
# Recipes too large for the small GPU shared with local services: never run them locally.
KAGGLE_ONLY_RECIPES = frozenset({"llm_sft_unsloth"})
# Hyperparameters a spec must state. A QLoRA run's adapter shape, sequence length and
# step budget decide its GPU-hours and quality, so they are never left to a default.
REQUIRED_HYPERPARAMS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "llm_sft_unsloth": frozenset(
            {"lora_r", "lora_alpha", "lora_dropout", "learning_rate", "max_seq_length", "max_steps"}
        ),
    }
)

TOP_LEVEL_KEYS = (
    "name", "recipe", "backend", "base_model", "dataset",
    "data_policy", "hyperparams", "max_hours", "eval",
)  # fmt: skip
OPTIONAL_TOP_LEVEL = frozenset({"hyperparams"})


class JobSpecError(ValueError):
    """A job spec is invalid. ``field`` is the dotted path of the offending field."""

    def __init__(self, field: str, message: str):
        """Build the error.

        Args:
            field: dotted path of the bad field (``<root>`` for the document itself).
            message: what is wrong with it.
        """
        self.field = field
        super().__init__(f"{field}: {message}")


@dataclass(frozen=True)
class Bound:
    """Allowed type and inclusive range for one hyperparameter."""

    kind: type
    low: float
    high: float


def _int(low: int, high: int) -> Bound:
    """Return an integer bound [low, high]."""
    return Bound(int, low, high)


def _float(low: float, high: float) -> Bound:
    """Return a float bound [low, high] (ints are accepted as floats)."""
    return Bound(float, low, high)


_COMMON = {
    "learning_rate": _float(1e-7, 1e-2),
    "epochs": _int(1, 50),
    "batch_size": _int(1, 256),
    "seed": _int(0, 2**31 - 1),
}
RECIPES: Mapping[str, Mapping[str, Bound]] = MappingProxyType(
    {
        "classifier_ft": MappingProxyType(
            {
                **_COMMON,
                "max_length": _int(16, 4096),
                "weight_decay": _float(0.0, 0.5),
                "warmup_ratio": _float(0.0, 0.5),
                "grad_accum": _int(1, 64),
                # Seeded subsets for smoke and local runs; two rows is the least
                # that can hold two classes.
                "max_train_rows": _int(2, 10_000_000),
                "max_eval_rows": _int(2, 10_000_000),
                # Local CV datasets: 1 = after the folds, fit on every item and save it.
                "final_fit": _int(0, 1),
                "max_folds": _int(1, 10),
            }
        ),
        "setfit_fewshot": MappingProxyType(
            {
                **_COMMON,
                "num_iterations": _int(1, 100),
                "samples_per_label": _int(1, 1000),
                "body_learning_rate": _float(1e-7, 1e-2),
                "max_length": _int(16, 4096),
                "final_fit": _int(0, 1),
                "max_folds": _int(1, 10),
            }
        ),
        "llm_sft_unsloth": MappingProxyType(
            {
                **_COMMON,
                "max_steps": _int(1, 100_000),
                "grad_accum": _int(1, 128),
                "lora_r": _int(1, 256),
                "lora_alpha": _int(1, 512),
                "lora_dropout": _float(0.0, 0.5),
                "max_seq_length": _int(128, 16_384),
                "warmup_ratio": _float(0.0, 0.5),
            }
        ),
        "gpu_smoke": MappingProxyType(
            {
                "matmul_size": _int(128, 16_384),
                "iterations": _int(1, 10_000),
                "seed": _int(0, 2**31 - 1),
            }
        ),
    }
)


@dataclass(frozen=True)
class ModelRef:
    """A base model pinned to an exact Hugging Face commit."""

    repo: str
    revision: str


@dataclass(frozen=True)
class DatasetRef:
    """Training data: HF (commit-pinned), Kaggle (version-pinned) or local redacted (sha256)."""

    source: str
    ref: str
    revision: str | None = None
    version: int | None = None


@dataclass(frozen=True)
class EvalSpec:
    """Which task the trained model is scored on and the headline metric."""

    task: str
    metric: str


@dataclass(frozen=True)
class JobSpec:
    """One validated training job. Build with ``load`` or ``from_dict``, not by hand."""

    name: str
    recipe: str
    backend: str
    base_model: ModelRef
    dataset: DatasetRef
    data_policy: str
    max_hours: float
    eval: EvalSpec
    hyperparams: Mapping[str, int | float] = field(default_factory=lambda: MappingProxyType({}))

    def validate(self) -> None:
        """Check every field and cross-field rule.

        Raises:
            JobSpecError: naming the first offending field.
        """
        _check_choice("name", self.name, None, NAME_RE)
        _check_choice("recipe", self.recipe, RECIPES)
        _check_choice("backend", self.backend, BACKENDS)
        if self.recipe in KAGGLE_ONLY_RECIPES and self.backend != "kaggle":
            raise JobSpecError("backend", f"recipe {self.recipe} runs on kaggle only")
        _check_model(self.base_model)
        _check_dataset(self.dataset)
        _check_policy(self.data_policy, self.dataset)
        if self.dataset.source in LOCAL_ONLY_SOURCES and self.backend != "local":
            raise JobSpecError("backend", f"{self.dataset.source} data never leaves the lab machine")
        if self.dataset.source in KAGGLE_ONLY_SOURCES and self.backend != "kaggle":
            raise JobSpecError("backend", f"{self.dataset.source} data is mounted on kaggle only")
        _check_hyperparams(self.recipe, self.hyperparams)
        _check_hours(self.max_hours, self.backend)
        _check_choice("eval.task", self.eval.task, EVAL_TASKS)
        _check_choice("eval.metric", self.eval.metric, EVAL_METRICS)


def _check_choice(
    path: str, value: object, allowed: Mapping | frozenset | None, pattern: re.Pattern | None = None
) -> None:
    """Require ``value`` to be a string in ``allowed`` and/or matching ``pattern``.

    Args:
        path: dotted field path for the error.
        value: the raw value.
        allowed: permitted values (or None to skip the membership check).
        pattern: regex the value must fully match (or None).

    Raises:
        JobSpecError: if the value is not a str, not allowed, or does not match.
    """
    if not isinstance(value, str):
        raise JobSpecError(path, f"must be a string, got {type(value).__name__}")
    if allowed is not None and value not in allowed:
        raise JobSpecError(path, f"{value!r} is not one of {sorted(allowed)}")
    if pattern is not None and not pattern.fullmatch(value):
        raise JobSpecError(path, f"{value!r} does not match {pattern.pattern}")


def _check_model(model: ModelRef) -> None:
    """Validate the base model repo id and its 40-hex commit pin."""
    _check_choice("base_model.repo", model.repo, None, HF_REPO_RE)
    if not isinstance(model.revision, str) or not COMMIT_RE.fullmatch(model.revision):
        raise JobSpecError(
            "base_model.revision", "must be a full 40-hex lowercase commit sha (no branches/tags)"
        )


def _check_dataset(dataset: DatasetRef) -> None:
    """Validate the dataset source, ref and the pin that source requires."""
    _check_choice("dataset.source", dataset.source, DATASET_SOURCES)
    if dataset.source == "hf":
        _check_choice("dataset.ref", dataset.ref, None, HF_REPO_RE)
        _require_pin("dataset.revision", dataset.revision, COMMIT_RE, "a 40-hex commit sha")
    elif dataset.source == "kaggle":
        _check_choice("dataset.ref", dataset.ref, None, KAGGLE_REF_RE)
        version = dataset.version
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise JobSpecError("dataset.version", "kaggle datasets need an integer version >= 1")
    elif dataset.source == "kaggle-redacted":
        _check_choice("dataset.ref", dataset.ref, None, KAGGLE_REF_RE)
        _require_pin("dataset.revision", dataset.revision, SHA256_RE, "its manifest revision")
    else:
        _check_choice("dataset.ref", dataset.ref, None, LOCAL_REF_RE)
        _require_pin("dataset.revision", dataset.revision, SHA256_RE, "the sha256 of the file")


def _require_pin(path: str, value: object, pattern: re.Pattern, what: str) -> None:
    """Raise unless ``value`` is a string fully matching ``pattern``."""
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise JobSpecError(path, f"must be {what}")


def _check_policy(policy: object, dataset: DatasetRef) -> None:
    """Validate data_policy; our own data is never public, and raw data pairs only with raw."""
    _check_choice("data_policy", policy, DATA_POLICIES)
    if dataset.source in REDACTED_SOURCES and policy != "redacted-private":
        raise JobSpecError("data_policy", f"{dataset.source} data requires redacted-private")
    if (dataset.source == "local-raw") != (policy == "raw-local-only"):
        raise JobSpecError("data_policy", "local-raw data requires raw-local-only, and only it")


def _check_hyperparams(recipe: str, hyperparams: object) -> None:
    """Check each hyperparameter is allowed for the recipe, well-typed and in bounds."""
    if not isinstance(hyperparams, Mapping):
        raise JobSpecError("hyperparams", "must be a mapping")
    bounds = RECIPES[recipe]
    missing = sorted(REQUIRED_HYPERPARAMS.get(recipe, frozenset()) - set(hyperparams))
    if missing:
        raise JobSpecError("hyperparams", f"recipe {recipe} requires {missing}")
    for key, value in hyperparams.items():
        path = f"hyperparams.{key}"
        if key not in bounds:
            raise JobSpecError(path, f"not allowed for recipe {recipe} (allowed: {sorted(bounds)})")
        _check_bound(path, value, bounds[key])


def _check_bound(path: str, value: object, bound: Bound) -> None:
    """Raise unless ``value`` has the bound's type and lies in [low, high]."""
    # bool is an int subclass; `epochs: true` is a typo, not 1.
    numeric = isinstance(value, int | float) and not isinstance(value, bool)
    if not numeric or (bound.kind is int and not isinstance(value, int)):
        raise JobSpecError(path, f"must be {bound.kind.__name__}, got {value!r}")
    if math.isnan(value) or not bound.low <= value <= bound.high:
        raise JobSpecError(path, f"{value!r} outside [{bound.low}, {bound.high}]")


def _check_hours(hours: object, backend: str) -> None:
    """Require a positive number of hours within the backend's cap."""
    if isinstance(hours, bool) or not isinstance(hours, int | float) or not hours > 0:
        raise JobSpecError("max_hours", f"must be a positive number, got {hours!r}")
    cap = MAX_HOURS[backend]
    if hours > cap:
        raise JobSpecError("max_hours", f"{hours} exceeds the {backend} cap of {cap} h")


def _section(data: Mapping, key: str, allowed: tuple[str, ...]) -> Mapping:
    """Return ``data[key]`` as a mapping with no keys outside ``allowed``."""
    value = data[key]
    if not isinstance(value, Mapping):
        raise JobSpecError(key, "must be a mapping")
    for extra in set(value) - set(allowed):
        raise JobSpecError(f"{key}.{extra}", "unknown key")
    return value


def _check_top_level(data: object) -> Mapping:
    """Require a mapping with every required key and no unknown ones."""
    if not isinstance(data, Mapping):
        raise JobSpecError("<root>", "job spec must be a YAML mapping")
    for key in data:
        if key not in TOP_LEVEL_KEYS:
            raise JobSpecError(str(key), "unknown key")
    for key in TOP_LEVEL_KEYS:
        if key not in data and key not in OPTIONAL_TOP_LEVEL:
            raise JobSpecError(key, "required")
    return data


def from_dict(data: object) -> JobSpec:
    """Build and validate a ``JobSpec`` from parsed YAML.

    Args:
        data: the parsed document (must be a mapping). Not mutated or aliased.

    Returns:
        A frozen, validated ``JobSpec`` (``max_hours`` normalised to float).

    Raises:
        JobSpecError: naming the first offending field.
    """
    data = _check_top_level(data)
    model = _section(data, "base_model", ("repo", "revision"))
    dataset = _section(data, "dataset", ("source", "ref", "revision", "version"))
    evaluation = _section(data, "eval", ("task", "metric"))
    hyperparams = data.get("hyperparams", {})
    if not isinstance(hyperparams, Mapping):
        raise JobSpecError("hyperparams", "must be a mapping")
    spec = JobSpec(
        name=data["name"],
        recipe=data["recipe"],
        backend=data["backend"],
        base_model=ModelRef(repo=model.get("repo"), revision=model.get("revision")),
        dataset=DatasetRef(
            source=dataset.get("source"),
            ref=dataset.get("ref"),
            revision=dataset.get("revision"),
            version=dataset.get("version"),
        ),
        data_policy=data["data_policy"],
        max_hours=data["max_hours"],
        eval=EvalSpec(task=evaluation.get("task"), metric=evaluation.get("metric")),
        hyperparams=MappingProxyType(dict(hyperparams)),
    )
    spec.validate()
    return dataclasses.replace(spec, max_hours=float(spec.max_hours))


def load(path: str | Path) -> JobSpec:
    """Read a YAML job spec from disk and validate it.

    Args:
        path: the ``.yaml`` file.

    Returns:
        The validated ``JobSpec``.

    Raises:
        JobSpecError: if the file is unreadable, not YAML, or fails validation.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except OSError as exc:
        raise JobSpecError("<root>", f"cannot read {path}: {exc.strerror}") from None
    except yaml.YAMLError as exc:
        raise JobSpecError("<root>", f"invalid YAML in {path}: {exc}") from None
    return from_dict(data)
