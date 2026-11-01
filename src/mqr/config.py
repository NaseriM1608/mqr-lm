"""Strict, serializable configuration for the MQR-v1 research implementation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


def _integer(name: str, value: Any, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: Any, minimum: float, *, inclusive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        comparator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {comparator} {minimum}")


def _strict_section(cls: type, values: Mapping[str, Any], name: str) -> Any:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(values) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(map(str, unknown)))}")
    values = dict(values)
    if cls is TrainConfig and "betas" in values:
        if not isinstance(values["betas"], (list, tuple)):
            raise ValueError("training.betas must contain two numbers")
        values["betas"] = tuple(values["betas"])
    return cls(**values)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    layers: int = 6
    d_model: int = 256
    attention_heads: int = 4
    ffn_dim: int = 1024
    context_length: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("layers", "d_model", "attention_heads", "ffn_dim", "context_length"):
            _integer(f"model.{name}", getattr(self, name))
        _integer("model.vocab_size", self.vocab_size, 2)
        if self.d_model % self.attention_heads:
            raise ValueError("model.d_model must be divisible by model.attention_heads")
        if (self.d_model // self.attention_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        _number("model.dropout", self.dropout, 0.0)
        if self.dropout >= 1.0:
            raise ValueError("model.dropout must be less than 1")


@dataclass(frozen=True)
class HeadConfig:
    kind: str = "mqr"
    d_e: int = 128
    d_c: int = 128
    num_queries: int = 4
    activation: str = "gelu"
    temperature: float = 1.0
    normalize_queries: bool = False
    normalize_output_rows: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.kind not in ("mqr", "direct", "dense"):
            raise ValueError("head.kind must be mqr, direct, or dense")
        for name in ("d_e", "d_c", "num_queries"):
            _integer(f"head.{name}", getattr(self, name))
        if self.activation != "gelu":
            raise ValueError("MQR-v1 supports head.activation='gelu'")
        _number("head.temperature", self.temperature, 0.0, inclusive=False)
        for name in ("normalize_queries", "normalize_output_rows"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"head.{name} must be boolean")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    device: str = "cpu"
    precision: str = "fp32"
    batch_size: int = 2
    gradient_accumulation: int = 1
    max_steps: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 5
    min_lr_ratio: float = 0.1
    loss_chunk_positions: int = 64
    checkpoint_every: int = 25
    eval_every: int = 25
    eval_batches: int = 4
    num_threads: int = 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "batch_size", "gradient_accumulation", "max_steps", "loss_chunk_positions",
            "eval_batches", "num_threads",
        ):
            _integer(f"training.{name}", getattr(self, name))
        for name in ("seed", "warmup_steps", "checkpoint_every", "eval_every"):
            _integer(f"training.{name}", getattr(self, name), 0)
        if not isinstance(self.device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device) is None:
            raise ValueError("training.device must be cpu, cuda, or cuda:<index>")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("training.precision must be fp32, bf16, or fp16")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("This implementation requires fp32 for CPU training")
        _number("training.learning_rate", self.learning_rate, 0.0, inclusive=False)
        _number("training.weight_decay", self.weight_decay, 0.0)
        _number("training.grad_clip", self.grad_clip, 0.0, inclusive=False)
        _number("training.min_lr_ratio", self.min_lr_ratio, 0.0)
        if self.min_lr_ratio > 1.0:
            raise ValueError("training.min_lr_ratio must be <= 1")
        if not isinstance(self.betas, tuple) or len(self.betas) != 2:
            raise ValueError("training.betas must be a tuple of two numbers")
        for value in self.betas:
            _number("training.betas", value, 0.0)
            if value >= 1.0:
                raise ValueError("training.betas values must be < 1")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, cls in (("model", ModelConfig), ("head", HeadConfig), ("training", TrainConfig)):
            value = getattr(self, name)
            if not isinstance(value, cls):
                raise ValueError(f"{name} must be a {cls.__name__}")
            value.validate()

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ExperimentConfig:
        if not isinstance(values, Mapping):
            raise ValueError("Experiment config must be a JSON object")
        unknown = set(values) - {"model", "head", "training"}
        if unknown:
            raise ValueError(f"Unknown experiment fields: {', '.join(sorted(map(str, unknown)))}")
        return cls(**{
            name: _strict_section(section, values.get(name, {}), name)
            for name, section in (("model", ModelConfig), ("head", HeadConfig), ("training", TrainConfig))
        })

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["training"]["betas"] = list(self.training.betas)
        return result
