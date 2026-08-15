"""Pinned, provenance-preserving prompt dataset loading."""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, cast

from .errors import DatasetError
from .runtime import require


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    description: str
    loader: str
    text_field: str
    cohort: str
    expected_behavior: str
    license_spdx: str
    commercial_ok: bool
    gated: bool
    use_for_direction: bool
    use_for_eval: bool
    use_for_ranking: bool = False
    revision: str | None = None
    url: str | None = None
    sha256: str | None = None
    repo: str | None = None
    config: str | None = None
    split: str | None = None
    category_field: str | None = None
    filters: dict[str, Any] | None = None
    choices_field: str | None = None
    answer_field: str | None = None


@dataclass(frozen=True)
class PromptRecord:
    id: str
    prompt: str
    cohort: str
    expected_behavior: str
    domain: str
    category: str | None
    source_id: str
    source_revision: str | None
    source_row_id: str
    group_id: str
    license_spdx: str
    commercial_ok: bool
    gated: bool
    use_for_direction: bool
    use_for_eval: bool
    reference_answer: str | None = None

    def public_metadata(self) -> dict[str, Any]:
        """Metadata safe to log; deliberately excludes prompt text."""
        value = asdict(self)
        value.pop("prompt")
        value["prompt_sha256"] = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        return value


def registry() -> dict[str, DatasetSpec]:
    data_path = resources.files("abliterator.data").joinpath("datasets.json")
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    specs = [DatasetSpec(**value) for value in raw]
    return {spec.id: spec for spec in specs}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return registry()[dataset_id]
    except KeyError as exc:
        raise DatasetError(
            f"Unknown dataset {dataset_id!r}; available: {', '.join(sorted(registry()))}"
        ) from exc


def load_dataset_records(
    dataset_id: str,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[PromptRecord]:
    spec = get_dataset_spec(dataset_id)
    if limit is not None and limit < 1:
        raise DatasetError("limit must be positive")
    rows: Iterable[dict[str, Any]]
    if spec.loader == "json-url":
        rows = _load_verified_url(spec)
    elif spec.loader == "huggingface":
        rows = _load_huggingface(spec, seed=seed)
    else:
        raise DatasetError(f"Unsupported loader {spec.loader!r} for {dataset_id}")
    records: list[PromptRecord] = []
    seen: set[str] = set()
    stop_at_limit = spec.loader == "huggingface"
    try:
        for record in _normalize_rows(spec, rows):
            digest = normalized_prompt_hash(record.prompt)
            if digest in seen:
                continue
            seen.add(digest)
            records.append(record)
            if stop_at_limit and limit is not None and len(records) >= limit:
                break
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Could not stream pinned dataset {spec.id}") from exc
    rng = random.Random(seed)
    rng.shuffle(records)
    return records if limit is None else records[:limit]


def _load_verified_url(spec: DatasetSpec) -> list[dict[str, Any]]:
    if not spec.url or not spec.sha256:
        raise DatasetError(f"{spec.id} must register both url and sha256")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "abliterator-pro/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except OSError as exc:
        raise DatasetError(f"Could not download pinned dataset {spec.id}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != spec.sha256:
        raise DatasetError(
            f"Checksum mismatch for {spec.id}: expected {spec.sha256}, received {actual}"
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Pinned dataset {spec.id} is not valid JSON") from exc
    if not isinstance(value, list):
        raise DatasetError(f"Pinned dataset {spec.id} must contain a JSON array")
    return value


def _load_huggingface(spec: DatasetSpec, seed: int) -> Iterable[dict[str, Any]]:
    if not spec.repo or not spec.split:
        raise DatasetError(f"{spec.id} must register repo and split")
    datasets = require("datasets")
    try:
        dataset = datasets.load_dataset(
            spec.repo,
            spec.config,
            split=spec.split,
            revision=spec.revision,
            streaming=True,
        )
        # A bounded streaming shuffle gives different seeded cohorts without consuming an
        # unbounded source. Row IDs are content-derived later, so buffer order cannot affect
        # provenance.
        return cast(Iterable[dict[str, Any]], dataset.shuffle(seed=seed, buffer_size=10_000))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise DatasetError(f"Could not load pinned Hugging Face dataset {spec.id}") from exc


def _normalize_rows(spec: DatasetSpec, rows: Iterable[dict[str, Any]]) -> Iterator[PromptRecord]:
    for index, row in enumerate(rows):
        if spec.filters and any(row.get(key) != value for key, value in spec.filters.items()):
            continue
        prompt = row.get(spec.text_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetError(
                f"{spec.id} row {index} lacks non-empty text field {spec.text_field!r}"
            )
        reference_answer: str | None = None
        if spec.choices_field and spec.answer_field:
            choices = row.get(spec.choices_field)
            answer = row.get(spec.answer_field)
            if (
                not isinstance(choices, (list, tuple))
                or not choices
                or not isinstance(answer, int)
                or answer < 0
                or answer >= len(choices)
                or len(choices) > 26
            ):
                raise DatasetError(f"{spec.id} row {index} has invalid multiple-choice fields")
            labels = [chr(ord("A") + choice_index) for choice_index in range(len(choices))]
            rendered_choices = "\n".join(
                f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
            )
            prompt = f"{prompt.strip()}\n\n{rendered_choices}\n\nAnswer with only the letter."
            reference_answer = labels[answer]
        upstream_row_id = row.get("id", row.get("Index", row.get("BehaviorID")))
        row_id = (
            str(upstream_row_id)
            if upstream_row_id is not None
            else hashlib.sha256(
                json.dumps(row, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        category = row.get(spec.category_field) if spec.category_field else None
        stable = f"{spec.id}:{spec.revision}:{row_id}"
        yield PromptRecord(
            id=hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20],
            prompt=prompt.strip(),
            cohort=spec.cohort,
            expected_behavior=spec.expected_behavior,
            domain="cybersecurity" if "cyber" in spec.cohort else "general",
            category=None if category is None else str(category),
            source_id=spec.id,
            source_revision=spec.revision,
            source_row_id=row_id,
            group_id=str(row.get("group_id", row_id)),
            license_spdx=spec.license_spdx,
            commercial_ok=spec.commercial_ok,
            gated=spec.gated,
            use_for_direction=spec.use_for_direction,
            use_for_eval=spec.use_for_eval,
            reference_answer=reference_answer,
        )


def load_local_jsonl(path: str | Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(PromptRecord(**value))
            except (json.JSONDecodeError, TypeError) as exc:
                raise DatasetError(f"Invalid prompt record at {path}:{line_number}") from exc
    return deduplicate_records(records)


def normalized_prompt_hash(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.strip().casefold())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_records(records: Iterable[PromptRecord]) -> list[PromptRecord]:
    output: list[PromptRecord] = []
    seen: set[str] = set()
    for record in records:
        digest = normalized_prompt_hash(record.prompt)
        if digest in seen:
            continue
        seen.add(digest)
        output.append(record)
    return output


def stable_partition(
    records: Iterable[PromptRecord],
    validation_fraction: float = 0.2,
) -> tuple[list[PromptRecord], list[PromptRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise DatasetError("validation_fraction must be between zero and one")
    train: list[PromptRecord] = []
    validation: list[PromptRecord] = []
    threshold = int(validation_fraction * 10_000)
    for record in records:
        group_key = f"{record.source_id}:{record.source_revision}:{record.group_id}"
        bucket = int(hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:8], 16) % 10_000
        (validation if bucket < threshold else train).append(record)
    return train, validation


def length_match(
    target: Iterable[PromptRecord],
    contrast: Iterable[PromptRecord],
    *,
    seed: int = 42,
    bin_width: int = 128,
) -> tuple[list[PromptRecord], list[PromptRecord]]:
    """Match cohorts by coarse character length to reduce style-direction leakage."""
    if bin_width < 16:
        raise DatasetError("bin_width must be at least 16 characters")
    rng = random.Random(seed)
    target_bins: dict[int, list[PromptRecord]] = {}
    contrast_bins: dict[int, list[PromptRecord]] = {}
    for record in target:
        target_bins.setdefault(len(record.prompt) // bin_width, []).append(record)
    for record in contrast:
        contrast_bins.setdefault(len(record.prompt) // bin_width, []).append(record)
    matched_target: list[PromptRecord] = []
    matched_contrast: list[PromptRecord] = []
    for bucket in sorted(set(target_bins) & set(contrast_bins)):
        left = target_bins[bucket]
        right = contrast_bins[bucket]
        rng.shuffle(left)
        rng.shuffle(right)
        count = min(len(left), len(right))
        matched_target.extend(left[:count])
        matched_contrast.extend(right[:count])
    if not matched_target:
        raise DatasetError("No examples remained after length matching")
    pairs = list(zip(matched_target, matched_contrast, strict=True))
    rng.shuffle(pairs)
    shuffled_target, shuffled_contrast = zip(*pairs, strict=True)
    return list(shuffled_target), list(shuffled_contrast)
