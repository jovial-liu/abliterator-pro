from __future__ import annotations

from dataclasses import replace

from abliterator.datasets import (
    PromptRecord,
    deduplicate_records,
    length_match,
    registry,
    stable_partition,
)


def record(identifier: str, prompt: str, group: str | None = None) -> PromptRecord:
    return PromptRecord(
        id=identifier,
        prompt=prompt,
        cohort="test",
        expected_behavior="allow",
        domain="cybersecurity",
        category=None,
        source_id="unit",
        source_revision="abc",
        source_row_id=identifier,
        group_id=group or identifier,
        license_spdx="MIT",
        commercial_ok=True,
        gated=False,
        use_for_direction=True,
        use_for_eval=False,
    )


def test_registry_sources_are_pinned_and_role_separated() -> None:
    specs = registry()
    assert "cyberseceval-mitre" in specs
    assert "cyberseceval-frr" in specs
    assert specs["cyberseceval-mitre"].sha256
    assert specs["cyberseceval-frr"].sha256
    assert specs["wmdp-cyber"].use_for_eval
    assert not specs["wmdp-cyber"].use_for_direction
    assert specs["wmdp-cyber"].choices_field == "choices"
    assert specs["wmdp-cyber"].answer_field == "answer"
    assert not specs["ifeval"].use_for_eval
    assert not specs["wildguard-test"].use_for_eval
    assert not specs["wildguard-test"].commercial_ok
    assert all(spec.revision for spec in specs.values())


def test_deduplication_normalizes_case_and_whitespace() -> None:
    records = [record("a", "Explain  TLS"), record("b", " explain tls ")]
    assert [item.id for item in deduplicate_records(records)] == ["a"]


def test_stable_partition_keeps_groups_together() -> None:
    records = [record("a", "one", "shared"), record("b", "two", "shared")]
    train, validation = stable_partition(records, validation_fraction=0.5)
    assert not (train and validation)


def test_length_matching_balances_bins() -> None:
    short = record("a", "a" * 100)
    long = record("b", "b" * 300)
    contrast = [replace(short, id="c", prompt="c" * 110), replace(long, id="d", prompt="d" * 320)]
    left, right = length_match([short, long], contrast, bin_width=128)
    assert len(left) == len(right) == 2
    assert [len(item.prompt) // 128 for item in left] == [len(item.prompt) // 128 for item in right]
