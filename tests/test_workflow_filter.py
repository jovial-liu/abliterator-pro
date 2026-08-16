from __future__ import annotations

import json

import pytest

from abliterator.datasets import PromptRecord
from abliterator.errors import ConfigurationError, DatasetError
from abliterator.workflow import filter_direction_records_by_behavior


def _record(identifier: str, behavior: str) -> PromptRecord:
    return PromptRecord(
        id=identifier,
        prompt=f"secret prompt {identifier}",
        cohort="cyber-unsafe" if behavior == "deny" else "cyber-benign",
        expected_behavior=behavior,
        domain="cybersecurity",
        category=None,
        source_id="unit-source",
        source_revision="abc123",
        source_row_id=identifier,
        group_id=identifier,
        license_spdx="MIT",
        commercial_ok=True,
        gated=False,
        use_for_direction=True,
        use_for_eval=True,
    )


class _ResponseAdapter:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], int, int]] = []

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        max_length: int,
    ) -> list[str]:
        self.calls.append((prompts, max_new_tokens, max_length))
        return [self.responses[prompt] for prompt in prompts]


def test_behavior_filter_selects_actual_refusals_and_actual_answers() -> None:
    targets = [_record(f"target-{index}", "deny") for index in range(4)]
    contrasts = [_record(f"contrast-{index}", "allow") for index in range(4)]
    responses = {
        targets[0].prompt: "I can't assist with that request.",
        targets[1].prompt: "Here are the requested steps.",
        targets[2].prompt: "I'm sorry, but I cannot help with that.",
        targets[3].prompt: "I won't provide those instructions.",
        contrasts[0].prompt: "I can't help with that.",
        contrasts[1].prompt: "Use least privilege.",
        contrasts[2].prompt: "Rotate the credential.",
        contrasts[3].prompt: "Patch the service.",
    }
    adapter = _ResponseAdapter(responses)

    selected_targets, selected_contrasts, receipt = filter_direction_records_by_behavior(
        adapter,
        targets,
        contrasts,
        desired_count=2,
        batch_size=2,
        max_new_tokens=17,
        max_length=96,
    )

    assert [record.id for record in selected_targets] == ["target-0", "target-2"]
    assert [record.id for record in selected_contrasts] == ["contrast-1", "contrast-2"]
    assert receipt["format"] == "abpro-direction-behavior-filter-v1"
    assert receipt["desired_count"] == 2
    assert receipt["selection_strategy"] == "first-matching-candidate-in-input-order"
    assert receipt["target"]["candidate_count"] == 4
    assert receipt["target"]["matching_count"] == 3
    assert receipt["target"]["selected_count"] == 2
    assert receipt["contrast"]["candidate_count"] == 4
    assert receipt["contrast"]["matching_count"] == 3
    assert receipt["contrast"]["selected_count"] == 2
    assert [row["prompt_id"] for row in receipt["target"]["selected"]] == [
        "target-0",
        "target-2",
    ]
    assert all(row["refused"] is True for row in receipt["target"]["selected"])
    assert all(row["refused"] is False for row in receipt["contrast"]["selected"])
    assert all(len(row["prompt_sha256"]) == 64 for row in receipt["target"]["selected"])
    assert all(len(row["response_sha256"]) == 64 for row in receipt["contrast"]["selected"])
    assert len(adapter.calls) == 4
    assert all(call[1:] == (17, 96) for call in adapter.calls)


def test_behavior_filter_receipt_contains_no_prompt_or_response_text() -> None:
    target = _record("target-secret-marker", "deny")
    contrast = _record("contrast-secret-marker", "allow")
    adapter = _ResponseAdapter(
        {
            target.prompt: "I cannot assist with response-secret-marker.",
            contrast.prompt: "response-secret-marker benign answer",
        }
    )

    _, _, receipt = filter_direction_records_by_behavior(
        adapter,
        [target],
        [contrast],
        desired_count=1,
    )

    serialized = json.dumps(receipt)
    assert target.prompt not in serialized
    assert contrast.prompt not in serialized
    assert "response-secret-marker" not in serialized
    assert "prompt" not in receipt["target"]["selected"][0]
    assert "response" not in receipt["target"]["selected"][0]


def test_behavior_filter_fails_when_either_behavior_pool_is_insufficient() -> None:
    targets = [_record("target-0", "deny"), _record("target-1", "deny")]
    contrasts = [_record("contrast-0", "allow"), _record("contrast-1", "allow")]
    adapter = _ResponseAdapter(
        {
            targets[0].prompt: "I can't assist with that.",
            targets[1].prompt: "Here are the steps.",
            contrasts[0].prompt: "I can't help with that.",
            contrasts[1].prompt: "A benign answer.",
        }
    )

    with pytest.raises(
        DatasetError,
        match=r"required 2.*target refusals=1/2.*contrast answers=1/2",
    ):
        filter_direction_records_by_behavior(
            adapter,
            targets,
            contrasts,
            desired_count=2,
        )


@pytest.mark.parametrize("desired_count", [0, -1])
def test_behavior_filter_rejects_nonpositive_desired_count(desired_count: int) -> None:
    with pytest.raises(ConfigurationError, match="desired_count"):
        filter_direction_records_by_behavior(
            _ResponseAdapter({}),
            [],
            [],
            desired_count=desired_count,
        )


def test_behavior_filter_rejects_mislabeled_cohorts_before_generation() -> None:
    adapter = _ResponseAdapter({})
    with pytest.raises(ConfigurationError, match="expected_behavior='deny'"):
        filter_direction_records_by_behavior(
            adapter,
            [_record("wrong-target", "allow")],
            [_record("contrast", "allow")],
            desired_count=1,
        )
    assert adapter.calls == []
