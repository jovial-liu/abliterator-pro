from __future__ import annotations

import pytest

from abliterator.datasets import PromptRecord
from abliterator.errors import ConfigurationError
from abliterator.evaluation import RefusalDetector, compare_reports, evaluate_prompts


def test_refusal_detector_matches_clear_refusal() -> None:
    detector = RefusalDetector()
    assert detector.is_refusal("I'm sorry, but I can't assist with that request.")


def test_refusal_detector_does_not_match_normal_answer() -> None:
    detector = RefusalDetector()
    assert not detector.is_refusal("Use least privilege and rotate the credentials after the test.")


class ChoiceAdapter:
    def generate(self, prompts, max_new_tokens, max_length):
        return ["B" for _ in prompts]


class ThinkingChoiceAdapter:
    def generate(self, prompts, max_new_tokens, max_length):
        return ["<think>I will reason privately here.</think>\nB" for _ in prompts]


def test_multiple_choice_capability_accuracy() -> None:
    record = PromptRecord(
        id="choice",
        prompt="Question\nA. first\nB. second",
        cohort="capability",
        expected_behavior="evaluate",
        domain="cybersecurity",
        category=None,
        source_id="unit",
        source_revision="abc",
        source_row_id="1",
        group_id="1",
        license_spdx="MIT",
        commercial_ok=True,
        gated=False,
        use_for_direction=False,
        use_for_eval=True,
        reference_answer="B",
    )
    report = evaluate_prompts(ChoiceAdapter(), [record])
    assert report.capability_total == 1
    assert report.capability_accuracy == 1.0
    assert report.unscored_total == 0


def test_multiple_choice_parses_final_answer_after_thinking() -> None:
    report = evaluate_prompts(ThinkingChoiceAdapter(), [choice_record("thinking")])
    assert report.capability_accuracy == 1.0


def test_evaluation_validates_batch_size() -> None:
    with pytest.raises(ConfigurationError, match="batch_size"):
        evaluate_prompts(ChoiceAdapter(), [choice_record("one")], batch_size=0)


def test_compare_reports_rejects_different_cohorts() -> None:
    first = evaluate_prompts(ChoiceAdapter(), [choice_record("first")])
    second = evaluate_prompts(ChoiceAdapter(), [choice_record("second")])
    with pytest.raises(ConfigurationError, match="different prompt cohorts"):
        compare_reports(first, second)


def choice_record(identifier: str) -> PromptRecord:
    return PromptRecord(
        id=identifier,
        prompt="Question\nA. first\nB. second",
        cohort="capability",
        expected_behavior="evaluate",
        domain="cybersecurity",
        category=None,
        source_id="unit",
        source_revision="abc",
        source_row_id=identifier,
        group_id=identifier,
        license_spdx="MIT",
        commercial_ok=True,
        gated=False,
        use_for_direction=False,
        use_for_eval=True,
        reference_answer="B",
    )
