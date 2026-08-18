from __future__ import annotations

import io
import json

from abliterator.progress import (
    JsonProgressRenderer,
    PlainProgressRenderer,
    WorkflowEvent,
    WorkflowProgress,
)


def _plan() -> dict[str, object]:
    return {
        "source_model": "/models/base",
        "output": "/models/base-obliterated",
        "objective": "guarded",
        "direction_source": "discover",
        "direction_samples": 40,
        "candidate_layers": "auto",
        "candidate_count": 3,
        "apply_layers": "all",
        "strength": 1.0,
        "kinds": ["attention", "mlp"],
        "truthfulness_samples_per_dataset": 20,
    }


def test_workflow_progress_emits_stable_lifecycle_events() -> None:
    events: list[WorkflowEvent] = []
    progress = WorkflowProgress(events.append)

    progress.start("preflight", "Checking")
    progress.complete("preflight", "Checked", details={"output": "/tmp/fork"})

    assert [(event.stage, event.status) for event in events] == [
        ("preflight", "started"),
        ("preflight", "completed"),
    ]
    assert events[0].index == 1
    assert events[0].total == 10
    assert events[1].details == {"output": "/tmp/fork"}
    assert progress.timings["preflight"] >= 0


def test_progress_callback_failure_does_not_break_workflow() -> None:
    def broken_callback(_: WorkflowEvent) -> None:
        raise OSError("closed stream")

    progress = WorkflowProgress(broken_callback)
    progress.start("preflight", "Checking")
    progress.complete("preflight", "Checked")


def test_plain_progress_renderer_shows_plan_stage_and_summary() -> None:
    stream = io.StringIO()
    renderer = PlainProgressRenderer(stream)
    renderer.show_plan(_plan(), dry_run=True)
    renderer(
        WorkflowEvent(
            stage="preflight",
            status="completed",
            index=1,
            total=10,
            message="Plan validated",
            elapsed_seconds=1.25,
        )
    )
    renderer.show_result(
        {
            "output": "/models/base-obliterated",
            "selected_direction": "layer.12.resid_pre",
            "edited_targets": 72,
            "reload_verified": True,
            "receipt": "/models/base-obliterated/abpro/run.json",
        }
    )

    rendered = stream.getvalue()
    assert "AB Pro · model fork workflow" in rendered
    assert "40 target + 40 contrast prompts" in rendered
    assert "plan only; no model will be loaded or written" in rendered
    assert "[01/10] done  Plan validated (1.2 s)" in rendered
    assert "AB Pro · fork complete" in rendered
    assert "72 projection targets edited" in rendered


def test_json_progress_renderer_emits_json_lines() -> None:
    stream = io.StringIO()
    renderer = JsonProgressRenderer(stream)
    renderer.show_plan(_plan())
    renderer(
        WorkflowEvent(
            stage="source",
            status="started",
            index=3,
            total=10,
            message="Loading source model",
        )
    )

    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0]["type"] == "plan"
    assert rows[1]["type"] == "progress"
    assert rows[1]["stage"] == "source"
    assert rows[1]["status"] == "started"
