"""Small, dependency-free progress UI for AB Pro workflows."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Literal, TextIO

ProgressStatus = Literal["started", "completed", "skipped", "failed"]

WORKFLOW_STAGES: tuple[str, ...] = (
    "preflight",
    "cohorts",
    "source",
    "directions",
    "ranking",
    "baseline-quality",
    "edit",
    "reload",
    "validation",
    "publish",
)


@dataclass(frozen=True)
class WorkflowEvent:
    """One stable, machine-readable workflow lifecycle event."""

    stage: str
    status: ProgressStatus
    index: int
    total: int
    message: str
    elapsed_seconds: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[WorkflowEvent], None]


class WorkflowProgress:
    """Track workflow stages and isolate callbacks from model execution."""

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self.callback = callback
        self._started: dict[str, float] = {}
        self._timings: dict[str, float] = {}
        self.current_stage: str | None = None

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)

    def start(
        self,
        stage: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._validate_stage(stage)
        self.current_stage = stage
        self._started[stage] = monotonic()
        self._emit(stage, "started", message, details=details)

    def complete(
        self,
        stage: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._validate_stage(stage)
        started = self._started.get(stage)
        elapsed = None if started is None else monotonic() - started
        if elapsed is not None:
            self._timings[stage] = elapsed
        self._emit(stage, "completed", message, elapsed=elapsed, details=details)
        if self.current_stage == stage:
            self.current_stage = None

    def skip(
        self,
        stage: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._validate_stage(stage)
        self._emit(stage, "skipped", message, details=details)
        if self.current_stage == stage:
            self.current_stage = None

    def fail(self, message: str) -> None:
        stage = self.current_stage or "preflight"
        started = self._started.get(stage)
        elapsed = None if started is None else monotonic() - started
        self._emit(stage, "failed", message, elapsed=elapsed)
        self.current_stage = None

    def _validate_stage(self, stage: str) -> None:
        if stage not in WORKFLOW_STAGES:
            raise ValueError(f"Unknown workflow progress stage {stage!r}")

    def _emit(
        self,
        stage: str,
        status: ProgressStatus,
        message: str,
        *,
        elapsed: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.callback is None:
            return
        event = WorkflowEvent(
            stage=stage,
            status=status,
            index=WORKFLOW_STAGES.index(stage) + 1,
            total=len(WORKFLOW_STAGES),
            message=message,
            elapsed_seconds=None if elapsed is None else round(elapsed, 3),
            details={} if details is None else details,
        )
        try:
            self.callback(event)
        except Exception:
            # Display or telemetry failures must never corrupt a model workflow.
            return


class PlainProgressRenderer:
    """Render concise terminal progress while keeping stdout machine-readable."""

    _MARKERS = {
        "started": "...",
        "completed": "done",
        "skipped": "skip",
        "failed": "FAIL",
    }

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def show_plan(self, plan: dict[str, Any], *, dry_run: bool = False) -> None:
        direction_source = plan["direction_source"]
        if direction_source == "discover":
            direction_source = (
                f"discover from {plan['direction_samples']} target + "
                f"{plan['direction_samples']} contrast prompts"
            )
        candidates = plan["candidate_layers"]
        if candidates == "auto":
            candidates = f"{plan['candidate_count']} automatic middle-layer candidates"
        print("AB Pro · model fork workflow", file=self.stream)
        print(f"  Source:     {plan['source_model']}", file=self.stream)
        print(f"  Output:     {plan['output']}", file=self.stream)
        print(f"  Objective:  {plan['objective']}", file=self.stream)
        print(f"  Directions: {direction_source}", file=self.stream)
        print(f"  Candidates: {candidates}", file=self.stream)
        print(
            f"  Edit:       layers={plan['apply_layers']}; strength={plan['strength']}; "
            f"targets={','.join(plan['kinds'])}",
            file=self.stream,
        )
        checks = "reload + behavior replay"
        if plan["truthfulness_samples_per_dataset"]:
            checks += " + truthfulness regression"
        print(f"  Checks:     {checks}", file=self.stream)
        if dry_run:
            print("  Dry run:    plan only; no model will be loaded or written", file=self.stream)
        print(file=self.stream)

    def __call__(self, event: WorkflowEvent) -> None:
        marker = self._MARKERS[event.status]
        elapsed = (
            "" if event.elapsed_seconds is None else f" ({_format_duration(event.elapsed_seconds)})"
        )
        print(
            f"[{event.index:02d}/{event.total:02d}] {marker:>4}  {event.message}{elapsed}",
            file=self.stream,
            flush=True,
        )

    def show_result(self, result: dict[str, Any]) -> None:
        print(file=self.stream)
        print("AB Pro · fork complete", file=self.stream)
        print(f"  Model:      {result['output']}", file=self.stream)
        print(f"  Direction:  {result['selected_direction']}", file=self.stream)
        print(
            f"  Weights:    {result['edited_targets']} projection targets edited", file=self.stream
        )
        print(f"  Verified:   {'yes' if result['reload_verified'] else 'no'}", file=self.stream)
        print(f"  Receipt:    {result['receipt']}", file=self.stream, flush=True)


class JsonProgressRenderer:
    """Render newline-delimited JSON progress for application integrations."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def show_plan(self, plan: dict[str, Any], *, dry_run: bool = False) -> None:
        self._write({"type": "plan", "dry_run": dry_run, "plan": plan})

    def __call__(self, event: WorkflowEvent) -> None:
        self._write({"type": "progress", **event.to_dict()})

    def show_result(self, result: dict[str, Any]) -> None:
        self._write({"type": "result", "result": result})

    def _write(self, value: dict[str, Any]) -> None:
        print(json.dumps(value, sort_keys=True, default=str), file=self.stream, flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder:02d}s"
