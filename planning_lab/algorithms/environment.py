from __future__ import annotations

from collections.abc import Callable

from ..models import EnvironmentFeedback

CandidateValidator = Callable[[str], EnvironmentFeedback]


class Environment:
    """
    Generic adapter for real external feedback.

    The Torque-Tune project supplies the validator in `planning/torque_tune_environment.py`. It will validate a
    candidate plan using MCP results, SQLite evidence, and compliance rules.
    """

    def __init__(self, validator: CandidateValidator) -> None:
        self._validator = validator

    def evaluate(self, candidate_plan: str) -> EnvironmentFeedback:
        return self._validator(candidate_plan)