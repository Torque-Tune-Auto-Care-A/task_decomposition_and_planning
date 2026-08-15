from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from ..models import PlanningRequest, Thought


class ThoughtCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[str] = Field(min_length=1, max_length=3)


class ThoughtEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def tree_of_thoughts(
    problem: str,
    llm: BaseChatModel,
    depth: int = 2,
    beam_width: int = 2,
) -> list[Thought]:
    frontier = [Thought(state="Start", score=0.5, rationale="root")]
    for _ in range(depth):
        candidates: list[Thought] = []
        for parent in frontier:
            generated = llm.with_structured_output(
                ThoughtCandidates,
                method="json_schema",
            ).invoke([
                ("system", "Generate distinct candidate next steps for Tree-of-Thoughts search."),
                ("human", f"""Problem: {problem}
Partial path: {parent.state}
Propose two distinct promising continuations."""),
            ], temperature=0.5)
            for state in generated.candidates[:2]:
                judged = llm.with_structured_output(
                    ThoughtEvaluation,
                    method="json_schema",
                ).invoke([
                    ("system", "Independently evaluate a partial solution."),
                    ("human", f"""Problem: {problem}
Candidate path: {state}
Score correctness, feasibility, and progress. Do not reward confident wording."""),
                ], temperature=0.1)
                candidates.append(
                    Thought(state=state, score=judged.score, rationale=judged.rationale)
                )
        frontier = sorted(candidates, key=lambda item: item.score, reverse=True)[:beam_width]
        if not frontier:
            break
    return frontier


# ---------------------------------------------------------------------------
# Torque-Tune domain adapter — single source per concern.
# ---------------------------------------------------------------------------

def build_tot_question(planning_request: PlanningRequest) -> str:
    return f"""
You are evaluating a high-risk vehicle tuning job for Torque-Tune-Auto-Care.

Customer request:
{planning_request.request}

Client ID: {planning_request.client_id}
Vehicle ID: {planning_request.vehicle_id}
Technician ID: {planning_request.tech_id}
Appointment ID: {planning_request.appointment_id}

Your task is to compare possible decisions for this planning request.

Possible decisions:

1. RELEASE
   Sufficient verified evidence exists to safely proceed.

2. HOLD
   Required evidence is missing, so the job cannot currently be released.

3. ESCALATE
   Shift-lead review or customer confirmation is required.

Use Tree-of-Thoughts reasoning to consider multiple possible
decision paths before selecting the most appropriate candidate.

For every candidate reasoning path, consider:

- Client and vehicle identity.
- Previous tuning history.
- Appointment validity.
- Technician identity, authentication, and authorization.
- Compliance and warranty requirements.
- Required customer disclosure and sign-off.
- Whether the requested modification is emissions-affecting.
- Whether any consequential database write is permitted.

The candidate decision must be grounded in available evidence.

Do not invent evidence.
Do not assume that an MCP or database check succeeded.
Do not perform any consequential action.

Prefer a candidate that correctly identifies whether the available
evidence supports RELEASE, HOLD, or ESCALATE.
"""


def evaluate_decisions(
    planning_request: PlanningRequest,
    llm: BaseChatModel,
    depth: int = 2,
    beam_width: int = 2,
) -> list[Thought]:
    """Domain-adapted entry point: decision comparison via beam search."""
    return tree_of_thoughts(
        build_tot_question(planning_request),
        llm,
        depth=depth,
        beam_width=beam_width,
    )