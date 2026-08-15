from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from ..models import PlanningRequest


def plan_and_solve(question: str, llm: BaseChatModel) -> str:
    """Generic Plan-and-Solve prompting (toolkit base, unchanged)."""
    response = llm.invoke([
        ("system", "You use Plan-and-Solve prompting. Clearly separate PLAN from SOLUTION."),
        ("human", f"""{question}

First understand the problem and devise a plan to solve it. Then carry out the
plan step by step. Check calculations and common-sense assumptions."""),
    ], temperature=0.2)
    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    return response.content.strip()


def build_planning_question(planning_request: PlanningRequest) -> str:
    return f"""
You are planning a high-risk vehicle tuning job for Torque-Tune-Auto-Care.

Customer request:
{planning_request.request}

Client ID: {planning_request.client_id}
Vehicle ID: {planning_request.vehicle_id}
Technician ID: {planning_request.tech_id}
Appointment ID: {planning_request.appointment_id}

Your task is to create a complete verification plan before any
consequential database write is performed.

The plan must verify:

1. Client and vehicle identity.
2. Previous tuning history.
3. Appointment validity.
4. Technician identity and authorization.
5. Compliance and warranty requirements.
6. Required customer disclosure and sign-off.

The planning process must eventually support one of these decisions:

- RELEASE: sufficient verified evidence exists to safely proceed.
- HOLD: required evidence is missing.
- ESCALATE: shift-lead review or customer confirmation is required.

Do not invent evidence.
Do not assume that a check passed.
Do not perform any consequential action.

Clearly separate the PLAN from the SOLUTION.

The plan should:
- List the verification steps in a logical order.
- State what evidence is required for each step.
- Identify information that must come from MCP tools or the database.
- Explain how the verified evidence supports the final RELEASE,
  HOLD, or ESCALATE decision.

Do not claim that any verification step has passed unless
actual evidence is provided.
"""


def plan_job(planning_request: PlanningRequest, llm: BaseChatModel) -> str:
    """Domain-adapted entry point: Torque-Tune verification planning."""
    return plan_and_solve(build_planning_question(planning_request), llm)