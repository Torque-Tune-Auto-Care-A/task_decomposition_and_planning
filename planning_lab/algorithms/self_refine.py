from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from .environment import Environment


RELEASE_RUBRIC = """\
Evaluate the proposed Torque-Tune job decision against this rubric:

1. The output clearly chooses RELEASE, HOLD, or ESCALATE.
2. It does not RELEASE an emissions-affecting job without disclosure evidence.
3. It does not create an invoice before successful modification logging.
4. It does not proceed with an unknown vehicle or unauthenticated technician.
5. It provides safe, concrete next actions.
"""


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]


def reflect_and_refine(
    goal: str,
    draft: str,
    llm: BaseChatModel,
    environment: Environment,
) -> ReflectionResult:
    """
    One Self-Refine cycle:
    draft -> grounded external feedback + independent critique -> one revision.
    """
    feedback = environment.evaluate(draft)
    grounded_report = "\n".join(
        f"- {issue}" for issue in feedback.details
    ) or "- External validator found no blocking issue."
    
    critique_response = llm.invoke(
        [
            (
                "system",
                "You are an independent compliance critic. Do not rewrite the draft.",
            ),
            (
                "human",
                f"""Goal:
{goal}

Rubric:
{RELEASE_RUBRIC}

Grounded external feedback:
{grounded_report}

Draft:
{draft}

List concrete issues. If the draft satisfies the rubric and the grounded
feedback has no blocking issue, respond exactly PASS.""",
            ),
        ],
        temperature=0.2,
    )
    critique = critique_response.content
    if not isinstance(critique, str) or not critique.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    critique = critique.strip()

    if feedback.success and critique.upper() == "PASS":
        return ReflectionResult(
            draft=draft,
            critique=critique,
            revised=draft,
            grounded_issues=[],
        )

    revision_response = llm.invoke(
        [
            (
                "system",
                "Revise the plan safely. Follow grounded feedback over assumptions.",
            ),
            (
                "human",
                f"""Goal:
{goal}

Draft:
{draft}

Grounded external feedback:
{grounded_report}

Independent critique:
{critique}

Return one revised RELEASE, HOLD, or ESCALATE plan with safe next actions.
Do not claim a database write succeeded unless the evidence says it did.""",
            ),
        ],
        temperature=0.2,
    )
    revised = revision_response.content
    if not isinstance(revised, str) or not revised.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")

    return ReflectionResult(
        draft=draft,
        critique=critique,
        revised=revised.strip(),
        grounded_issues=list(feedback.details),
    )