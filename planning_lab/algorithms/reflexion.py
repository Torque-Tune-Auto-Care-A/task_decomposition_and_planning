from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ..models import EnvironmentFeedback
from .environment import Environment


@dataclass
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]


def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    max_trials: int = 3,
    memory_size: int = 3,
) -> ReflexionResult:
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive")

    memory: list[str] = []
    trials: list[ReflexionTrial] = []
    best_attempt = ""
    best_score = -1.0

    for number in range(1, max_trials + 1):
        recalled = "\n".join(
            f"- {item}" for item in memory[-memory_size:]
        ) or "- No prior failed trials."

        response = llm.invoke(
            [
                (
                    "system",
                    "You are a Torque-Tune planning agent retrying a high-risk "
                    "job-release decision.",
                ),
                (
                    "human",
                    f"""Task:
{task}

Lessons from previous failed trials:
{recalled}

Return a complete RELEASE, HOLD, or ESCALATE plan. Include safe next actions.
Never propose modification completion or invoicing without the required evidence.
Use the remembered lessons, but do not mention this memory in the answer.""",
                ),
            ],
            temperature=0.2,
        )
        attempt = response.content
        if not isinstance(attempt, str) or not attempt.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        attempt = attempt.strip()

        feedback = environment.evaluate(attempt)
        trial = ReflexionTrial(number=number, attempt=attempt, feedback=feedback)

        if feedback.score > best_score:
            best_attempt, best_score = attempt, feedback.score

        if feedback.success:
            trials.append(trial)
            return ReflexionResult(
                success=True,
                output=attempt,
                trials=trials,
                memory=memory[-memory_size:],
            )

        reflection_response = llm.invoke(
            [
                (
                    "system",
                    "Write one concise first-person lesson for the next planning trial.",
                ),
                (
                    "human",
                    f"""Task:
{task}

Failed plan:
{attempt}

Grounded validator feedback:
{chr(10).join("- " + item for item in feedback.details)}

State what I did wrong and what I must check or do differently next time.
Start with "I".""",
                ),
            ],
            temperature=0.2,
        )
        reflection = reflection_response.content
        if not isinstance(reflection, str) or not reflection.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")

        trial.reflection = reflection.strip()
        trials.append(trial)
        memory.append(trial.reflection)

    return ReflexionResult(
        success=False,
        output=best_attempt,
        trials=trials,
        memory=memory[-memory_size:],
    )