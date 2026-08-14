from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict
import asyncio
import inspect


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str


async def dynamic_decomposition(
    goal: str,
    llm: BaseChatModel,
    max_steps: int = 4,
    executor=None,
) -> list[tuple[str, str]]:

    history: list[tuple[str, str]] = []

    for step in range(max_steps):
        observation = "\n".join(
            f"{task}: {result}" for task, result in history
        ) or "None"

        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke([
            (
                "system",
                "You are an adaptive planner. "
                "Use prior observations before deciding what comes next.",
            ),
            (
                "human",
                f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met.
When done is true, use an empty string for next_task.""",
            ),
        ], temperature=0.1)

        if decision.done:
            break

        task = decision.next_task.strip()

        if not task:
            raise ValueError(
                f"Dynamic planner omitted next_task at step {step + 1}"
            )

        # NEW: use the project's executor when provided
        if executor is not None:
            result = executor(task)
        if inspect.isawaitable(result):
            result = await result
        else:
            response = await asyncio.to_thread(llm.invoke([
                (
                    "system",
                    "Execute the next adaptive sub-task using the observations provided.",
                ),
                (
                    "human",
                    f"Goal: {goal}\n"
                    f"Next task: {task}\n"
                    f"Prior observations:\n{observation}",
                ),
            ], temperature=0.2))

            result = response.content

        if not isinstance(result, str) or not result.strip():
            raise RuntimeError(
                "The task executor returned an empty or unsupported response"
            )

        history.append((task, result.strip()))

    return history