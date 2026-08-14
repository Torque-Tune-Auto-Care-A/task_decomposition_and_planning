from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from .algorithms import (
    Environment,
    decompose_goal,
    dynamic_decomposition,
    execute_plan,
    final_output,
    flatten_lats_tree,
    lats,
    plan_and_solve,
    reflect_and_refine,
    reflexion,
    tree_of_thoughts,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.client import TorqueTuneAgent
from agent.config import load_config
from planning.torque_tune_environment import PlanningContext, TorqueTuneEnvironment


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Week 4: decomposition, planning, and reflection lab"
    )
    cli.add_argument(
        "goal",
        nargs="?",
        default="Review a high-risk ECU remap and decat job and decide whether to release, hold, or escalate it.",
    )
    cli.add_argument(
        "--mode",
        choices=["dag", "dynamic", "ps", "tot", "reflexion", "lats"],
        default="dag",
    )
    cli.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    cli.add_argument("--client-id", type=int, default=2)
    cli.add_argument("--vehicle-id", type=int, default=3)
    cli.add_argument("--tech-id", type=int, default=2)
    cli.add_argument("--appointment-id", type=int, default=3)
    cli.add_argument("--technician-authenticated", action="store_true")
    cli.add_argument("--disclosure-confirmed", action="store_true")
    cli.add_argument("--modification-logged", action="store_true")
    cli.add_argument("--depth", type=int, default=2, choices=range(1, 4))
    cli.add_argument("--beam-width", type=int, default=2, choices=range(1, 4))
    cli.add_argument("--max-trials", type=int, default=3, choices=range(1, 6))
    cli.add_argument("--memory-size", type=int, default=3, choices=range(1, 6))
    cli.add_argument("--iterations", type=int, default=2, choices=range(1, 6))
    cli.add_argument("--n-actions", type=int, default=2, choices=range(1, 4))
    cli.add_argument("--no-reflection", action="store_true")
    return cli


def save_artifact(payload: dict) -> Path:
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = artifact_dir / f"run-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


async def main() -> None:
    # Models may return arrows, em dashes, or other characters that Windows'
    # legacy cp1252 console cannot encode. UTF-8 keeps CLI output portable.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in the project .env")
    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=args.model,
        temperature=0.2,
        max_retries=2,
    )
    environment = Environment(
        TorqueTuneEnvironment(
            PlanningContext(
                client_id=args.client_id,
                vehicle_id=args.vehicle_id,
                tech_id=args.tech_id,
                appointment_id=args.appointment_id,
                technician_authenticated=args.technician_authenticated,
                disclosure_confirmed=args.disclosure_confirmed,
                modification_logged=args.modification_logged,
                request_text=args.goal,
            )
        ).evaluate
    )
    payload: dict = {"mode": args.mode, "model": args.model, "goal": args.goal}

    if args.mode == "dag":
        plan = decompose_goal(args.goal, llm)
        print("Execution batches:", plan.execution_batches())
        async with TorqueTuneAgent(load_config()) as agent:
            outputs = await execute_plan(
                plan,
                llm,
                pipeline=agent.memory,
            )
        draft = final_output(plan, outputs)
        reflection = (
            reflect_and_refine(args.goal, draft, llm, environment)
            if not args.no_reflection
            else None
        )
        result = reflection.revised if reflection else draft
        payload.update(plan=plan.model_dump(), outputs=outputs, result=result)
        if reflection:
            payload["reflection"] = {
                "grounded_issues": reflection.grounded_issues,
                "critique": reflection.critique,
                "revised": reflection.revised != reflection.draft,
            }
    elif args.mode == "dynamic":
        async with TorqueTuneAgent(load_config()) as agent:

            async def project_executor(task: str) -> str:
                return await agent.run_turn(
                    f"""Work on one step of an automotive-service goal.

Overall goal:
{args.goal}

Current task:
{task}

Use MCP tools, scoped memory, and grounded knowledge-base evidence when needed.
Do not invent vehicle facts, policies, or technical specifications."""
                )

            history = await dynamic_decomposition(
                args.goal,
                llm,
                executor=project_executor,
            )

        result = (
            history[-1][1]
            if history
            else "Planner reported the goal was already complete."
        )
        payload.update(history=history, result=result)
    elif args.mode == "ps":
        result = plan_and_solve(args.goal, llm)
        payload["result"] = result
    elif args.mode == "tot":
        thoughts = tree_of_thoughts(args.goal, llm, args.depth, args.beam_width)
        result = thoughts[0].state if thoughts else "No viable thought survived."
        payload.update(thoughts=[thought.model_dump() for thought in thoughts], result=result)
    elif args.mode == "reflexion":
        outcome = reflexion(args.goal, llm, environment, args.max_trials, args.memory_size)
        result = outcome.output
        payload.update(
            success=outcome.success,
            trials=[
                {
                    "number": trial.number,
                    "attempt": trial.attempt,
                    "feedback": trial.feedback.model_dump(),
                    "reflection": trial.reflection,
                }
                for trial in outcome.trials
            ],
            memory=outcome.memory,
            result=result,
        )
    else:
        outcome = lats(args.goal, llm, environment, args.iterations, args.n_actions)
        result = outcome.output
        payload.update(
            success=outcome.success,
            best_score=outcome.best_score,
            iterations=outcome.iterations,
            tree=flatten_lats_tree(outcome.root),
            result=result,
        )

    artifact = save_artifact(payload)
    print("\nRESULT\n======\n" + result)
    print(f"\nRun artifact: {artifact}")


if __name__ == "__main__":
    asyncio.run(main())