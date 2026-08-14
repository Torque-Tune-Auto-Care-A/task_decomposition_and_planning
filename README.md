# Task Decomposition & Planning Lab — Torque-Tune fork

Forked from [AmrSheta22/task_decomposition_and_planning](https://github.com/AmrSheta22/task_decomposition_and_planning)
and adapted as the implementation layer for the Torque-Tune planning agent:
reviewing high-risk tuning jobs (ECU remap / decat) and deciding
RELEASE / HOLD / ESCALATE before any consequential database write.

The pipeline keeps the upstream structure:

- Decomposition-first: structured task DAG, executed in topological order.
- DAG validation: Pydantic validates ids/dependencies; NetworkX rejects cycles.
- Parallel scheduling: independent nodes share dependency-safe batches.
- Dynamic decomposition: `--mode dynamic` interleaves planning and observations.
- Plan-and-Solve: `--mode ps` (plan phase, then solve phase).
- Tree of Thoughts: `--mode tot` (generate/evaluate/beam search).
- Self-Refine: `--mode dag` adds an independent compliance critic plus grounded
  external feedback before one revision.
- Reflexion: `--mode reflexion` retries the full task with bounded episodic memory.
- LATS: `--mode lats` (MCTS with external feedback, branch reflection, UCT,
  backpropagation).

Structured outputs use Pydantic schemas via
`with_structured_output(..., method="json_schema")`; the provider was swapped
from Mistral to Google Gemini to match the host repository.

## Torque-Tune adaptations (delta vs. upstream)

- **Provider**: `ChatMistralAI` → `ChatGoogleGenerativeAI` (`GEMINI_API_KEY`).
- **Domain prompts**: decomposition and self-correction prompts now review
  release requests and must choose RELEASE/HOLD/ESCALATE with safe next actions.
- **`algorithms/environment.py`**: the upstream randomized evaluator is replaced
  by a generic adapter accepting any validator callable. The host project injects
  its grounded validator (`planning/torque_tune_environment.py`), which checks
  SQLite evidence (vehicle/client ownership, technician existence, appointment
  match) and compliance rules (no RELEASE of emissions work without disclosure,
  no invoice before modification logging, explicit decision required).
- **`cli.py`**: builds that environment from MCP-session context flags
  (`--client-id`, `--vehicle-id`, `--tech-id`, `--appointment-id`,
  `--technician-authenticated`, `--disclosure-confirmed`, `--modification-logged`)
  and passes it to Self-Refine, Reflexion, and LATS.

## Setup

This folder lives inside the host project as `planning_toolkit/`. From the
project root:

    .\venv\Scripts\python.exe -m pip install -r requirements.txt

The project `.env` must contain `GEMINI_API_KEY`; the grounded validator reads
`db/redline.db` (see the host README for seeding).

## Run

    .\venv\Scripts\python.exe -m planning_toolkit.planning_lab.cli --mode reflexion --max-trials 3 --memory-size 2
    .\venv\Scripts\python.exe -m planning_toolkit.planning_lab.cli --mode lats --iterations 2 --n-actions 2
    .\venv\Scripts\python.exe -m planning_toolkit.planning_lab.cli --mode dag

Each run saves a JSON trace in `artifacts/` (plans, node outputs, critic
feedback, episodic memories, MCTS visits, branch reflections).

## Test

From the project root (`pytest.ini` sets the import paths):

    .\venv\Scripts\python.exe -m pytest -q planning_toolkit/tests planning/tests

Toolkit tests use deterministic fake models and spend no API credits; the
validator tests in `planning/tests` read the real `db/redline.db`.

## Suggested exercises

- Introduce a cycle in a test plan and watch validation fail before execution.
- Compare sequential execution with the reported parallel batches.
- Make an early dynamic-decomposition observation fail and inspect the next task.
- Compare PS with ToT on a lookahead problem, then count calls.
- Swap the injected validator for an always-success one and compare what
  Reflexion/LATS accept — the grounded validator catches RELEASE without
  disclosure that the ungrounded baseline misses.
- Run `--mode reflexion` with `--disclosure-confirmed` off vs. on and inspect
  how episodic memory changes across trials.
- Compare ToT's model-only scores with LATS's environment scores and UCT visit
  counts in the artifacts.