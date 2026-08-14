# Torque-Tune Planning Agent

## Goal

Build a Planning Agent that reviews high-risk tuning jobs and decides
whether the job should be:

-   **RELEASE** --- safe to proceed.
-   **HOLD** --- required evidence is missing.
-   **ESCALATE** --- shift-lead review or customer confirmation is
    required.

## Business Problem

A request such as:

> "Review this ECU remap and decat job and decide whether it can be
> released."

requires multiple checks before any consequential database write.

The agent must verify:

-   Client and vehicle identity
-   Previous tuning history
-   Appointment
-   Technician identity and authorization
-   Compliance and warranty requirements
-   Required disclosure/sign-off

The MCP server provides small tools for these operations, but there is
no single `release_modification_job()` tool. Therefore, the agent must
plan and orchestrate the steps.

## Planning Approaches

### 1. Decomposition-First

Generate the complete task graph first, then execute it in topological
order.

``` text
vehicle → history → appointment → technician → compliance → disclosure → decision
```

### 2. Dynamic / Interleaved Decomposition

Execute a step, inspect the MCP/database result, and adapt the next
steps.

``` text
vehicle → disclosure declined → HOLD / ESCALATE
```

All task graphs must be acyclic (DAG).

## Algorithm Routing

  Task                                                     Method
  -------------------------------------------------------- ------------------
  Client/vehicle, history, appointment, policy checks      Plan-and-Solve
  Compare RELEASE / HOLD / ESCALATE                        Tree of Thoughts
  Final plan selection with MCP/DB feedback                LATS
  Improve HOLD/ESCALATE notices                            Self-Refine
  Retry a failed plan using previous failure information   Reflexion

## Grounded Validation

The proposed plan must be validated against real evidence from:

-   SQLite/database
-   MCP tool results
-   Compliance policy
-   Technician authentication state

A plan must fail if it:

-   Uses an unknown or ambiguous vehicle.
-   Uses an unauthenticated or mismatched technician.
-   Completes emissions-affecting work without successful disclosure
    evidence.
-   Creates an invoice before a valid release decision and successful
    modification logging.
-   Ignores a required HOLD or ESCALATE decision.

An independent critic should review the proposed plan against the
available evidence before final consequential writes.

## Self-Correction

-   **Self-Refine:** Improve single outputs such as HOLD or ESCALATE
    notices using a defined rubric.
-   **Reflexion:** Learn from failed release attempts and use the
    reflection in later trials within the same run.

## Evaluation

Use a fixed test suite to compare:

-   Decomposition-First vs Dynamic Decomposition
-   Plan-and-Solve
-   Tree of Thoughts
-   LATS
-   Self-Refine
-   Reflexion
-   Grounded vs Ungrounded Critique

Measure:

-   Task success / accuracy
-   LLM calls
-   Token consumption
-   Latency
-   Estimated cost
