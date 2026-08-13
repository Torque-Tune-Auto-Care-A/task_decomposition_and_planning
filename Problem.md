# Torque-Tune Planning Agent

## Master Business Problem

### High-Risk Multi-Stage Performance Tuning & Compliance Resolution

Torque-Tune is a performance and tuning garage chain where front-desk staff and technicians handle requests involving vehicle tuning, previous work, appointments, performance goals, budgets, warranty considerations, and compliance-sensitive modifications.

The goal of this project is to build a **Planning Agent** capable of handling complex tuning requests that cannot be safely resolved through a single LLM turn or a single MCP tool call.

A typical request may look like:

> "I want to improve my car's performance using a setup similar to the previous tuning, keep it within my budget, and let me know if the requested modifications could affect the warranty or compliance."

Resolving this request requires the agent to identify the correct client and vehicle, review tuning history and appointments, understand the requested modifications, evaluate the relevant constraints, select an appropriate work plan, execute the required operations through MCP tools, and validate the final state before committing consequential database changes.

The key challenge is that **the result of one step can change what should happen next**. For example, discovering that a client owns multiple vehicles may require additional disambiguation, while discovering previous modifications may introduce additional compliance requirements or make the original plan unsuitable.

---

## Why This Is a Planning Problem

The MCP server intentionally provides narrow tools for individual operations, such as:

- Finding a client.
- Identifying a vehicle.
- Retrieving tuning history.
- Checking appointments.
- Checking technician information.
- Logging completed work.
- Creating invoices.

There is no single high-level operation such as `complete_tuning_request()`.

The Planning Agent must therefore orchestrate these tools and reason about their dependencies.

A complex request may require the agent to:

1. Identify the correct client.
2. Identify the correct vehicle.
3. Retrieve previous tuning history.
4. Check the relevant appointment.
5. Interpret the requested modifications.
6. Determine performance, budget, warranty, and compliance constraints.
7. Generate and evaluate possible work plans.
8. Verify required approvals or sign-offs.
9. Execute approved operations through MCP.
10. Validate the final state before creating the invoice.

This makes the problem fundamentally different from a simple retrieval or memory task. The agent must decide **what to do, when to do it, and whether it is safe to proceed**.

---

## Core Business Constraints

The agent must balance several potentially competing objectives:

| Constraint | Role |
|---|---|
| **Performance** | Achieve the customer's requested performance improvements. |
| **Budget** | Keep the proposed work within the customer's available budget when possible. |
| **Warranty** | Identify modifications that may affect manufacturer warranty coverage. |
| **Compliance** | Detect sensitive modifications that require additional verification. |
| **Operational Constraints** | Consider appointments, technician availability, and previous vehicle work. |
| **Safety** | Prevent invalid or unauthorized work from being recorded as completed. |

The agent therefore cannot simply optimize for maximum performance. It must reason about the trade-offs between these constraints before selecting a plan.

---

## Detailed Implementation Concerns

### 1. DAG Decomposition Methods
The agent implements two distinct decomposition strategies against the same request type:
- **Decomposition-First:** Generates the complete task graph upfront and executes in strict topological order.
- **Dynamic / Interleaved Decomposition:** Evaluates intermediate tool outputs after each step, allowing early discoveries (e.g., unexpected diagnostic codes or undocumented modifications) to dynamically alter or replace downstream subtasks.
- **Acyclicity Enforcement:** All generated task graphs are validated at construct time to ensure zero circular dependencies or potential deadlocks.

### 2. Planning Algorithm Routing
Subtasks are routed to specific planning algorithms based on their structural requirements:
- **Plan-and-Solve:** Used for deterministic, single-pass operations (e.g., client lookups, historical records retrieval, simple cost summation).
- **Tree of Thoughts (ToT):** Used for combinatorial optimization subtasks with competing constraints (e.g., balancing ECU remap stages, budget limits, and emissions compliance).
- **Language Agent Tree Search (LATS):** Used for parameter evaluation requiring external feedback (e.g., simulating ECU parameters against dynamic vehicle thresholds with verbal reflections on failure branches).

### 3. Self-Correction Scopes
- **Self-Refine:** Applied to single-turn text outputs (e.g., drafting legal exemption disclaimers and customer notices) using explicit rubric-based evaluation.
- **Reflexion:** Applied to complex MCP database write operations where schema or authorization constraints may cause initial failure. Maintains a capped episodic buffer of verbal reflections across trials within the run.

### 4. Grounded vs. Ungrounded Critique
To eliminate self-evaluation bias:
- **Grounded Critique Sources:** Hard database schema checks, strict emissions compliance validator functions, and explicit technician signature presence checks serve as absolute sources of truth.
- **Independent Critic:** An isolated model instance reviews compliance logs and authorization state before final transaction commitment.

### 5. Benchmark & Evaluation Matrix
The system is evaluated across a standardized test suite comparing Accuracy, LLM Calls, Token Consumption, and Latency across all planning, decomposition, and critique modalities.