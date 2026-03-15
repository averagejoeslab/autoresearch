# autoresearch

Autonomous research of agent runtime primitives using the [autoresearch](https://github.com/karpathy/autoresearch) pattern.

Agent runtime primitives — context management, memory, tool use, planning, verification, orchestration — are the unsolved problems blocking agents from being reliable at scale. Every framework implements the same set of primitives. The research community agrees on *what* agents need. What's missing is principled answers for *how to implement each primitive well*.

This repo applies an AI agent to autonomously iterate on each primitive's implementation: modify the code, run a benchmark, keep improvements, discard regressions, repeat overnight.

## Structure

Each experiment folder contains three files:

```
benchmark.py    — locked evaluation (trust boundary, agent cannot modify)
harness.py      — mutable implementation (agent iterates on this)
program.md      — agent instructions (defines the experiment loop)
```

### Research areas

```
harnesses/
├── context/              #1 open problem in agent quality
│   ├── compaction/        Summarizing/compressing conversation history
│   ├── window-packing/    Optimal token budget allocation
│   └── jit-loading/       Loading context on-demand vs upfront
├── memory/               Long-term persistence and retrieval
│   ├── retrieval/         Storage backends + search strategies
│   └── consolidation/    Episodic → semantic generalization
├── tools/                Tool interface design + usage patterns
│   ├── interface-design/  How tool descriptions affect performance
│   ├── selection/         When to use which tool
│   └── composition/      Chaining tools for multi-step operations
├── planning/             Task decomposition + strategy selection
│   ├── decomposition/     Breaking tasks into executable steps
│   ├── strategy-selection/ Choosing CoT vs ToT vs ReAct
│   └── recovery/         Re-planning after failures
├── verification/         Self-checking + error recovery
│   ├── self-check/        Output quality assessment + calibration
│   ├── error-recovery/    Detecting and recovering from errors
│   └── automated-grading/ LLM-as-judge accuracy + calibration
└── orchestration/        Sub-agent delegation + multi-agent
    ├── delegation/        When and how to delegate
    ├── context-filtering/  What context sub-agents need
    └── result-aggregation/ Merging outputs from multiple agents

benchmarks/               Benchmark design research
├── task-generation/       Auto-generating discriminative tasks
├── difficulty-calibration/ Calibrating task difficulty
└── contamination-resistance/ Preventing benchmark gaming

evals/                    Evaluation methodology research
├── grading-strategies/    Code-based vs model-based vs hybrid
├── reliability-metrics/   pass@k vs pass^k, consistency
├── prompt-evaluation/     Prompt effectiveness across models
└── llm-judge/             LLM-as-judge harness optimization

training-data/            Agent training data research
├── agent-trajectories/    Curating data from agent runs
└── tool-use-data/         Training data for tool calling

security/                 Agent security research (defensive)
├── prompt-injection/      Direct injection techniques + detection
├── indirect-injection/    Attacks embedded in tool outputs / documents
├── data-exfiltration/     Preventing sensitive data leaks
├── tool-misuse/           Detecting harmful tool usage patterns
├── guardrail-evasion/     Hardening safety guardrails against bypasses
└── defense-strategies/    Combined multi-layer defense stacks

architectures/            Model architecture research
├── classifier/            Best encoder + head for agent classification tasks
└── inference-pipeline/    Multi-stage pipeline structure optimization

runtime/                  Compose best variants into a full agent
```

## Quick start

```bash
# Run a specific experiment
cd harnesses/tools/interface-design
python benchmark.py

# Or point an AI agent at it
# "Read program.md and start experimenting"
```

## Inner model

Each benchmark needs an LLM as the "brain" inside the harness:

- **Local (default):** Set `LOCAL_MODEL_PATH` to a GGUF model file. Free, fast, overnight-capable.
- **API:** Set `MODEL_BACKEND=api` + `MODEL_API_KEY` + `MODEL_NAME`. Any OpenAI-compatible endpoint.

## Research basis

Topic selection based on: CoALA (TMLR 2024), LangChain State of Agent Engineering 2025, METR evaluations, MemAgents (ICLR 2026), and analysis of LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, and Claude Agent SDK architectures.

## License

MIT
