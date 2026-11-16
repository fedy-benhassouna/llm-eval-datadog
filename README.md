# LLM Evaluation & Observability for an AI Support Agent

An evaluation and monitoring framework for a tool-using e-commerce customer-support agent, built on **Google Gemini** and **Datadog LLM Observability**.

The same agent and the same evaluators run in two places:

- **Offline experiments.** Before release, the agent runs against a golden dataset and every response is scored on five metrics. Each run is a versioned Datadog experiment that can be compared with a baseline.
- **Online monitoring.** In production, every live conversation is traced end to end and scored by reference-free evaluators. The scores are attached to the trace.

---

## Architecture

```mermaid
flowchart LR
    subgraph Inputs
        DS[Golden dataset<br/>9 scenarios]
        LIVE[Live customer chats]
    end

    subgraph Agent["Support agent (Gemini 3.1 Flash-Lite)"]
        LLM[Function-calling LLM]
        T1[lookup_order]
        T2[issue_refund<br/>≤ $50 enforced in code]
        LLM --> T1
        LLM --> T2
    end

    subgraph Judges["Evaluation layer"]
        J1[Faithfulness]
        J2[Topic relevancy]
        J3[Negative sentiment]
        J4[Toxicity & policy safety]
        J5[Tool trajectory]
    end

    subgraph Datadog["Datadog LLM Observability"]
        EXP[Experiments<br/>versioned runs + baseline comparison]
        TR[Traces<br/>spans + attached evaluations]
    end

    DS --> Agent
    LIVE --> Agent
    Agent --> Judges
    Judges -->|offline| EXP
    Judges -->|online| TR
```

| Layer | Responsibility | Implementation |
|---|---|---|
| Agent | Answers order, shipping and refund questions with function calling. Enforces the $50 refund policy in code. | [`agent/support_agent.py`](agent/support_agent.py) |
| Dataset | Golden test cases with expected output and expected tool trajectory, in three categories: happy path, edge case, adversarial. | [`data/golden_dataset.jsonl`](data/golden_dataset.jsonl), [`generate_dataset.py`](generate_dataset.py) |
| Evaluation | Four LLM-as-a-judge evaluators with structured JSON output, plus one deterministic evaluator. | [`evaluators/judge.py`](evaluators/judge.py) |
| Offline experiments | Runs the agent over the dataset and logs every result to a Datadog experiment. | [`run_experiment.py`](run_experiment.py) |
| Online monitoring | Traces production conversations and attaches reference-free scores to each trace. | [`prod_monitor.py`](prod_monitor.py) |

Each conversation is traced as a span tree: workflow → LLM calls → tool calls, with latency, token usage and cost per step.

<p align="center">
  <img src="docs/images/experiment-trace.png" alt="Datadog trace of an agent conversation" width="800">
  <br><em>Datadog trace for "What is the status of order #ORD-101?": the model selects the tool, the order lookup runs, and the model writes the answer.</em>
</p>

---

## Evaluation metrics

| Metric | What it measures | Method | Scale | Offline | Online |
|---|---|---|---|:-:|:-:|
| **Faithfulness** | Factual claims about orders are backed by the order data (no hallucination) | LLM judge, claim-level | 0–1 (higher is better) | ✓ | |
| **Topic relevancy** | Stays within the support domain, or politely declines | LLM judge | pass / fail | ✓ | ✓ |
| **Negative sentiment** | Share of statements with a hostile or frustrated tone | LLM judge | 0–1 (lower is better) | ✓ | ✓ |
| **Toxicity & policy safety** | Toxic content, or compliance with prompt injection | LLM judge | 1–5 Likert (1 = safe) | ✓ | ✓ |
| **Tool trajectory** | The agent called exactly the expected tools | Deterministic | pass / fail | ✓ | |

Online monitoring uses only the metrics that need no expected answer, so they can score any live conversation.

<p align="center">
  <img src="docs/images/production-trace.png" alt="Datadog production trace with attached evaluations" width="800">
  <br><em>Online monitoring: a customer attempts a $900 refund override. The agent declines, and three evaluations are attached to the trace.</em>
</p>

---

## Results (current baseline)

| Metric | Score |
|---|---|
| Faithfulness | 0.89 avg |
| Topic relevancy | 89% |
| Toxicity & policy safety | 1.0 avg (all responses safe) |
| Negative sentiment | 0.0 |
| Tool trajectory | 89% |

The agent reached the correct outcome in **9 of 9** scenarios and declined **3 of 3** adversarial prompts, including prompt injection and policy override.

<p align="center">
  <img src="docs/images/experiment-comparison.png" alt="Datadog experiment compared with its baseline" width="900">
  <br><em>Datadog Experiments: a run (A) compared with its baseline (B), metric by metric.</em>
</p>

---

## Getting started

### Prerequisites

- Python 3.11+
- A Google Gemini API key
- A Datadog account with LLM Observability, plus an **API key** and an **Application key**

### Setup

```bash
git clone https://github.com/fedy-benhassouna/llm-eval-datadog.git
cd llm-eval-datadog

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in your keys
```

`DD_SITE` must match your Datadog region, for example `datadoghq.eu` for EU accounts.

### Usage

```bash
# 1. (Optional) Regenerate the golden dataset with Gemini
python generate_dataset.py

# 2. Run an offline experiment; prints the Datadog experiment URL
python run_experiment.py

# 3. Run the online monitoring example
python prod_monitor.py
```

A full offline run takes a few minutes, mostly waiting on model API calls.

---

## Project structure

```
.
├── agent/
│   └── support_agent.py      # Tool-using Gemini agent, instrumented with Datadog decorators
├── evaluators/
│   └── judge.py              # LLM-as-a-judge and deterministic evaluators
├── docs/images/           # Datadog screenshots
├── data/
│   └── golden_dataset.jsonl  # Golden test cases
├── generate_dataset.py       # Synthetic dataset generation with structured output
├── run_experiment.py         # Offline evaluation → Datadog experiment
├── prod_monitor.py           # Online evaluation → Datadog traces
├── requirements.txt
└── .env.example
```

---

## Design decisions

- **Policy enforced in code, not only in the prompt.** The $50 refund cap is checked inside the tool, so no prompt injection can bypass it.
- **Structured judge output.** Every judge returns a Pydantic-validated JSON schema, so scores are parsed reliably.
- **One evaluator set for offline and online.** Pre-release and production quality are measured the same way.
- **Resilient model calls.** Calls are retried automatically with backoff on HTTP 429, 500 and 503.
- **Low-cost model tier.** The agent and the judges run on Gemini 3.1 Flash-Lite.

---

## Roadmap

- [ ] Pass tool results to the faithfulness judge to verify action outcomes
- [ ] Expand the golden dataset to 100+ scenarios drawn from real support tickets
- [ ] Datadog monitors and alerts on production safety and relevancy scores
- [ ] Run experiments in CI on every prompt, tool or model change
- [ ] A/B test the judge models against human-labelled samples
