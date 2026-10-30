# Pipeline Latency Governance for a GenAI Claims Assistant

> A Forward Deployed Engineering (FDE) proof of concept. This repository contains both the **pre-transformation** (before) and **post-transformation** (after) states of the solution, so the impact of the work can be seen side by side.

---

## 1. Overview

| Item | Details |
|------|---------|
| **POC Name** | Pipeline Latency Governance for a GenAI Claims Assistant |
| **Scenario / Customer** | Illustrative insurance motor-claims assistant: a PDF claim is ingested, extracted, redacted, retrieved against, summarised by an LLM, validated and stored; a handler later opens the stored case |
| **Owner** | Adi Raj |
| **Status** | Completed |
| **Program** | Birlasoft FORGE FDE Academy · Sprint 1 · Day 16 |

**One-line summary:** Measures real LLM generation latency on a live Azure OpenAI deployment, composes it with the other pipeline stages, and turns the result into a defensible, two-path latency requirement instead of a single unachievable "P95 < 5 s" target.

---

## 2. Problem Statement

A blanket latency target ("P95 under 5 seconds") was handed down without saying which part of the system it applies to. Applied to the wrong path it is either impossible to meet or meaningless, and a team held to it will report a number it cannot hit or quietly redefine what it measures.

- Latency is **not a bell curve**. It is a lognormal body with a heavy tail (cold starts, long completions, noisy neighbours), so mean-based reasoning and alerts miss the events users actually feel.
- Stage-by-stage budgets are misleading: summing stage P95s and summing stage P50s are both wrong, in opposite directions.
- Ingestion (asynchronous, nobody waiting) and handler read (synchronous, a person waiting) have completely different latency needs but were governed by one number.

---

## 3. Pre-Transformation (Before)

The starting state is an **unexecuted** Colab notebook and a prompt-and-review exercise brief: the structure, prompts and acceptance checklists exist, but no measurements have been taken and no requirement has been written.

| Aspect | Current State |
|--------|---------------|
| Process | Single latency target assumed for the whole system; no measured distribution |
| Model / Tooling | Notebook with no cell outputs; Azure OpenAI client not yet exercised |
| Cost | Not measured (about 61 billed calls planned for a full run) |
| Latency / Performance | Unknown. No P50/P95/P99 for generation, streaming or the pipeline |
| Accuracy / Quality | No evidence for or against the 5 s target |
| Pain points | Unnamed path in the target, risk of mean-based alerting, percentile-addition shortcuts |

**Location in repo:** [Pre-Transformation/](Pre-Transformation/)

---

## 4. Post-Transformation (After)

The same notebook, executed **live** against an Azure OpenAI deployment (`gpt-4.1-mini`), with every measurement cell populated, a composed pipeline simulation, an exported report and a written analysis.

| Aspect | New State |
|--------|-----------|
| Process | Measure → characterise tail → compose pipeline → compare two paths → write NFR |
| Model / Tooling | `AzureOpenAI` SDK with `max_retries=0`, `numpy` Monte Carlo (20,000 runs), `matplotlib` |
| Cost | About 61 live calls for the full run |
| Latency / Performance | Generation P50 1.63 s / P95 3.27 s / P99 8.84 s (30 calls); ingestion P95 about 32 s; interactive P95 527 ms |
| Accuracy / Quality | Failures counted as data; TTFT measured at first content chunk; bootstrapped real tail |
| Benefits | A two-part, path-named NFR and a correct alerting strategy |

**Location in repo:** [Post-Transformation/](Post-Transformation/)

---

## 5. Before vs. After

| Metric | Pre-Transformation | Post-Transformation | Change |
|--------|--------------------|---------------------|--------|
| Generation latency (P50 / P95 / P99) | Not measured | 1,630 / 3,272 / 8,843 ms | Baseline established |
| Streaming time-to-first-token (P50) | Not measured | 1,941 ms (total P50 2,809 ms) | Measured |
| Chain-of-thought cost multiplier | Not measured | 2.28× | Measured |
| Ingestion path P95 (parallel) | Assumed "< 5 s" | 32,348 ms | About 6.5× over the 5 s target |
| Interactive path P95 | Assumed "< 5 s" | 527 ms | PASS (about 9× headroom) |
| Latency requirement | One unnamed number | Two numbers, each tied to a path and a window | Defensible |

Composition check (shows percentiles are not additive):

| Shortcut | Value | Versus actual pipeline P95 (32,688 ms) |
|---|---|---|
| Sum of every stage's P95 | 35,661 ms | Overestimates |
| Sum of every stage's P50 | 12,351 ms | Underestimates |

Source: [Post-Transformation/latency_report.json](Post-Transformation/latency_report.json)

---

## 6. Approach

1. **Discover** – Identify the two user paths (ingestion vs. handler read) and the stages in each.
2. **Baseline** – Make 30 sequential live calls with retries disabled to capture the real generation distribution.
3. **Evaluate** – Measure streaming TTFT, output-length slope and chain-of-thought cost; test interventions by how much each moves P95 relative to P50.
4. **Build** – Compose 8 modelled stages (lognormal plus tail event, with a dependency graph) with bootstrapped real generation samples.
5. **Measure** – Compare both paths to the 5,000 ms target and check the sum-of-P95 and sum-of-P50 shortcuts.
6. **Hand off** – Export `latency_report.json`, write the NFR and answer the six ADR questions in the notebook.

### Decision Criteria

| Criterion | Weight | Why |
|---|---|---|
| Honest about measured vs. modelled | High | Only generation is measured live; other stages are course-specified assumptions |
| Tail behaviour over averages | High | The commitment is a tail percentile, and a mean-based alert cannot detect P95 degradation |
| Path-specific requirements | High | One number for two different paths is "an argument waiting to happen" |
| Cost of measurement | Medium | Small, stated call budget; a tighter P95 means a real spend |

---

## 7. Repository Structure

```
Assignment9/
├── README.md
├── Pre-Transformation/                  # "Before" state
│   ├── DecisionStream_Day16_Latency_Colab.ipynb   # Notebook, no outputs
│   ├── Day16_Latency_Generate_and_Govern_Exercises.md
│   └── P50_P95_P99_Grounded_Case_Studies_for_FDE_Students.docx
└── Post-Transformation/                 # "After" state
    ├── DecisionStream_Day16_Latency_Colab.ipynb   # Executed live, outputs and answers
    ├── Day16_Latency_Generate_and_Govern_Exercises.md
    ├── P50_P95_P99_Grounded_Case_Studies_for_FDE_Students.docx
    ├── latency_report.json              # Exported measurements
    ├── project_analysis.md              # Detailed written analysis
    └── .env                             # Local credentials (do not commit)
```

---

## 8. Getting Started

### Prerequisites

- Python 3.12 (or Google Colab / any Jupyter runtime)
- An Azure OpenAI / Azure AI Foundry resource with a deployed chat model: endpoint, key and **deployment name**

### Setup

```powershell
cd Post-Transformation
py -3.12 -m venv .venv
.venv\Scripts\pip install openai numpy matplotlib jupyter
```

Set `AOAI_ENDPOINT`, `AOAI_KEY` and `AOAI_DEPLOYMENT` in the environment (or in `.env`, loaded into the environment before launching), or enter them at the notebook's prompts.

### Run

```powershell
.venv\Scripts\jupyter notebook DecisionStream_Day16_Latency_Colab.ipynb
```

Run the cells top to bottom. Leaving the endpoint blank runs a fully offline simulated mode.

---

## 9. Results & Learnings

- **Key result:** The "P95 < 5 s" target fits the interactive path (P95 527 ms) and cannot be met on ingestion (P95 about 32 s, with document extraction alone about 9 s at the median).
- **What worked:** Bootstrapping the real measured tail, never-raise measurement functions, and comparing both paths before recommending any optimisation.
- **What didn't:** 30 samples give a P95 resting on the top one or two calls, so it is too thin to quote in a contract.
- **Recommendations:** State the NFR as two numbers. Use P95 on the handler read path over a rolling hour, and a separately budgeted ingestion P95 measured daily. Alert on per-path P95 and tail-event rate, never the mean, and track correctness metrics alongside latency.

> Caveats: Only the generation stage is measured against a real endpoint. The other pipeline stages are modelled with course-supplied parameters. The percentile figures in `latency_report.json` differ from those printed in the notebook's embedded outputs (for example generation P50 1.63 s vs. 2.25 s), so confirm which run is canonical before quoting either externally.

---

## 10. Next Steps

- [ ] Re-run with a larger sample (100+ calls) for a trustworthy P95
- [ ] Replace modelled stage parameters with real timings from each service
- [ ] Reconcile the notebook's embedded outputs with `latency_report.json`
- [ ] Wire per-path P95 and tail-rate alerts into monitoring

---

## 11. Contact

Adi Raj · adiraj2503@gmail.com
