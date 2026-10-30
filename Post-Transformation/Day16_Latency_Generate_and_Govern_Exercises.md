# Day 16 — Latency: Generate and Govern

**Birlasoft FORGE FDE Academy · Sprint 1 Day 16 · Learner Handout**

---

## How this works

You are not going to write this code. You are going to **specify it, read what
comes back, and find what is wrong with it.**

That is not a shortcut. On a client site you will review far more inference and
measurement code than you write, and a model will hand you something plausible
in four seconds. The skill being practised is the one that matters: **knowing
what good looks like well enough to notice when you have not got it.**

Each exercise gives you:

| | |
|---|---|
| **The brief** | What the cell has to achieve. Read this first. |
| **The prompt** | Paste into Gemini. Change it if you can do better. |
| **Accept it when** | Your checklist before you move on. |
| **It will probably get wrong** | Do not read this until you have reviewed the output yourself. |

> **Work in that order.** Reading the failure list first turns a review
> exercise into a spot-the-difference puzzle, and you learn nothing.

**In Colab:** use the Gemini panel, or `Ctrl+Shift+Space` in a code cell.

---

## Setup — before Exercise 1

Run this once by hand. It is not part of the exercise.

```python
%pip install -q openai numpy matplotlib
import os, time, math, getpass, json
import numpy as np, matplotlib.pyplot as plt

ENDPOINT   = input("Azure OpenAI endpoint: ").strip()
KEY        = getpass.getpass("Key (hidden): ").strip()
DEPLOYMENT = input("Deployment name: ").strip()

from openai import AzureOpenAI
client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=KEY,
                     api_version="2024-10-21", max_retries=0, timeout=90.0)

SYSTEM = ("You assess motor insurance claims. Answer only from the case text. "
          "Every figure carries its source. Never state a decision.")
CASE = """Claim CLM-2026-4471. Front nearside collision, 14 March 2026 (page 1).
Repair estimate GBP 12,400.00 (page 7). Policy limit GBP 25,000.00, excess
GBP 350.00 (page 3). Supplementary estimate GBP 14,750.00 (page 11)."""
print("ready")
```

**Note `max_retries=0`.** That is deliberate and you should be able to say why
by the end of Exercise 2.

---

# Exercise 1 · Measure one call properly

**15 minutes**

### The brief

A function that makes one chat completion call and returns how long it took,
what came back, and how many tokens were involved. Every later exercise builds
on this, so a defect here propagates through all of them.

The hard part is not timing. It is **what happens when the call fails**.

### The prompt

```
Write a Python function `one_call(max_tokens=220, task="...")` for a Colab
notebook that measures Azure OpenAI chat completion latency.

Context: `client` is an existing AzureOpenAI instance. `SYSTEM` and `CASE` are
existing strings. `DEPLOYMENT` is the deployment name.

Requirements:
- Return a tuple: (elapsed_ms, response_text, usage_dict)
- Time the whole round trip, starting before the request is constructed
- Use time.perf_counter(), not time.time()
- usage_dict must include prompt_tokens, completion_tokens, and the model
  string the API actually returned
- temperature=0.0
- The function must NEVER raise. If the call fails, return the elapsed time,
  an empty string, and a usage_dict containing an "error" key with the
  exception type and a truncated message.

Explain in comments why a failed call still returns its elapsed time.
```

### Accept it when

- [ ] It uses `time.perf_counter()`
- [ ] The timer starts **before** the request is built, not after
- [ ] A failure returns a value rather than raising
- [ ] A failed call still reports elapsed time
- [ ] It reads `response.model`, not just echoing `DEPLOYMENT` back
- [ ] Nothing prints the key

### It will probably get wrong

> Read only after your own review.

- **`time.time()` instead of `perf_counter()`.** Wall-clock time is subject to
  NTP adjustment. It is usually fine and occasionally gives you a negative
  duration, which is a confusing afternoon.
- **A bare `try/except: pass`** that swallows the error and returns nothing.
  You then have a silent hole in your distribution and no idea it is there.
- **Not returning elapsed time on failure.** A call that times out after 90
  seconds is your most interesting data point. Discarding it makes your P95
  look better than reality, which is the wrong direction to be wrong in.
- **Reporting `DEPLOYMENT` as the model.** Auto-upgrade means the version that
  answered may not be the one you think you deployed. Log what the API said.

---

# Exercise 2 · The distribution, and the tail

**20 minutes**

### The brief

Call it N times and characterise the result. This is the cell that decides
whether the rest of the session lands, because **it is where people discover
that latency is not a bell curve.**

### The prompt

```
Write a Colab cell that calls `one_call()` 30 times sequentially and
characterises the latency distribution.

Requirements:
- Print a running line per call showing: call number, elapsed ms, completion
  tokens, and the running P50, P95 and maximum so far
- Count failures separately and keep going — do not abort the run
- Exclude failed calls from the percentile maths, but report how many there were
- At the end, print n, errors, min, mean, P50, P95, P99 and max
- Store the successful samples in a numpy array called `gen`
- Use np.percentile

Then add a short printed comment stating what percentile the MEAN sits at, and
what that implies for alerting.

Do not use retries anywhere.
```

### Accept it when

- [ ] Failures are counted, reported, and excluded from percentiles
- [ ] A failure does not abort the loop
- [ ] It reports P95 and P99, not just mean and P50
- [ ] `gen` holds only successful samples
- [ ] It states where the mean falls in the distribution

### It will probably get wrong

- **Reporting mean and standard deviation as the headline.** Standard deviation
  is a bell-curve statistic. Applied to a lognormal-with-tail it tells you
  almost nothing, and quoting `mean ± σ` to a client is how a P95 commitment
  gets made by accident.
- **Including failed calls as zero.** This drags every percentile down. If you
  see it, ask what your P95 would have been.
- **Adding `time.sleep()` between calls** "to be polite". Reasonable instinct,
  but it changes what you are measuring — you have just removed any chance of
  observing throttling behaviour under load.
- **Silently re-enabling retries.** The SDK retries by default. A retry turns
  one 12-second tail event into one 2-second success and you never see it. That
  is why `max_retries=0` is in the setup.

### Then answer

With 30 samples, your P95 is estimated from roughly the **top one or two
calls**. How much would you trust that number in a contract? What sample size
would you want, and what would it cost?

---

# Exercise 3 · Streaming, and what it actually buys

**20 minutes**

### The brief

Measure two numbers on the same calls: **time to first token** and **time to
completion**. The gap between them is the entire argument for streaming.

### The prompt

```
Write a Colab cell that measures streaming latency against Azure OpenAI.

Requirements:
- A function `one_stream(max_tokens=220)` returning
  (ttft_ms, total_ms, n_content_chunks)
- Use stream=True
- Time to first token must be measured at the first chunk that contains actual
  CONTENT, not the first chunk received — some chunks carry only role or
  filter metadata and contain no text
- Guard against chunks with an empty `choices` list
- Run it 8 times, print a table of ttft / total / chunks per run
- Print P50 and P95 for both ttft and total
- Finish with a printed comparison showing first-visible-text with and without
  streaming, and the ratio between them

Add a comment explaining what streaming does NOT improve.
```

### Accept it when

- [ ] TTFT is measured at first **content**, not first chunk
- [ ] Empty `choices` lists do not crash it
- [ ] Both TTFT and total are reported at P50 and P95
- [ ] The output makes clear that total time is unchanged

### It will probably get wrong

- **Timing the first chunk of any kind.** Azure often sends an opening chunk
  carrying a role and no content, and on filtered deployments a prompt-filter
  chunk before that. Timing those gives you a flatteringly small TTFT that has
  nothing to do with when a human sees text.
- **`chunk.choices[0]` without checking the list is non-empty.** This throws an
  `IndexError` partway through a run, and because it is intermittent it will
  work in your demo.
- **Claiming streaming made it faster.** Nothing got faster. The reader stopped
  waiting in silence. If the generated comment says "improves latency", correct
  it — that sentence in a client document is a commitment you cannot meet.

### Then answer

Your ingestion path takes tens of seconds and nobody is watching it. Your
handler screen reads a stored profile in under a second. **On which of those
would you spend engineering effort adding streaming, and why?**

---

# Exercise 4 · What output length costs you

**15 minutes**

### The brief

Generation time is dominated by how many tokens come out. Measure the slope.
Everything in Exercise 5 depends on this being real rather than assumed.

### The prompt

```
Write a Colab cell that measures how output length affects generation latency.

Requirements:
- Test three max_tokens settings: 80, 240, 600
- For each, call one_call() 5 times with a task that explicitly asks for
  roughly the corresponding number of words
- Record median latency and MEAN ACTUAL completion tokens returned
- Print a table: max_tokens, p50 ms, actual output tokens, ms per token
- Compute the slope in ms per token between the smallest and largest settings

Important: if the actual token counts barely differ between settings, the
prompt was the binding constraint rather than max_tokens. Detect that case and
print a message saying so instead of reporting a meaningless slope.
```

### Accept it when

- [ ] It reports **actual** completion tokens, not the `max_tokens` setting
- [ ] It handles the case where output length did not actually vary
- [ ] The slope is computed from measured tokens

### It will probably get wrong

- **Plotting latency against `max_tokens`.** That is the ceiling you requested,
  not the length you got. If the model wanted to write 90 tokens it writes 90
  whether you allowed 240 or 600, and your "slope" is an artefact.
- **Reporting a slope from three noisy points without saying so.** Five samples
  per bucket is enough to see a trend and not enough to quote a coefficient.
- **Not noticing a negative slope.** If longer output came back faster, you
  have measured noise. The generated code will report it with a straight face.

---

# Exercise 5 · Compose the pipeline — and find the trap

**25 minutes · the important one**

### The brief

Combine your measured generation latency with the other pipeline stages, and
work out the end-to-end P95.

Then check the obvious shortcut, because a project manager is going to ask you
for a stage-by-stage budget and the obvious way to build one is wrong.

### The prompt

```
Write a Colab cell that composes an end-to-end pipeline latency distribution.

Stages, as (median_ms, sigma, tail_probability, tail_multiplier):
  blob_upload      (400,  0.35, 0.03, 6.0)
  doc_intelligence (9000, 0.45, 0.06, 4.5)
  redact_pii       (250,  0.30, 0.02, 3.0)
  search_retrieve  (320,  0.40, 0.05, 5.0)
  prompt_assemble  (60,   0.25, 0.01, 2.5)
  generation       -- do NOT model this; resample from my measured array `gen`
  validate_schema  (45,   0.20, 0.01, 2.0)
  audit_write      (180,  0.30, 0.03, 4.0)
  persist_profile  (140,  0.30, 0.02, 3.5)

Requirements:
- Model each stage as LOGNORMAL, plus a separate tail event that multiplies the
  duration with the given probability. Do NOT use a normal distribution.
- For generation, bootstrap by resampling with replacement from `gen`, so my
  real measured distribution including its tail flows into the composition
- 20,000 simulated runs
- Sum the stages sequentially and report pipeline P50, P95, P99
- Then compute and print three numbers side by side:
    the sum of every stage's individual P95
    the actual composed pipeline P95
    the sum of every stage's individual P50
- Print which of the two shortcuts overestimates and which underestimates,
  with the size of each error

Explain in a comment why the three numbers differ.
```

### Accept it when

- [ ] Stages are lognormal with an explicit tail event, **not** normal
- [ ] Generation is bootstrapped from `gen`, not modelled
- [ ] The three composition numbers are printed together
- [ ] The explanation is correct — see below

### It will probably get wrong

- **`np.random.normal` for stage durations.** This is the big one. A normal
  distribution puts P95 close to the mean, which makes the whole session's
  conclusion evaporate. It may also produce negative durations, which the code
  will happily sum.
- **Modelling generation instead of resampling your data.** You measured a real
  distribution. Replacing it with a fitted lognormal throws away the shape you
  paid for.
- **Explaining the composition wrongly.** A common generated answer is "because
  percentiles are not additive" — true and useless. The reason that means
  something: **the stages do not all have a bad day at the same time.** Summing
  P95s assumes they do; summing P50s assumes they never do. The truth sits
  between, and neither shortcut is a rounding error.

### Then answer

Your project manager wants a stage-by-stage latency budget so each team knows
its allowance. **What do you actually give them, and what do you warn them
about?**

---

# Exercise 6 · The two paths, and the NFR

**20 minutes · the one that goes in the ADR**

### The brief

Your session outcome says **P95 under 5 seconds for a single document**. Your
composed ingestion pipeline is nowhere near that.

Work out whether the target is wrong, the pipeline is wrong, or something else
is going on — then write the NFR that survives contact with a client.

### The prompt

```
Extend my pipeline latency analysis to compare two paths.

Path A — INGESTION: the pipeline I already composed. A PDF arrives, is
extracted, retrieved against, generated from, and the profile is stored.
Nobody is waiting on it.

Path B — INTERACTIVE: a handler opens a case and the stored profile is
rendered. Stages, as (median_ms, sigma, tail_p, tail_mult):
  authn          (45,  0.30, 0.02, 3.0)
  read_profile   (110, 0.35, 0.03, 5.0)
  read_citations (90,  0.35, 0.03, 4.0)
  render         (55,  0.25, 0.01, 2.5)

Requirements:
- Report P50, P95 and P99 for both paths
- For each, state PASS or FAIL against a 5000 ms target, and by what factor
- Plot the ingestion distribution with the 5 second target marked
- Print a suggested two-part non-functional requirement that names the path,
  the percentile, and the measurement window for each

Do not conclude that the pipeline needs optimising until you have compared the
two paths.
```

### Accept it when

- [ ] Both paths reported at P50, P95 and P99
- [ ] The suggested NFR **names the path** in both halves
- [ ] It specifies a measurement window
- [ ] It does not claim the ingestion path can be optimised to 5 seconds

### It will probably get wrong

- **Recommending optimisation to hit 5 seconds on the ingestion path.** It
  cannot be done — Document Intelligence alone is around nine seconds at the
  median. Generated advice will confidently suggest caching, parallelism and a
  smaller model. Add them all up and you are still four times over.
- **Writing an NFR with a number and no path.** "P95 under 5 seconds" is the
  sentence that causes the argument. Both halves need a named path.
- **Omitting the measurement window.** P95 over what — an hour, a day, a month?
  A P95 measured over a month can hide an entire bad afternoon.

### Then write it out

Your NFR should end up shaped like this, with your own measured numbers:

> P95 under 5 seconds on the handler read path, measured over a rolling hour.
> The ingestion path is asynchronous and budgeted per document at P95 under
> **N** seconds, measured daily.

**Two numbers, two paths, both defensible.** One number is an argument waiting
to happen.

---

# The defence — 15 minutes

Present to another pod. 

1. What percentile does your mean sit at, and what does that mean for your
   alerting?
2. Your P95 came from 30 samples. How much would you trust it in a contract?
3. Which is bigger — the sum of your stage P95s, or your pipeline P95? Why?
4. Which of your changes would move P50 and which would move P95? Are they the
   same work?
5. Which path does your NFR apply to, and could somebody who was not here today
   misread it?


---

