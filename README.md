# ecoMAS

ecoMAS is a compact single-path multi-agent system for benchmark experiments.
It keeps the orchestration idea from `ChatDev/puppeteer`, but removes external
tools and path branching:

- every specialist agent calls the same frozen base LLM by default;
- the active dataset decides a fixed agent pool and fixed step budget;
- the orchestrator selects exactly one agent per step with `argmax`;
- every dataset uses a frozen Nemotron-70B state encoder plus a trainable MLP router;
- runs can be executed in train or inference mode.

Inference supports deterministic greedy routing (the default) or categorical
sampling from the MLP policy. Repeated runs can be requested to estimate an
empirical final-answer distribution:

```bash
python main.py run --task mmlu_pro --router-mode sample --num-samples 32 --seed 0
```

The helpers in `ecomas.uncertainty` compute semantic-cluster frequencies and
second-order Tsallis entropy, `1 - sum_s p(s)^2`. The `explain` command runs the
paper's paired C/R/Z continuations and attributes this uncertainty to each
orchestration decision and agent execution.

```bash
python main.py explain --task mmlu_pro --sampling-budget 10 --seed 0
```

The explanation artifact contains the system-level U-statistic, per-step
orchestration and agent-execution estimates, semantic-boundary frequencies,
the finite-sample closure difference, and the main/branch outputs needed to
audit the estimate. Conditional continuations use fresh randomness. Real-Qwen
generation defaults to one worker because concurrent calls into one shared
model are not thread-safe. `explain` and `train` default to temperature `0.7`
so resampled agent executions are genuinely stochastic; ordinary `run` keeps
the deterministic temperature-`0.0` default.

## Calibration

Calibration training reuses the main trajectories and C/R branches from the
explanation estimator. For each input and trajectory `b`, it builds the
leave-one-out semantic distribution `q_-b`, compares it with that input's true
answer distribution `q*`, and applies the paper's `2/B` stepwise policy-gradient
estimator. MMLU-Pro and MATH-500 use point-mass targets; ChaosNLI uses its human
label distribution when available. Invalid outputs occupy an explicit semantic
cluster instead of being dropped or causing label-normalization failures.

```bash
python main.py train --task chaosnli --loss calibration \
  --calibration-samples 10 --epochs 1 --seed 0
```

After calibration, the CLI evaluates the pre- and post-training routers on the
same seeded draws and writes `calibration_report.json`. The primary calibration
metric is the mean per-input squared semantic-distribution error; Brier score
and cross entropy also support soft true-answer distributions.

## Default Model

The default LLM is local Qwen2.5-7B-Instruct:

```text
/data2/guoyuhan/qwen_semantic_clustering_feasibility/.hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

The router state encoder defaults to:

```text
/home_bak/guoyuhan/models/Llama-3.1-Nemotron-70B-Reward-HF
```

Its Llama configuration has `hidden_size=8192`, so the Router MLP is built as
`8192 -> 512 -> 128 -> 32 -> num_agents`. The encoder uses
`apply_chat_template(..., add_generation_prompt=False)` followed by the last
valid token hidden state. There is no hash or mean-pooling fallback. The
specialist generation model remains Qwen2.5-7B-Instruct; encoder and generator
paths are deliberately independent.
Specialist generation uses Puppeteer's 2048-token context and a configurable
generation limit that defaults to 768 tokens.

## Output Integrity

EcoMAS treats answer extraction as part of the experimental protocol rather
than as a permissive text heuristic. MMLU-Pro accepts only A-J, and ChaosNLI
accepts only entailment, neutral, or contradiction. Math answers must be
explicitly marked or boxed and prompt placeholders are rejected. Invalid text
is normalized to one task-specific `invalid::<task>` outcome so unrelated
prompt fragments cannot create artificial answer clusters or uncertainty.

Every step records `parse_valid`, `protocol_valid`, and `parse_error` so a
reliably extractable answer remains distinguishable from exact output-format
compliance. A run also records whether the terminal step was parseable and
protocol-compliant, which step supplied the final valid answer, and whether the
documented last-valid-answer fallback was used. Previous-agent evidence is
passed as bounded structured text instead of nested raw Python lists,
preventing prompt instructions and placeholders from recursively leaking into
later agent outputs.

CLI summaries report task accuracy together with final/terminal/step parse
rates, exact protocol-compliance rates, fallback rates, accuracy among parsed
outputs, and the answer-matching methods used. Accuracy and output integrity
must be interpreted together; neither metric substitutes for the other.

## Quick Checks

```bash
/home_bak/guoyuhan/Code/ChatDev/.conda/bin/python main.py list-tasks
/home_bak/guoyuhan/Code/ChatDev/.conda/bin/python main.py run --task mmlu_pro --split validation --limit 1 --llm-backend mock
/home_bak/guoyuhan/Code/ChatDev/.conda/bin/python main.py train --task mmlu_pro --split validation --limit 2 --llm-backend mock
```

## Real Qwen Run

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
/home_bak/guoyuhan/Code/ChatDev/.conda/bin/python main.py run \
  --task mmlu_pro --split validation --limit 1
```

Results are written to `runs/<task>/<mode>/<timestamp>/results.jsonl`.

Each inference run also writes `semantic_clusters.json`. EcoMAS first collects
all final answers for each question, maps them through one symbolic semantic
interface with task-specific adapters, builds the complete bidirectional
entailment/equivalence matrices, and only then freezes complete-link clusters.
MMLU-Pro and ChaosNLI use symbolic task labels; MATH-500 uses structured SymPy
representations with deterministic normalized-text symbols as a total fallback.
The paired sampler uses the resulting question-local same-cluster kernel after
all main and C/R/Z continuation outputs have been collected.
Router checkpoints store `model_state_dict`, `input_dim`, `output_dim`, and the
encoder model metadata. Checkpoints created for the former Qwen encoder have a
different input dimension and are rejected; train a new Router for the
8192-dimensional Nemotron embeddings.
