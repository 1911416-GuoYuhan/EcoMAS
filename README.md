# Explaining and Calibrating Dynamically Orchestrated LLM MAS through Stepwise Uncertainty Decomposition

> **EcoMAS** — Explanation and Calibration through Orchestration for Multi-Agent Systems

EcoMAS is a framework for explaining and calibrating dynamically orchestrated large-language-model multi-agent systems. It treats the final answer as a semantic random variable, measures its uncertainty with second-order Tsallis entropy, and attributes that uncertainty to each orchestration decision and each selected agent's execution. The same paired samples used for explanation are then reused to optimize the orchestration policy toward the reference answer distribution, connecting interpretability and calibration in one workflow.

---

## 1. Overview

### 1.1 The Problem

A dynamically orchestrated multi-agent system evolves through a sequence of dependent decisions. At step $t$, an orchestration policy selects an agent from the current context, the selected agent generates a response, and that response changes the context for every later step. Two distinct sources of randomness are therefore nested throughout execution:

| Source | Random variable | Practical question |
|---|---|---|
| **Orchestration** | Which agent is selected? | How much does selecting a particular specialist change the final answer distribution? |
| **Agent execution** | What response does the selected agent produce? | How much does the generated reasoning or answer revision change the final answer distribution? |

Local variability at one step is not sufficient to answer either question because its effect depends on all possible downstream continuations. Directly enumerating those continuations grows exponentially with the execution horizon. At the same time, optimizing only single-answer accuracy does not ensure that the system assigns appropriate probability mass across semantically distinct answers, especially when the reference itself represents disagreement.

### 1.2 The EcoMAS Approach

EcoMAS addresses these challenges through three connected components:

1. **Semantic uncertainty.** Outputs with the same meaning are grouped into one semantic cluster. If $q(s)$ is the probability of cluster $s$, system uncertainty is measured as

   $$
   H(S)=1-\sum_{s}q(s)^2.
   $$

   This quantity is also the probability that two independent system executions produce semantically different answers.

2. **Stepwise uncertainty decomposition.** Final semantic uncertainty is decomposed into non-negative information contributions from orchestration and agent execution at every step. The decomposition explains whether uncertainty is resolved by choosing an agent, by that agent's response, or by later reasoning.

3. **Attribution-driven calibration.** EcoMAS reuses the semantic changes collected for explanation to estimate a calibration gradient. A leave-one-out construction separates the estimated system distribution from the trajectory whose gradient term is being evaluated, allowing the Router to be optimized toward the true-answer distribution without additional continuation calls for the same batch.

### 1.3 Repository Scope

The repository provides the executable EcoMAS pipeline for three complementary benchmarks:

| Task key | Benchmark | Output space | Steps | Specialists |
|---|---|---|---:|---:|
| `mmlu_pro` | MMLU-Pro | Options A–J | 4 | 5 |
| `math500` | MATH-500 | Open-ended mathematical answers | 6 | 5 |
| `chaosnli` | ChaosNLI | Entailment, neutral, contradiction | 5 | 5 |

For each task, a frozen generation model produces specialist responses, a frozen encoder embeds the evolving execution context, and a trainable MLP Router selects one specialist at each step. Semantic clustering is task-aware: option identity is used for MMLU-Pro, symbolic equivalence is used for MATH-500, and label identity is used for ChaosNLI.

The implementation includes paired C/R/Z continuation sampling, second-order Tsallis uncertainty estimation, stepwise orchestration and agent-execution attribution, semantic calibration training, checkpoint validation, and structured experiment artifacts. The paper's proofs and finite-sample concentration bounds are theoretical results and are not implemented as runtime procedures.

### 1.4 Main Empirical Findings

The paper evaluates EcoMAS from four complementary perspectives:

| Research question | Finding |
|---|---|
| **Estimation accuracy** | Paired sampling closely tracks exact contributions on finite execution trees, with error decreasing as the sampling budget grows. |
| **Interpretability** | Stepwise attributions distinguish specialist selection, answer correction, and later reinforcement, including reinforcement of incorrect answers. |
| **Faithfulness** | Making highly ranked orchestration and execution variables deterministic reduces remaining uncertainty more effectively than competing rankings across the evaluated tasks and model backbones. |
| **Calibration** | Reusing attribution samples for Router optimization reduces semantic calibration error while generally improving task accuracy. |

These findings support the paper's central claim: the same conditional semantic changes can explain how a collective answer emerges and provide a principled signal for improving its distributional reliability.

---

## 2. Installation and Configuration

### 2.1 Requirements

- Python 3.10 or later
- PyTorch with a build compatible with the target accelerator
- A local Hugging Face causal language model for specialist generation
- A local Hugging Face encoder whose hidden size matches the Router checkpoint
- Sufficient accelerator memory for both frozen models

Create an isolated environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers accelerate pandas pyarrow sympy tqdm
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

> [!NOTE]
> Install the PyTorch build appropriate for the local CUDA or CPU environment when the generic installation command is not suitable.

### 2.2 Configure Paths and Defaults

All adjustable settings are centralized in [`config.json`](config.json). Replace the anonymous path placeholders before running the system:

```json
{
  "paths": {
    "benchmark_root": "<PATH_TO_BENCHMARKS>",
    "generation_model": "<PATH_TO_GENERATION_MODEL>",
    "encoder_model": "<PATH_TO_ENCODER_MODEL>"
  }
}
```

The same file controls generation parameters, encoder limits, Router dimensions, sampling seeds, optimization defaults, dataset layouts, task horizons, specialist roles, and prompt protocols. Command-line arguments override the corresponding runtime defaults without modifying the file.

The default Router expects an encoder output dimension of `8192` and uses the following architecture:

```text
8192 → 512 → 128 → 32 → number of specialists
```

If a different encoder is used, update `encoder.output_dimension` and supply compatible pretrained Router checkpoints.

### 2.3 Prepare Benchmark Data

Arrange the benchmark root according to one of the supported layouts:

```text
<PATH_TO_BENCHMARKS>/
├── mmlupro/
│   ├── validation.parquet
│   └── test.parquet
├── math500/
│   └── test.jsonl
└── chaosnli/
    ├── chaosNLI_mnli_m.jsonl
    ├── chaosNLI_snli.jsonl
    ├── chaosNLI_alphanli.jsonl
    └── split_60_20_20/
        ├── train.jsonl
        ├── validation.jsonl
        └── test.jsonl
```

The configured fallback directories also support derived MMLU-Pro and MATH-500 splits. Exact directory names and file extensions can be changed in `config.json`.

### 2.4 Prepare Pretrained Orchestrators

Uncertainty decomposition and calibration training always initialize from the dataset-specific checkpoint:

```text
pretrained/
├── mmlu_pro/router.pt
├── math500/router.pt
└── chaosnli/router.pt
```

The repository includes structurally valid dummy checkpoints so the checkpoint-loading contract can be inspected without publishing trained parameters. They contain zero-valued Router weights and consequently produce a uniform initial distribution over the five specialists.

> [!IMPORTANT]
> Dummy checkpoints are placeholders, not pretrained research artifacts. Replace them with the corresponding trained orchestrators before producing experimental results. EcoMAS stops immediately when a required checkpoint is missing or incompatible with the configured encoder dimension or specialist pool.

### 2.5 Verify the Setup

List the registered tasks without loading either model:

```bash
python main.py list-tasks
```

For every full run, EcoMAS records the task, split, output location, initialized checkpoint, backends, routing path, generation seeds, parsing status, and semantic outputs needed for later analysis.

---

## 3. Running Uncertainty Decomposition

### 3.1 Purpose

The `explain` workflow estimates:

- total semantic system uncertainty;
- the information contribution of orchestration at each step;
- the information contribution of agent execution at each step;
- their stepwise sum and finite-sample closure difference;
- semantic-boundary probabilities for both contribution types;
- the main trajectories and conditional branch outputs supporting the estimates.

For each input, EcoMAS samples $B$ main trajectories. At every execution step it constructs three conditional continuations while sharing the relevant execution history:

| Branch | Retained state | Resampled variables |
|---|---|---|
| **C** | Context before the current decision | Current agent selection, current response, and the remaining suffix |
| **R** | Context and selected agent | Current agent response and the remaining suffix |
| **Z** | Context, selected agent, and current response | Remaining suffix only |

Comparisons among the main output and the C/R/Z outputs isolate the orchestration and agent-execution contributions without exhaustive enumeration of the full continuation tree.

### 3.2 Basic Command

```bash
python main.py explain \
  --task mmlu_pro \
  --split validation \
  --limit 10 \
  --sampling-budget 10 \
  --seed 0
```

Equivalent examples for the other tasks are:

```bash
python main.py explain --task math500 --split test --limit 10 --sampling-budget 10 --seed 0
python main.py explain --task chaosnli --split mnli_m --limit 10 --sampling-budget 10 --seed 0
```

### 3.3 Key Options

| Option | Meaning |
|---|---|
| `--task` | One of `mmlu_pro`, `math500`, or `chaosnli` |
| `--split` | Dataset split or configured ChaosNLI source subset |
| `--limit` | Maximum number of examples to process |
| `--sampling-budget` | Number of main paired-sampling trajectories per example; must be at least 2 |
| `--seed` | Root seed for reproducible routing and generation draws |
| `--parallel-workers` | Number of concurrent continuation workers; use `1` when a shared local model is not thread-safe |
| `--device` | Runtime device selection, such as `auto`, `cuda`, or `cpu` |

Larger sampling budgets reduce Monte Carlo error but require proportionally more model calls. The paper derives the estimator's finite-sample behavior; this repository reports the empirical estimate and its closure diagnostic rather than evaluating a theorem at runtime.

### 3.4 Outputs

Results are written under:

```text
runs/<task>/explain/<timestamp>/
├── results.jsonl
└── explanation.json
```

`results.jsonl` contains the sampled execution records. `explanation.json` contains one report per input with the principal fields below:

| Field | Interpretation |
|---|---|
| `system_uncertainty` | Estimated second-order Tsallis uncertainty of the final semantic answer |
| `orchestration[t]` | Contribution from the Router decision at step `t` |
| `agent_execution[t]` | Contribution from the selected agent's generation at step `t` |
| `stepwise_total[t]` | Sum of both contributions at step `t` |
| `total_stepwise` | Sum of all estimated stepwise contributions |
| `closure_difference` | Finite-sample difference between total stepwise contribution and system uncertainty |
| `main_outputs` | Final answers from the main trajectories |
| `branch_outputs` | Conditional C/R/Z continuation answers used by paired sampling |

A positive contribution means that observing that decision or response resolves final-output uncertainty. It does not by itself imply correctness: a step can strongly concentrate the system on an incorrect semantic answer.

---

## 4. Running Calibration Training

### 4.1 Purpose

Calibration training optimizes the Router so that the induced semantic output distribution approaches the reference answer distribution. For MMLU-Pro and MATH-500, the target is the correct-answer point mass. For ChaosNLI, human label distributions are used when present in the dataset metadata.

The default `calibration` objective reuses the paired explanation samples. For each trajectory, EcoMAS estimates the system semantic distribution from the other $B-1$ trajectories and combines this leave-one-out estimate with the sampled conditional semantic increment. This prevents the trajectory from being multiplied by an estimate containing itself and allows the explanation batch to provide the Router gradient without another set of continuation calls.

### 4.2 Basic Command

```bash
python main.py train \
  --task mmlu_pro \
  --split validation \
  --test-split test \
  --limit 300 \
  --test-limit 100 \
  --loss calibration \
  --calibration-samples 10 \
  --epochs 1 \
  --lr 0.001 \
  --seed 0
```

Equivalent task-specific starting points are:

```bash
python main.py train --task math500 --split test --loss calibration --calibration-samples 10 --epochs 1
python main.py train --task chaosnli --split train --test-split test --loss calibration --calibration-samples 10 --epochs 1
```

### 4.3 Training Objectives and Options

| Option | Meaning |
|---|---|
| `--loss calibration` | Optimize semantic distribution calibration with the paired leave-one-out estimator |
| `--loss accuracy` | Optimize trajectory-level task reward |
| `--loss mixed` | Combine calibration training with the configured accuracy term |
| `--calibration-samples` | Number of paired trajectories per training example; must be at least 2 for calibration objectives |
| `--epochs` | Number of passes over the selected training examples |
| `--lr` | Router learning rate |
| `--test-split` | Split used for matched pre-/post-training calibration evaluation |
| `--test-limit` | Maximum number of evaluation examples |

All specialist-model and encoder parameters remain frozen. Only the MLP Router is updated.

### 4.4 Checkpoint and Report Lifecycle

Training artifacts are written to two locations:

```text
checkpoints/<task>/router.pt
runs/<task>/train/<timestamp>/
├── results.jsonl
└── calibration_report.json
```

The saved Router checkpoint includes the model state, input dimension, output dimension, task identifier, ordered specialist names, and encoder metadata. `calibration_report.json` compares the pretrained and optimized Router on matched seeded draws and reports distributional calibration and task-quality diagnostics.

> [!CAUTION]
> Training does not overwrite the initialization file under `pretrained/`. The optimized checkpoint is written under `checkpoints/`, preserving the original initialization for reproducible reruns.

---

## 5. Reproducibility and Interpretation

### Reproducibility Checklist

- Use the same `config.json`, benchmark splits, model checkpoints, and task-specific pretrained Router.
- Record the command-line overrides, seed, sampling budget, and number of workers.
- Keep the generation temperature stochastic for decomposition and calibration; deterministic generation removes an intended source of agent-execution uncertainty.
- Use `--parallel-workers 1` unless the generation backend is safe for concurrent calls.
- Compare experiments with identical semantic parsing and clustering rules.
- Treat dummy checkpoints and mock-model runs as integration checks only, never as benchmark evidence.

### Reading the Results Responsibly

EcoMAS explains how randomness changes the final semantic answer distribution. It does not equate certainty with correctness. A low-uncertainty system may be confidently wrong, and a large information contribution may correspond to either a beneficial correction or reinforcement of an error. The most informative analysis therefore considers stepwise contributions together with the trajectory text, semantic answer distribution, task accuracy, and calibration error.

---

## 6. Summary

EcoMAS provides a unified way to study and improve dynamically orchestrated LLM multi-agent systems:

| Capability | Result |
|---|---|
| **Semantic uncertainty modeling** | Measures substantive answer disagreement rather than surface-form variation |
| **Stepwise decomposition** | Separates information contributed by agent selection and agent execution |
| **Efficient estimation** | Uses paired continuations instead of enumerating complete execution trees |
| **Faithful diagnosis** | Identifies the stages that most strongly shape final-output uncertainty |
| **Calibration training** | Reuses explanation samples to optimize the orchestration policy |

The central idea is that explanation and optimization need not be separate procedures. The semantic increments that reveal how a collective answer emerges can also provide the learning signal that makes the system's answer distribution more reliable.
