# ecoMAS

ecoMAS is a compact single-path multi-agent system for benchmark experiments.
It keeps the orchestration idea from `ChatDev/puppeteer`, but removes external
tools and path branching:

- every specialist agent calls the same frozen base LLM by default;
- the active dataset decides a fixed agent pool and fixed step budget;
- the orchestrator selects exactly one agent per step with `argmax`;
- each dataset owns a Puppeteer-compatible frozen Qwen state encoder plus trainable MLP router;
- runs can be executed in train or inference mode.

Inference supports deterministic greedy routing (the default) or categorical
sampling from the MLP policy. Repeated runs can be requested to estimate an
empirical final-answer distribution:

```bash
python main.py run --task mmlu_pro --router-mode sample --num-samples 32 --seed 0
```

The helpers in `ecomas.uncertainty` compute answer frequencies and second-order
Tsallis entropy, `1 - sum_y p(y)^2`. Paired orchestration/execution
decomposition is a separate next layer.

## Default Model

The default LLM is local Qwen2.5-7B-Instruct:

```text
/data2/guoyuhan/qwen_semantic_clustering_feasibility/.hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

The router state encoder always uses the Puppeteer-compatible Qwen path:
`apply_chat_template(..., add_generation_prompt=False)` followed by the last
valid token hidden state. There is no hash or mean-pooling fallback.
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
Puppeteer policy checkpoints can be loaded directly with `--checkpoint`.
New EcoMAS checkpoints use the same `model_state_dict`, `input_dim`, and
`output_dim` format.
