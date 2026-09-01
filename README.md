# ecoMAS

ecoMAS is a compact single-path multi-agent system for benchmark experiments.
It keeps the orchestration idea from `ChatDev/puppeteer`, but removes external
tools and path branching:

- every specialist agent calls the same frozen base LLM by default;
- the active dataset decides a fixed agent pool and fixed step budget;
- the orchestrator selects exactly one agent per step with `argmax`;
- each dataset owns an independent frozen encoder plus trainable MLP router;
- runs can be executed in train or inference mode.

## Default Model

The default LLM is local Qwen2.5-7B-Instruct:

```text
/data2/guoyuhan/qwen_semantic_clustering_feasibility/.hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

You can replace it with `--llm-model-path` or use `--llm-backend mock` for fast
pipeline tests that do not load a model.

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
Router checkpoints are written to `checkpoints/<task>/router.pt`.
