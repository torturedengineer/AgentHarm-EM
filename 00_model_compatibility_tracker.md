# Model / Adapter Compatibility Tracker

Fill this in as you confirm things. Purpose: make sure the base model you run
through AgentHarm is the SAME checkpoint (same revision, same instruct-tune)
that Turner et al.'s EM LoRA adapter was trained on, AND is a model AgentHarm
has published reference numbers for (so your refusal-rate sanity check in
Step 1 is meaningful).

## A. Betley et al. (2025) / Turner et al. (2025) — EM model organisms

| Base model (exact HF repo id) | Revision/commit (if pinned) | EM LoRA adapter HF repo id | Adapter rank | Confirmed loads with PEFT? (Y/N + date) | Notes |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |
| | | | | | |

Where to check:
- Turner et al. paper / repo — look for the exact base model + adapter links they publish.
- Betley et al. (2025) "Emergent Misalignment" paper/repo — cross-check which base
  models their original narrow fine-tuning used, in case you want to compare
  against their original (non-LoRA) organisms too.

## B. AgentHarm (Andriushchenko et al., 2024) — models with published reference numbers

Official eval code: https://github.com/UKGovernmentBEIS/inspect_evals (see `src/inspect_evals/agentharm`)
Dataset: https://huggingface.co/datasets/ai-safety-institute/AgentHarm

| Model | Reported refusal rate (direct, no jailbreak) | Reported HarmScore | Source (paper table / page) |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 93.9% | TODO | TODO |
| Llama-3.1-70B-Instruct | 70.5% | TODO | TODO |
| Llama-3.1-405B-Instruct | 88.3% | TODO | TODO |
| Qwen2.5-7B-Instruct | TODO | TODO | TODO |
| GPT-4o | TODO | 48.4%→54.9% (see paper note on release subset) | TODO |
| Claude 3.5 Sonnet | TODO | TODO | TODO |

## C. Overlap check (the actual thing you need)

For each candidate base model, confirm ALL THREE:
1. [ ] It (or an identical revision) appears in table B, so you have a reference
       refusal-rate/HarmScore number to sanity-check your harness against in Step 1.
2. [ ] Turner et al. published an EM LoRA adapter for exactly this base model/revision.
3. [ ] The adapter loads cleanly with PEFT on top of that base model with no
       shape/tokenizer mismatches (record the test in table A).

Only once a row satisfies all three should you use it as your primary
EM-vs-base comparison pair.
