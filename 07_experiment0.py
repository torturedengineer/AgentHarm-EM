"""
Experiment 0: Can the model produce AgentHarm trajectories at all?

No GPT-4o judge dependency, no 176 tasks, no benign benchmark, no parallel
execution, no eval_set()/resumability machinery. Just: one model, one
harmful sample, straight generation, then print the raw trajectory.

Why this works without a functioning OpenAI key: AgentHarm's own task
config sets score_on_error=False, meaning if grading throws (broken
openai package, no key, whatever), the SAMPLE still completes and its
transcript is still real and readable — only the score is missing. So
this experiment is genuinely decoupled from all the judge/package drama.

If this works: the harness and model can produce coherent tool-calling
trajectories, and the problem living in every earlier failed run really
was the scoring/orchestration plumbing, not the core thing you're
actually trying to measure.

If this DOESN'T work: the problem is real and lives in generation itself
(model + chat template + tool schema), and now you know exactly where to
look instead of debugging blind through 5 layers of eval_set machinery.
"""

import os
import torch
from transformers import BitsAndBytesConfig
from inspect_ai import eval
from inspect_evals.agentharm import agentharm

BASE_DIR = "/content/drive/MyDrive/AgentHarm-EM"
LOG_DIR = os.path.join(BASE_DIR, "logs", "experiment0")
os.makedirs(LOG_DIR, exist_ok=True)

MODEL = "hf/meta-llama/Llama-3.1-8B-Instruct"  # swap to Qwen etc. freely


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # eval() — not eval_set() — one task, one sample, no retries, no
    # parallel tasks, no fancy resumability. Returns a list of EvalLog
    # objects directly in memory.
    logs = eval(
        agentharm,  # harmful task only — agentharm_benign not even imported
        model=MODEL,
        model_args=dict(quantization_config=bnb_config),
        limit=1,  # exactly one sample
        temperature=0.01,  # avoids the temperature=0.0 hf-provider crash
        log_dir=LOG_DIR,
        log_dir_allow_dirty=True,
    )

    log = logs[0]
    print(f"\nStatus: {log.status}")

    if log.status == "error":
        print(f"Error: {log.error.message}")
        print("\n(Generation itself failed — this is the real problem to debug, independent of any judge/OpenAI issues.)")
        return

    if not log.samples:
        print("No samples in log — task didn't produce anything. Check the error above.")
        return

    sample = log.samples[0]
    print(f"\n{'=' * 70}")
    print(f"Trajectory for sample {sample.id}")
    print(f"{'=' * 70}")

    for msg in sample.messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"\n--- {msg.role} ---")
        print(content.strip())
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"  [tool call: {tc.function}({tc.arguments})]")

    if sample.scores:
        print(f"\n--- Score ---\n{sample.scores}")
    else:
        print("\n(No score recorded — grading was skipped or failed, but the")
        print(" trajectory above is real and doesn't depend on that at all.)")


if __name__ == "__main__":
    main()
