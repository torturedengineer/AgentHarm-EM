"""
Runs AgentHarm via Inspect AI's own `eval_set()`, which is Inspect's
built-in mechanism for exactly your problem (long runs on unreliable
compute): it logs incrementally to `log_dir` as each sample finishes, and
if you re-run this SAME script pointed at the SAME log_dir, it detects
which samples already completed and only runs the remaining/failed ones.

So: if Colab disconnects or your laptop dies mid-run, just re-run this
script unchanged. As long as BASE_DIR is on Drive (already true here),
the logs survive the runtime being recycled and the next run picks up
where it left off. You do not need the separate checkpoint_utils.py
module for this path — that module is only useful if you end up writing
a fully custom (non-Inspect) grading loop.

Everything you need to choose is marked TODO. Nothing else here should
need editing.
"""

import os
import torch
from transformers import BitsAndBytesConfig
from inspect_ai import eval_set
from inspect_evals.agentharm import agentharm, agentharm_benign  # both imported; use whichever/both you want

BASE_DIR = "/content/drive/MyDrive/AgentHarm-EM"  # keep in sync with 01_setup_colab.py
LOG_DIR = os.path.join(BASE_DIR, "logs")           # eval_set's incremental logs live here -> resumable
RESULTS_DIR = os.path.join(BASE_DIR, "results")    # anywhere you want to additionally dump derived results


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------------
    # TODO: pick your model string(s). Inspect supports many providers;
    # the two most relevant to your project:
    #
    #   Local Hugging Face model, no adapter:
    #     "hf/<org>/<repo>"
    #
    #   vLLM-served model WITH a LoRA adapter (Turner et al.'s adapters):
    #     "vllm/<base-org>/<base-repo>:<path-or-repo-to-adapter>"
    #     (requires the vllm package; see requirements.txt)
    #
    #   Hosted API model (e.g. for a quick sanity-check run):
    #     "openai/gpt-4o-2024-08-06", "anthropic/claude-3-5-sonnet-20240620", etc.
    #
    # You'll typically run this script twice per comparison: once with the
    # base model string, once with the base+adapter model string, into two
    # different log_dir subfolders (see TODO below) so results don't mix.
    # ---------------------------------------------------------------------
    model = "hf/Qwen/Qwen2.5-7B-Instruct"  # e.g. "hf/Qwen/Qwen2.5-7B-Instruct" or "vllm/Qwen/Qwen2.5-7B-Instruct:TODO-adapter-path"

    # TODO: give each distinct run (base vs EM-adapter, different splits, etc.)
    # its own log_dir subfolder so eval_set's resumability isn't comparing
    # across runs that are actually meant to be different.
    run_name = "Qwen2.5-7B-Instruct-base"  # bumped to v2 — the -run folder has several
    # botched attempts tangled in it (stale-script reruns, temperature bug,
    # broken openai package). Starting a clean folder for this smoke test
    # avoids any ambiguity about which .eval file reflects the current,
    # actually-fixed script. Once this completes cleanly, use a fresh,
    # descriptive name again for your real base-vs-EM-adapter runs.
    run_log_dir = os.path.join(LOG_DIR, run_name)

    # ---------------------------------------------------------------------
    # TODO: pick which AgentHarm task(s)/split/args to run. Examples of
    # valid task args (see inspect_evals/src/inspect_evals/agentharm):
    #   agentharm(split="val")          # validation split
    #   agentharm(chat_dataset=True)    # chat-only version of harmful tasks
    #   agentharm_benign()              # benign counterpart tasks
    # Leave both harmful + benign in the list if you want the paired
    # comparison AgentHarm's own design supports; remove whichever you don't.
    # ---------------------------------------------------------------------
    tasks = [
        agentharm,          # TODO: replace with e.g. agentharm(split="val") if you want args
        agentharm_benign,   # TODO: remove this line if you only want harmful tasks
    ]

    # NOTE: required on a T4 (15GB VRAM) for any 7-8B model. Inspect's `hf`
    # provider forwards `model_args` straight to
    # AutoModelForCausalLM.from_pretrained(...), so this is the same
    # BitsAndBytesConfig you already confirmed works, loading in 4-bit
    # instead of full precision. Without this, an 8B model's weights alone
    # (~16GB+ in bf16/fp32) don't fit in the T4's ~14.5GB usable memory —
    # which is exactly the CUDA OOM you just hit.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # eval_set returns (success: bool, logs: list[EvalLog])
    success, logs = eval_set(
        tasks=tasks,
        model=model,
        model_args=dict(
            quantization_config=bnb_config,
            # NOTE: do NOT also pass device_map here — Inspect's `hf`
            # provider sets device_map itself internally (it has its own
            # `device=` param for this, separate from model_args) and
            # passing device_map again causes:
            #   TypeError: from_pretrained() got multiple values for
            #   keyword argument 'device_map'
            # If you need to force a specific device, pass device="cuda:0"
            # as a top-level eval_set()/eval() kwarg instead, not inside
            # model_args.
        ),
        log_dir=run_log_dir,
        limit=5,  # smoke test — remove once a small run completes cleanly
        # NOTE: Inspect's default generate config uses temperature=0.0 to
        # signal greedy decoding. That's fine for API providers, but the
        # local `hf` provider forwards it straight to transformers'
        # generate(), which crashes on temperature=0.0 unless do_sample is
        # explicitly turned off — a known transformers quirk. Kept low
        # (not 0, not 1.0) since higher temperatures make the model more
        # likely to emit multiple simultaneous tool calls in one turn —
        # which Llama-3.1's chat template can't render at all (see
        # fail_on_error note below).
        temperature=0.1, #qwen 
        # Llama-3.1's official chat template does not support parallel/
        # multiple tool calls in a single turn (confirmed limitation, not
        # an Inspect/harness bug — see vLLM's tool-calling docs). AgentHarm's
        # agent loop can produce such a turn on some behaviors, which
        # crashes that one sample with TemplateError. Without this,
        # fail_on_error defaults to True and ONE such sample kills the
        # entire task. Tolerate a reasonable fraction instead — tune this
        # once you see how often it actually happens over a full run.
        fail_on_error=0.1, #qwen 0.1, llama needed 0.4
        # Retry settings: default is 10 attempts with exponential backoff
        # (30s, 60s, 120s...) — fine once things are working, but wasteful
        # while you're still debugging a real (non-transient) config bug,
        # since it'll retry the same failure 10 times before giving up.
        # Turned down here; raise this back up once a run completes cleanly.
        retry_attempts=2,
        retry_wait=10,
        # Caps retries on individual model API calls (both the hf provider
        # AND the OpenAI judge calls AgentHarm's scorer makes). Default is
        # unlimited, which is what turned a missing OPENAI_API_KEY into a
        # 30+ minute retry storm instead of a fast, clear failure.
        max_retries=3,
        # While iterating on bugs, each fixed re-run leaves the previous
        # failed attempt's .eval file behind under a different eval_set_id,
        # which Inspect refuses to mix into a fresh eval_set by default.
        # Allow it here since you're still debugging; once you're doing your
        # real, final run, use a brand-new run_name/log_dir instead so you
        # get a clean, unambiguous eval_set_id for that run's results.
        log_dir_allow_dirty=True,
        # TODO: max_tasks / max_connections left at Inspect's defaults
        # (4 and 10 respectively). Since you're on a single GPU with one
        # loaded model, the `hf` provider batches concurrent generate
        # requests together rather than spinning up separate model copies,
        # so raising/lowering these is a throughput tuning choice, not
        # something you need to fix right now.
    )

    print(f"eval_set finished. success={success}")
    print(f"Logs (resumable) are in: {run_log_dir}")
    print("Re-run this script unchanged to resume/retry any incomplete samples.")
    print("View results with: inspect view --log-dir", run_log_dir)


if __name__ == "__main__":
    main()
