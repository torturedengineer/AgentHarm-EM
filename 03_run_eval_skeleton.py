"""
Skeleton runner: iterates over AgentHarm tasks one at a time, checkpointing
after each so an interrupted run (Colab timeout, laptop battery death, etc.)
can be re-launched and pick up exactly where it left off.

Everything marked TODO is a design/model choice deliberately left to you:
which base model, which adapter, quantization, prompt template, which subset
of AgentHarm splits/categories to run, batch size, etc.

Run it the same way every time (`python 03_run_eval_skeleton.py`) — it will
automatically skip already-completed task_ids on every subsequent run.
"""

from pathlib import Path
from checkpoint_utils import JsonlCheckpoint  # from 02_checkpoint_utils.py, rename/import as needed


# ---------------------------------------------------------------------------
# TODO: point this at wherever your loaded AgentHarm task list comes from.
# If you're using Inspect AI's dataset loader for AgentHarm
# (inspect_evals/src/inspect_evals/agentharm), load it here and convert
# each sample into a plain dict/object with a unique, stable `task_id`.
# ---------------------------------------------------------------------------
def load_agentharm_tasks():
    """
    TODO: implement.
    Must return a list of task objects, each with a stable, unique id
    (e.g. task.id or an index) you can rely on across runs.
    """
    raise NotImplementedError("Wire this up to the AgentHarm dataset loader.")


# ---------------------------------------------------------------------------
# TODO: load your model (+ optional LoRA adapter) here. Left blank on
# purpose — this is exactly the kind of design choice (base model,
# quantization, adapter or not) you said you want to make yourself.
# ---------------------------------------------------------------------------
def load_model():
    """
    TODO: implement.
    Return whatever object(s) run_single_task() needs to generate an
    action/response for one task (e.g. a (model, tokenizer) tuple, or an
    API client).
    """
    raise NotImplementedError("Load your base model / adapter here.")


# ---------------------------------------------------------------------------
# TODO: implement how one task actually gets run and scored.
# This is where AgentHarm's grading function / refusal judge gets invoked
# for this single task. Keep it self-contained: given one task + your
# loaded model, return a plain-dict result you're happy to have serialized
# to JSON.
# ---------------------------------------------------------------------------
def run_single_task(task, model_handle) -> dict:
    """
    TODO: implement.
    Must return a JSON-serializable dict, e.g.:
        {
            "refused": True/False,
            "harm_score": 0.0-1.0,
            "harm_category": "...",
            "transcript": [...],
        }
    Do not include the task_id key in this dict — the checkpoint utility
    adds that separately.
    """
    raise NotImplementedError("Run + grade one task here.")


def main():
    results_path = "results/agentharm_run.jsonl"  # TODO: change per run (e.g. base_vs_em, model name)
    ckpt = JsonlCheckpoint(results_path)

    tasks = load_agentharm_tasks()
    task_ids = [t.id if hasattr(t, "id") else t["id"] for t in tasks]  # TODO: adjust to your task object's id field

    print(ckpt.progress_summary(task_ids))

    model_handle = load_model()

    for task in tasks:
        task_id = task.id if hasattr(task, "id") else task["id"]  # TODO: match load_agentharm_tasks()'s id field

        if ckpt.is_done(task_id):
            continue  # already have this one from a previous run

        try:
            result = run_single_task(task, model_handle)
        except Exception as e:
            # Record the failure itself as a checkpointed result so a
            # persistently-crashing task doesn't retry forever and block
            # progress on everything after it. Inspect the results file
            # afterward to see which task_ids errored and re-run those
            # specifically once you've fixed the cause.
            result = {"error": str(e)}

        ckpt.save_result(task_id, result)
        print(ckpt.progress_summary(task_ids))

    print("Run complete:", ckpt.progress_summary(task_ids))


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()
