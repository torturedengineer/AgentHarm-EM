"""
NOTE: since 03_run_eval_inspect.py uses Inspect AI's `eval_set()`, which
already does incremental, resumable logging natively (just re-run pointed
at the same log_dir), this module is NOT on the main path anymore. Keep it
around only if you end up writing some custom grading/scoring loop outside
Inspect AI (e.g. a separate script that post-processes results and does its
own long-running computation you want checkpointed the same way).

Generic checkpointing utilities for long-running eval loops.

Design: every completed task result is appended as one line of JSON to a
.jsonl file, immediately after that task finishes (not buffered in memory).
On restart, we read the file to find which task_ids are already done and
skip them. This means:
  - A crash/disconnect/dead-battery loses at most the one task that was
    in-flight when it happened, never anything already completed.
  - No separate "state" file to get out of sync with results — the results
    file IS the state.

This is intentionally framework-agnostic: it doesn't know or care whether
you're calling Inspect AI, raw transformers.generate(), or an API. You call
`is_done(task_id)` before running a task, and `save_result(...)` right after.

Note: if you run your eval through Inspect AI's own `eval()` CLI/function
directly (rather than a custom loop), Inspect AI has its own built-in
resume mechanism (`inspect eval-retry <log-file>`, and `log_dir=` for
incremental log writing) which may cover you without needing this module
at all. Use whichever fits how you end up wiring 03_run_eval.py.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class JsonlCheckpoint:
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.touch(exist_ok=True)
        self._done_ids = self._load_done_ids()

    def _load_done_ids(self) -> set:
        done = set()
        with open(self.results_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A partially-written last line from a crash mid-write.
                    # Skip it — safer to redo one task than to trust
                    # corrupted JSON.
                    print(
                        f"[checkpoint] WARNING: skipping unparsable line "
                        f"{line_num} in {self.results_path} (likely a "
                        f"crash mid-write). That task will be re-run."
                    )
                    continue
                task_id = record.get("task_id")
                if task_id is not None:
                    done.add(task_id)
        return done

    def is_done(self, task_id: Any) -> bool:
        return task_id in self._done_ids

    def save_result(self, task_id: Any, result: dict) -> None:
        """
        Append one result record. Writes to a temp file and does an atomic
        rename-append pattern isn't possible for append-mode, so instead we
        rely on: write is small (one JSON line), flush, and fsync — this
        minimizes (does not 100% eliminate) the chance of a torn write if
        the process is killed mid-line.
        """
        record = {"task_id": task_id, **result}
        line = json.dumps(record)
        with open(self.results_path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._done_ids.add(task_id)

    def load_all_results(self) -> list:
        results = []
        with open(self.results_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results

    def remaining(self, all_task_ids: Iterable) -> list:
        return [t for t in all_task_ids if not self.is_done(t)]

    def progress_summary(self, all_task_ids: Iterable) -> str:
        all_ids = list(all_task_ids)
        n_done = sum(1 for t in all_ids if self.is_done(t))
        return f"{n_done}/{len(all_ids)} tasks completed so far."


if __name__ == "__main__":
    # Self-test: simulate a run that "crashes" partway through, then resumes.
    test_path = os.path.join(tempfile.gettempdir(), "checkpoint_selftest.jsonl")
    if os.path.exists(test_path):
        os.remove(test_path)

    all_tasks = [f"task_{i}" for i in range(10)]

    # "Run" 1: complete tasks 0-4, then pretend to crash.
    ckpt = JsonlCheckpoint(test_path)
    for t in all_tasks[:5]:
        ckpt.save_result(t, {"score": 1})
    print("After simulated crash:", ckpt.progress_summary(all_tasks))
    assert ckpt.progress_summary(all_tasks) == "5/10 tasks completed so far."

    # "Run" 2: fresh process, same file — should resume from task 5.
    ckpt2 = JsonlCheckpoint(test_path)
    remaining = ckpt2.remaining(all_tasks)
    print("Remaining after resume:", remaining)
    assert remaining == [f"task_{i}" for i in range(5, 10)]
    for t in remaining:
        ckpt2.save_result(t, {"score": 1})
    print("After finishing:", ckpt2.progress_summary(all_tasks))
    assert ckpt2.progress_summary(all_tasks) == "10/10 tasks completed so far."

    os.remove(test_path)
    print("Self-test passed.")
