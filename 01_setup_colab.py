"""
Run this first, in a Colab cell (or as `!python 01_setup_colab.py`).

Design, on purpose:
  - CODE_DIR (the inspect_evals clone) lives on LOCAL Colab disk (/content),
    NOT on Drive. Cloning a 300MB+ repo of many small files directly onto
    Drive's FUSE mount is slow and can throw exactly the kind of
    "getcwd"/"folder can no longer be found" errors you just hit. Code is
    disposable — re-cloning costs ~30 seconds, so there's no reason to pay
    Drive's I/O tax for it.
  - BASE_DIR (on Drive) is reserved for what actually needs to survive a
    disconnect: logs/ and results/. That's the only data a lost runtime
    should ever cost you.
  - No venv. Each Colab `!cell` is a fresh subprocess, so a venv activated
    inside this script's one shell invocation will NOT carry over to your
    next cell anyway — it's a false sense of isolation on Colab
    specifically. Installing into the base environment is simpler and is
    what actually shows up in your next cell's imports.
"""

import os
import subprocess
import sys

BASE_DIR = "/content/drive/MyDrive/AgentHarm-EM"  # persists across sessions (Drive)
CODE_DIR = "/content/inspect_evals"                # local disk, re-cloned fresh each session

REQUIREMENTS_PATH = os.path.join(BASE_DIR, "requirements.txt")  # keep requirements.txt in BASE_DIR


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    # Drive side: only the things that must persist.
    os.makedirs(BASE_DIR, exist_ok=True)
    for sub in ["logs", "results"]:
        os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

    # Local side: disposable code.
    if os.path.isdir(CODE_DIR):
        print(f"{CODE_DIR} already exists (from earlier this session), skipping clone.")
    else:
        run(["git", "clone", "--depth", "1",
             "https://github.com/UKGovernmentBEIS/inspect_evals.git", CODE_DIR])

    run([sys.executable, "-m", "pip", "install", "-e", CODE_DIR])

    if os.path.exists(REQUIREMENTS_PATH):
        run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_PATH])
    else:
        print(
            f"WARNING: {REQUIREMENTS_PATH} not found. Put requirements.txt "
            f"in {BASE_DIR} before running this, or pip install manually."
        )

    print("\nSetup done.")
    print(f"AgentHarm eval code: {os.path.join(CODE_DIR, 'src', 'inspect_evals', 'agentharm')}")
    print(f"Logs will go to:     {os.path.join(BASE_DIR, 'logs')}   <- on Drive, survives disconnects")
    print(f"Results will go to:  {os.path.join(BASE_DIR, 'results')} <- on Drive, survives disconnects")
    print("\nNext: authenticate with Hugging Face, then run 03_run_eval_inspect.py.")
    print("  In a separate cell:  from huggingface_hub import login; login()")
    print("  (the `huggingface-cli login` you saw deprecated — use the python call, or `!hf auth login`)")


if __name__ == "__main__":
    main()
