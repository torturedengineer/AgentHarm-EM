"""
Minimal reconnect sequence, using your GitHub repo
(https://github.com/torturedengineer/AgentHarm-EM) as the source of truth for
scripts/code, and Drive purely for what must persist: logs and results.

Run this as one cell (or split into the numbered chunks) every time you
reconnect. After that, whenever you push an edit to GitHub, just re-run the
`git pull` line before your next `!python .../03_run_eval_inspect.py` call —
no caching to fight, since it's a fresh subprocess reading the file from disk
each time, not an imported/cached Python module.
"""

import os
import subprocess

BASE_DIR = "/content/drive/MyDrive/AgentHarm-EM"   # Drive: logs/results only
REPO_DIR = "/content/AgentHarm-EM"                  # local: your scripts, from GitHub
REPO_URL = "https://github.com/torturedengineer/AgentHarm-EM.git"
INSPECT_EVALS_DIR = "/content/inspect_evals"        # local: disposable, re-cloned each session


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    # 1. Drive — only for logs/results persistence
    os.makedirs(BASE_DIR, exist_ok=True)
    for sub in ["logs", "results"]:
        os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

    # 2. Your scripts repo — clone fresh each session (or pull if it somehow
    #    survived, e.g. you didn't fully reconnect)
    if os.path.isdir(REPO_DIR):
        run(["git", "-C", REPO_DIR, "pull"])
    else:
        run(["git", "clone", REPO_URL, REPO_DIR])

    # 3. inspect_evals — local, disposable, same as before
    if not os.path.isdir(INSPECT_EVALS_DIR):
        run(["git", "clone", "--depth", "1",
             "https://github.com/UKGovernmentBEIS/inspect_evals.git", INSPECT_EVALS_DIR])
    run(["pip", "install", "-e", INSPECT_EVALS_DIR])

    # 4. Your requirements
    requirements_path = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(requirements_path):
        run(["pip", "install", "-r", requirements_path])

    # 5. NOTE ON openai VERSION: this used to hardcode `pip install
    #    openai==2.54.0` here, which silently overrode whatever you set in
    #    requirements.txt — that's the exact "why did my edit not take
    #    effect" confusion from before. Fixed: the version now comes ONLY
    #    from requirements.txt. Since `inspect_evals` above is cloned fresh
    #    from `main` each session (not pinned to a release), its openai
    #    version requirement can change day to day — if you hit an openai
    #    version error, check requirements.txt and edit the version there,
    #    not here.

    print("\nBootstrap done.")
    print(f"Scripts (from GitHub): {REPO_DIR}")
    print(f"Logs/results (Drive):  {BASE_DIR}")
    print("\nNext: authenticate (HF + OpenAI), then run 03_run_eval_inspect.py from REPO_DIR.")


if __name__ == "__main__":
    main()
