#!/usr/bin/env bash
# Setup script for the EM-in-agentic-settings project.
#
# Works identically on a local machine or inside a Colab cell.
# On Colab, either:
#   (a) save this as a .sh file and run: !bash 01_setup.sh
#   (b) or paste each command into its own cell, prefixed with `!`
#       (e.g. `!git clone ...`) — Colab shell commands are just this
#       script's lines, one per cell.
# There is no "clone on Colab vs clone locally" distinction — `git clone`
# fetches the same repo either way; only the machine it lands on differs.

set -e  # stop on first error

# ---------------------------------------------------------------------------
# 1. Virtual environment
#    (On Colab you can skip this and pip-install directly into the runtime,
#    since each Colab session is already an isolated container. This block
#    is here so the exact same script also works on your local laptop.)
# ---------------------------------------------------------------------------
python3 -m venv em_agentic_env
source em_agentic_env/bin/activate
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2. Clone the official AgentHarm implementation
#    AgentHarm's eval code lives inside the Inspect AI evals monorepo.
# ---------------------------------------------------------------------------
git clone https://github.com/UKGovernmentBEIS/inspect_evals.git
cd inspect_evals
pip install -e .
cd ..

# ---------------------------------------------------------------------------
# 3. Core Python dependencies
#    (torch/transformers versions intentionally left unpinned here —
#    pin them yourself once you've picked your exact model/quantization
#    setup; that's a design choice left to you.)
# ---------------------------------------------------------------------------
pip install inspect-ai
pip install transformers accelerate peft bitsandbytes
pip install huggingface_hub datasets
pip install python-dotenv  # for keeping API keys out of scripts

# ---------------------------------------------------------------------------
# 4. Hugging Face auth (needed to pull gated models like Llama-3.1, and the
#    AgentHarm dataset + Turner et al.'s adapters)
# ---------------------------------------------------------------------------
huggingface-cli login
# Paste your HF token when prompted. Get one at https://huggingface.co/settings/tokens

echo ""
echo "Setup complete."
echo "AgentHarm eval code is in: ./inspect_evals/src/inspect_evals/agentharm"
echo "Next: run 02_checkpoint_utils.py's self-test, then wire up 03_run_eval.py"
