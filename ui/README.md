# PharmaRL — Live Operations Center (Gradio UI)

Single-screen demo dashboard for the PharmaRL env. Runs an episode in
real time inside the same Python process as the env (no HTTP, no separate
uvicorn), streams the per-step trace, and renders 2D molecule structures,
reward curves, oracle decomposition, and the published baselines table.

## Run it

From the `pharmarl/` repo root:

```bash
# the project venv already has gradio + rdkit + plotly
.venv/bin/python -m ui.app
# then open http://localhost:7860
```

If the port is busy: `PORT=7861 .venv/bin/python -m ui.app`.

## What it shows

- 🎯 **Scenario picker** — three difficulty tiers + binding target dropdown
- 🤖 **Agent picker** — Random + Scripted are *live* (run end-to-end);
  Llama / Gemini rows show the published baselines from `docs/baselines.md`
- 🧬 **Streaming trace** — each step prints the action, reward, and message
  as the env returns it
- 🧪 **Live 2D molecule** — RDKit-rendered structure that updates each step
- 📈 **Reward curve** — Plotly chart of cumulative + per-step reward
- 🎯 **Reward decomposition** — final binding / QED / SA / toxicity contributions
  with composite weights
- 🪄 **Action histogram** — ADD / REMOVE / SUBSTITUTE / TERMINATE usage
- 📋 **Final scores table** — composite breakdown vs reference drug
- 🏆 **Baseline leaderboard** — surfaces the inverted-scaling story (8B > 70B)

## Adding the trained adapter

The "Llama 1B trained (vijay-h200)" row is currently a placeholder. Once
the H200 GRPO adapter is uploaded to HF Hub, drop in a new agent class
in `ui/agents.py` that loads it with `unsloth.FastLanguageModel.from_pretrained`
and emits actions via `model.generate`. Then mark `live=True` on the
matching `AgentInfo` and the UI will route to it automatically.

## Dependencies

- `gradio`, `rdkit`, `plotly` — UI
- `openenv-core`, `selfies`, `PyTDC` — env (already in `server/requirements.txt`)
