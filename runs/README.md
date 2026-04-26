# `runs/` — training run audit bundles

Each completed training run lands in a subdirectory here. Convention:

```
runs/{user}-{gpu}-{model}/
```

Examples:
- `vijay-h200-llama1b/` — Vijay, NVIDIA H200, Llama-3.2-1B-Instruct
- `anshuman-a10g-llama3b/` — Anshuman, A10G-large, Llama-3.2-3B-Instruct
- `sahil-kaggle-llama3b/` — Sahil, Kaggle T4/P100, Llama-3.2-3B-Instruct

## Standard structure inside each run folder

```
runs/{user}-{gpu}-{model}/
├── README.md                  # Human-readable run summary
├── hf_hub/
│   ├── RUN_SUMMARY.md         # auto-generated on training finish
│   ├── run_metadata.json      # job id, GPU, model, hyperparams, exit code
│   ├── train.log              # full trainer stdout
│   ├── entry.log              # host setup + sanity checks
│   └── env_server.log         # env sidecar (if applicable)
├── wandb/
│   ├── run_config.json        # W&B config snapshot
│   ├── run_summary.json       # final-step W&B summary metrics
│   ├── run_history.csv        # per-step metric history (CSV)
│   ├── run_history.json       # per-step metric history (JSON)
│   ├── run_meta.json          # W&B run metadata (URL, project, name)
│   └── system_metrics.csv     # GPU/CPU/memory time series
└── plots/
    ├── 01_reward_over_steps.png
    ├── 02_policy_kl.png
    ├── 03_parse_rate.png
    ├── 04_reward_components.png
    └── 05_loss.png
```

## Fetching a run bundle from W&B + HF Hub

A `_fetch.py` script lives next to each bundle that re-downloads from the
sources (W&B run + HF Job logs). Useful if you want to refresh after the
run is older than 30 days (W&B free tier retention).

## Why this lives in main (not gitignored)

`runs/` was previously in `.gitignore` because per-validation-step logs (in
`logs/`) accumulate fast. Completed-training-run audit bundles are
different — they're small (~5-15 MB), shareable, and serve as the single
source of truth for the README's Results section + the pitch's W&B
references.

`logs/` stays gitignored for transient validation runs.

## Comparing runs

```bash
# Once two or more runs are bundled, you can plot side-by-side:
python -m examples.plot_results \
    --inputs runs/vijay-h200-llama1b/wandb/run_history.csv \
             runs/anshuman-a10g-llama3b/wandb/run_history.csv \
    --output-dir docs/plots/comparison/
```
