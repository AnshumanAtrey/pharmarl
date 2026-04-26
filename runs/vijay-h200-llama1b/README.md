# vijay-h200 — H200 GRPO run audit bundle

Self-contained dump of every artefact from Vijay's parallel-insurance H200
training run on Hugging Face Jobs (issue
[AnshumanAtrey/pharmarl#14](https://github.com/AnshumanAtrey/pharmarl/issues/14)).
**No auth required to read this folder** — everything was fetched from
W&B + the public HF Hub model repo and committed in-tree.

## TL;DR

- **Job:** `69edba4bd2c8bd8662bcf723` on `--flavor h200`, 139 minutes wall, ~$11.50
- **Status:** trainer exited 0; LoRA adapter pushed to HF Hub
- **Honest result:** the H200 stack worked end-to-end on Hopper (cu12.4),
  SFT format-priming converged (loss 1.35 → 0.10), but **GRPO never produced
  a learning signal**. Across all 200 GRPO steps `parse_rate=0` and
  `policy_loss ≈ 0` (~5.5e-09 final). The 8 G-rollouts kept hitting the
  hardcoded fallback action (`ADD_FRAGMENT C @0`), so reward variance
  collapsed to ~0 and no policy gradient flowed. The "trained" adapter is
  effectively the SFT-warmed model, not a GRPO-improved one. KL did drift
  0.74 → 0.40 but that's optimizer momentum noise on near-zero gradients.
- **What actually got produced:** a working H200 pipeline (env sidecar +
  trainer + audit-trail upload) and the SFT-primed Llama 1B adapter.

## Where to find the live artefacts (clickable)

| Artefact | URL |
|---|---|
| HF Hub model repo (adapter + audit logs) | <https://huggingface.co/vijay2776/pharmarl-llama-3b-trained-vijay-h200> |
| W&B run | <https://wandb.ai/vijaykota2776-itm/pharmarl/runs/zke7p0gr> |
| W&B project | <https://wandb.ai/vijaykota2776-itm/pharmarl> |
| GitHub branch (this code) | <https://github.com/AnshumanAtrey/pharmarl/tree/vijay> |
| Original issue | <https://github.com/AnshumanAtrey/pharmarl/issues/14> |
| Source HF Space (Job's code source) | <https://huggingface.co/spaces/vijay2776/pharmarl/tree/vijay-h200> |

## What's in this folder

```
resources/vijay-h200-run/
├── README.md                  ← this file
├── _fetch.py                  ← rerun to refresh (needs WANDB_API_KEY in pharmarl/.env)
├── wandb/
│   ├── run_history.csv        ← 208 rows × 28 cols, every logged step
│   ├── run_history.json       ← same data as JSON
│   ├── run_summary.json       ← 25 final-state metrics
│   ├── run_config.json        ← what we passed to the trainer
│   ├── system_metrics.csv     ← 1108 rows of GPU / CPU / mem / net stats
│   └── run_meta.json          ← run id, dates, URL
├── hf_hub/
│   ├── entry.log              ← in-container entrypoint sanity-check log
│   ├── train.log              ← full trainer stdout (200 GRPO steps)
│   ├── env_server.log         ← uvicorn env sidecar log
│   ├── run_metadata.json      ← Job metadata + GPU info
│   └── RUN_SUMMARY.md         ← the in-container summary the entry script wrote
└── plots/
    ├── 01_reward_over_steps.png       ← mean + max reward across 200 steps
    ├── 02_policy_kl.png               ← shows pol≈0, KL drift
    ├── 03_parse_rate.png              ← flat 0% (the smoking gun)
    ├── 04_reward_components.png       ← qed/docking/sa/toxicity per step
    └── 05_loss.png                    ← SFT + GRPO loss
```

## Headline numbers (from `wandb/run_summary.json`)

| Metric | Final value | What it means |
|---|---|---|
| `policy_loss` | **−5.5 × 10⁻⁹** | ~0. No GRPO gradient flow ever happened. |
| `parse_rate` | **0.0** | Across all 200 steps, the model never emitted parseable JSON. |
| `kl` | 0.400 | Drifted from 0.74 → 0.40; not from learning, from optimizer momentum on tiny grads. |
| `mean_reward` | +0.81 | Final-step group mean. The reward comes from the fallback action getting graded by the env, not from learning. |
| `max_reward` | +1.44 | Best single rollout in the final group. |
| `verifier/invalid_action_rate` | 0.50 | Half the env steps were invalid, consistent with a model parroting the fallback regardless of state. |
| `verifier/lipinski_pass_rate` | 0.50 | Half the final molecules satisfy Rule of 5. |
| `verifier/truncation_rate` | 1.00 | Every episode hit step-cap (no `TERMINATE` chosen because the model never emitted one). |
| `reward_component/qed` | 0.42 | drug-likeness raw |
| `reward_component/docking` | 0.012 | DRD2 binding raw — basically zero (long alkyl chains don't bind D2) |
| `reward_component/sa` | 0.92 | synthesizability raw |
| `reward_component/toxicity_clean` | 1.00 | non-toxic raw — but trivially because the fallback action produces benign small alkanes |
| `action_type/ADD_FRAGMENT` | 120 | every action was ADD; 0 SUBSTITUTE / REMOVE / TERMINATE — the fallback is `ADD_FRAGMENT C @0` |

The `action_type/ADD_FRAGMENT = 120` row is the single cleanest proof
the agent was stuck on the fallback action: GRPO step 199 did 8 rollouts
of 15 steps each, all of them ADD_FRAGMENT — i.e. **the model produced
zero usable JSON to choose anything else**.

## Failure ladder we crawled to get here

| # | Job ID | Failed at | Surgical fix |
|---|---|---|---|
| 1 | `69eda84fd70108f37acdfa93` | model load — `Unsloth: Please install vLLM` | `--no-fast-inference` |
| 2 | `69edabc3d70108f37acdfafe` | bnb dequant — `libnvJitLink.so.13` missing | `bitsandbytes==0.45.5` |
| 3 | `69edacefd70108f37acdfb19` | pip resolve — xformers 0.0.30 needs torch 2.7 | `xformers==0.0.29.post3` |
| 4 | `69edaff4d2c8bd8662bcf5b3` | SFT backward — `fast_lora.py:116 BFloat16 != float` | tried `unsloth_zoo<2025.3` |
| 5 | `69edb0f7d70108f37acdfb8a` | same | confirmed not a zoo issue |
| 6 | `69edb283d2c8bd8662bcf60d` | same | added `--no-4bit --no-gradient-checkpointing` |
| 7 | `69edb398d70108f37acdfbe6` | GRPO step 0 — `KeyError: 'reward'` | wrap `/step` body + `episode_id` |
| 8 | `69edb573d2c8bd8662bcf66e` | running, but `parse=0% pol=0` (cancelled) | `--sft-warmup-steps 120` |
| 9 | `69edb790d70108f37acdfc57` | same | `--gen-temp 0.3` |
| **10** | **`69edba4bd2c8bd8662bcf723`** | **completed** — exit 0, but parse=0% all 200 steps | (this is the run captured here) |

Pinned versions that finally produced a clean install on H200:

```
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
torch>=2.6.0,<2.7         transformers==4.46.3
trl==0.13.0               peft (latest)
unsloth==2025.2.15        unsloth_zoo>=2025.2.7,<2025.3
bitsandbytes==0.45.5      xformers==0.0.29.post3
```

Trainer flags used: `--no-fast-inference --no-4bit --no-gradient-checkpointing --gen-temp 0.3 --sft-warmup-steps 120`.

## Why parse stayed at 0% (root cause)

Three suspected causes, in order of likelihood:

1. **Pad-token == EOS-token + missing attention mask.** `train_grpo.py:rollout()`
   calls `tokenizer(prompt, return_tensors='pt').input_ids` and passes
   only `input_ids` (no `attention_mask`) to `model.generate`. With pad
   == eos, the generation can stop early before the closing `}`. The
   transformers warning is logged at the top of every run.
2. **SFT teaches `prompt + " " + json + eos` but inference continues from
   `prompt`** — the model has to produce the leading space first, then `{`.
   Even at `temp=0.3` the first sampled token can be off-format and
   subsequent tokens never recover.
3. **1B may be too small for this format under sampling.** Anshuman's
   a10g a10g-large run uses the same trainer code — worth checking his
   parse rate. If his is also 0%, the env's Reward Improvement criterion
   may be in trouble for the whole project.

## What I'd do, in order

### A. Eval the adapter — confirm it's actually different from base
**~10 min, $0.** Load `vijay2776/pharmarl-llama-3b-trained-vijay-h200/lora_final`
locally, run 10 episodes through the env, measure parse rate. If parse >
base parse → SFT did something real. If both are 0% → the entire training
did nothing useful and we should not claim it does.

### B. Wire the adapter into the UI as the "Trained Llama 1B" agent
**~30 min, $0.** Right now it's a placeholder button in `ui/agents.py`.
After eval, plug the adapter into the UI so it actually generates actions
in real time. **Don't host inference on HF Endpoints** ($, slow, overkill)
— just load it locally in the Gradio process.

### C. Decide on framing
Two options:
- **Honest:** "We used the H200 to SFT-format-prime a Llama 1B. The GRPO
  loop ran but didn't produce learning signal — see W&B."
- **Spin:** omit the GRPO part, present it as "SFT pipeline."

Honest is the right call — the W&B run is public, judges can see pol=0.

### D. Skip these
- HF Inference Endpoints: costs $, judges won't use it.
- A new HF Space for the demo: 30+ min to set up and we already have the
  UI working locally.
- Re-running training with parse fixes: ~3 h training each, no time before
  deadline.

### E. Possible recoveries if parse fix turns up time
1. Add `attention_mask` to `model.generate` in `train_grpo.py:rollout`.
2. Drop `gen_temp` further to 0.2.
3. Lower the post-SFT format check temperature too (currently hard-coded 0.7).
4. Pre-feed the leading space in inference: append `" "` to the prompt
   before tokenizing so the model only has to produce `{` first.

## Reproducing this bundle

```bash
cd pharmarl                                # repo root
.venv/bin/python -m pip install wandb      # if missing
# WANDB_API_KEY needs to be in pharmarl/.env (gitignored)
.venv/bin/python resources/vijay-h200-run/_fetch.py
```

Re-running won't change history rows (the run is closed); it will just
overwrite the CSVs/JSONs with the same data. Useful if you discover a new
metric was added to the W&B run via a follow-up `wandb.log` call.
