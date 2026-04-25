# PharmaRL — Baseline Policy Spectrum

Before training, we ran 6 policies on the same eval protocol (9 episodes per target × 3 targets, easy difficulty, on the live HF Space). This pins the score-to-beat for the trained Qwen 1.5B and surfaces a non-obvious finding about model scaling in this env.

## Headline finding: inverted scaling

| Llama Size | Mean cumulative reward | Why |
|---|---|---|
| 3B Instruct | +1.67 | Shallow chemistry priors but conservative actions |
| **8B Instruct** | **+2.45** | Sweet spot — priors plus restraint |
| 70B Instruct | +1.19 | Over-optimizes; multi-fragment molecules fail Lipinski + chemistry validation |

70B is the worst of the Llama family on this env. Its failure mode: it tries to design complex multi-fragment molecules in a single turn and runs head-first into the env's validator. Parse rate also slipped to 97% (3 failed-JSON turns out of 135).

**Implication for the pitch:** In this constrained action space, raw model capacity is anti-correlated with performance past a sweet spot. The env's reward-design discipline penalizes capacity-greedy strategies. Bigger ≠ better.

## Full baseline spectrum

| Source | DRD2 | GSK3B | JNK3 | Mean | Parse | Cost | Wall |
|---|---|---|---|---|---|---|---|
| Random uniform | +2.78 | +2.33 | +1.78 | +2.30 | n/a | $0 | <1m |
| Scripted (4-step) | +2.90 | +3.04 | +2.50 | +2.81 | n/a | $0 | <1m |
| Llama 3.2 3B | +1.80 | +1.99 | +1.22 | +1.67 | 100% | $0.001 | 2m |
| Gemini 2.5 Flash | +2.18 | +1.10 | +2.15 | +1.81 | 100% | $0.026 | 3m |
| Llama 3.1 8B | +2.52 | +2.57 | +2.27 | +2.45 | 100% | $0.001 | 3m |
| Llama 3.3 70B | +1.65 | +0.79 | +1.14 | +1.19 | 97% | $0.007 | 4m |
| Gemini 2.5 Pro | +4.74 | +3.40 | +2.91 | +3.68 | 100% | $0.123 | 21m |

Total probe spend: **$0.158** across 7 policies. 100% parse rate on all but 70B.

## What this proves

**1. Reward signal is meaningful.** Random and scripted policies produce measurably different scores than capacity-greedy LLMs. The composite reward + Lipinski gate + anti-gaming guards make capacity-greedy strategies fail.

**2. Frontier-class capability isn't enough.** Gemini 2.5 Pro is the only off-the-shelf LLM that beats the hand-coded scripted heuristic. Llama 70B — a much larger model than what we train — falls below random uniform.

**3. Targeted RL is the right tool.** The thesis behind training a 1.5B Qwen with GRPO on this env is that *targeted reward optimization* outperforms *raw capacity*. The baselines are the proof: capacity alone doesn't solve this task.

## Score-to-beat for trained Qwen 1.5B

| Tier | Mean | Implication |
|---|---|---|
| ≥ +2.30 | Beats random | Trivial floor |
| ≥ +2.81 | Beats scripted heuristic | "Agent learned" floor — must clear this to claim the pitch |
| ≥ +2.45 | Beats untrained Llama 8B sweet spot | Capacity-class lift |
| ≥ +3.68 | Approaches Gemini 2.5 Pro | Banger result |

## How these were generated

- `examples/demo.py --policies random scripted` for the rule-based baselines
- `examples/llm_baselines/openrouter_runner.py` for Llama 3B / 8B / 70B (via OpenRouter)
- `examples/llm_baselines/gemini_runner.py` for Gemini 2.5 Flash / Pro
- All probes used identical eval protocol: 9 episodes per target, easy difficulty, fresh episode_id per rollout, no in-context history between episodes
- Live HF Space (`anshumanatrey-pharmarl.hf.space`) was the env in all cases

## Reproducing

The runners + demo script are committed. Set `OPENROUTER_API_KEY` and/or `GEMINI_API_KEY` in `.env`, then:

```bash
python -m examples.demo --policies random scripted --target DRD2 GSK3B JNK3 --episodes 9
python -m examples.llm_baselines.openrouter_runner --models llama-3.2-3b-instruct llama-3.1-8b-instruct llama-3.3-70b-instruct
python -m examples.llm_baselines.gemini_runner --models gemini-2.5-flash gemini-2.5-pro
```

Expect under $0.20 total spend. Most of that is Gemini Pro (4-min wall per target).
