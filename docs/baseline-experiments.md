# PharmaRL — Baseline experiment log

Empirical baselines run against the deployed env (`https://anshumanatrey-pharmarl.hf.space`).
**These are the BEFORE numbers — every training-time claim should reference one of these.**

Append to this file as new runs complete. Do not rewrite historical entries — if a
config bug was found later, leave the bugged run in place and add a follow-up entry
that links back to it. Honesty about what was actually measured > pretending the
broken run did not happen.

---

## 2026-04-25 23:20 — Gemini 2.5 Flash, untrained, 3 × 3 probe

### Setup

| field | value |
|---|---|
| Runner | `examples/gemini_episode.py` |
| Model | `gemini-2.5-flash` (paid, $0.30/M in, $2.50/M out) |
| Env URL | `https://anshumanatrey-pharmarl.hf.space` |
| Difficulty | `easy` (binding component active; trivial would be QED-only) |
| Targets | DRD2, GSK3B, JNK3 |
| Episodes / target | 3 |
| Max steps / episode | 20 (server cap = 15 on `easy`, so episodes terminated at 15) |
| Temperature | 0.4 |
| `thinking_budget` | 0 (disabled — Flash supports it; saves output tokens + latency) |
| `max_output_tokens` | 200 |
| Result file | `/tmp/gemini_results.json` (not committed — dev artifact) |

Reproduce: `python -m examples.gemini_episode --target DRD2 GSK3B JNK3 --episodes 3 --model gemini-2.5-flash --out /tmp/gemini_results.json`

### Quantitative results

| Target | Mean cumulative reward | Episodes (cumulative) | Mean parse rate |
|---|---|---|---|
| DRD2  | **+2.184** | 2.556, 1.441, 2.556 | 100% |
| GSK3B | **+1.101** | 0.376, 0.376, 2.552 | 100% |
| JNK3  | **+2.145** | 2.078, 1.529, 2.829 | 100% |

Cost: **$0.026** total across 9 episodes. Wall: ~3 min. Tokens: ~62K input / ~3K output.

### Comparison to no-LLM baselines (run earlier, same env, same difficulty)

Source: `python -m examples.demo --policies random scripted --target DRD2 GSK3B JNK3` (8 ep / cell)

| Source | DRD2 | GSK3B | JNK3 |
|---|---|---|---|
| Random policy (uniform over valid_actions) | +2.78 | +2.33 | +1.78 |
| Scripted policy (4-step DRD2 plan: N → benzene → methyl → methoxy) | +2.90 | +3.04 | +2.50 |
| Gemini 2.5 Flash (this run) | **+2.18** | **+1.10** | **+2.15** |

**Untrained Flash is comparable to or worse than random across all three targets.**
That is the headline number for the pitch's BEFORE column.

### Qualitative findings

1. **Format compliance is solved.** 100% parse rate across 9 episodes. The JSON
   action format is not a bottleneck for a sub-10B chat model — the parse-fail
   penalty (-0.5) is essentially never charged. Whatever RL learns, it does not
   need to learn formatting.

2. **Failure mode = positional addressing.** Tail logs show repeated invalid
   `REMOVE_FRAGMENT` / `SUBSTITUTE_ATOM` calls at the same `position` even after
   the env returns the unchanged molecule. Each invalid action costs -0.10. In
   the worst episodes (GSK3B ep 1 & 2) Flash issued the same `ADD_FRAGMENT N`
   at position 0 twice in a row on `Fc1ccccc1`, never moved past the seed,
   and the cumulative reward came almost entirely from the terminal composite.

3. **It can build real molecules in the first 4-5 turns.** Examples:
   - DRD2 ep 1 final: `Cc1cccc(Nc2ccncc2)c1` — toluene + aminopyridine, basic
     amine + aromatic — small but D2-pharmacophore-adjacent.
   - GSK3B ep 3 final: `CNC(O)Cc1cccc(-c2cc(F)nc(-c3ccncc3)c2)c1` — N-methyl
     amine + biaryl pyridine — kinase-inhibitor-shaped.
   - JNK3 ep 3 final: `CN(C(=O)Oc1ccncc1)C(CCO)C(=O)c1ccccc1` — N-methyl
     carbamate + pyridine + benzoyl — realistic kinase scaffold.
   The model has shallow chemistry priors. RL on top of these priors is the
   experiment, not "teach the model chemistry from zero."

### Implications for training

- **Held-out JNK3 baseline locked at +2.15** for Gemini Flash, +1.78 for random,
  +2.50 for the scripted plan. Trained Qwen 1.5B must beat all three by a
  margin that exceeds the noise band before we claim transfer.
- Format penalty is irrelevant at this model scale — the trained Qwen 1.5B
  needs to learn *positional addressing*, not JSON formatting. That is the real
  RL signal.
- Don't pitch "the LLM brings chemistry knowledge" as the differentiator.
  Flash's chemistry priors produce sub-random performance once the addressing
  failures kick in. The differentiator is environment design + RL, not the
  base model's medicinal chemistry IQ.

---

## 2026-04-25 23:21 — Gemini 2.5 Pro, untrained, 2 × 3 probe — **CONFIG BUG, do not cite**

### Result

`parse_rate=0%`, `output_tokens=0` for all 6 episodes. The env saw fallback
actions every turn and final molecules degenerated into degenerate alkyl
chains (`CCCCCCCCCCCCCCCOc1ccccc1` and similar). **The Pro model never
actually played the env in this run.**

### Cause

`thinking_config(thinking_budget=0)` was applied to `gemini-2.5-pro`. Flash
supports disabled thinking; Pro silently returns 0 output tokens when thinking
is disabled. The Pro response was empty every turn → `parse_action()` failed →
fallback to `{"action_type":"ADD_FRAGMENT","fragment":"C","position":0}`. The
env's curriculum then accreted the carbon chain.

### Fix

`examples/gemini_episode.py` now picks the config per model family:

```python
if "flash" in model.lower():
    thinking_budget, max_out = 0, 200       # Flash: thinking off, short reply
else:                                       # Pro / 1.5
    thinking_budget, max_out = 1024, 2000   # Pro: thinking on, generous max
```

`max_output_tokens > thinking_budget` is required — otherwise thinking eats
the whole output budget.

Result file `/tmp/gemini_pro_results.json` is preserved for reproducibility
of the bug, but **none of those numbers represent Pro's actual performance**.

---

## 2026-04-25 ~23:30 — Gemini 2.5 Pro, untrained, 1-episode verbose smoke

### Setup

Single episode against DRD2, fixed config (thinking_budget=1024, max=2000),
verbose stdout only — no JSON file written.

### Result

| metric | value |
|---|---|
| Composite | **0.71** |
| Terminal reward | **+7.15** (composite × 10) |
| Cumulative reward | **+6.65** (terminal + step-shaping − step penalties) |
| Final molecule | `CC1NCCC(Oc2cccc(O)c2F)C1c1ccc(Cl)cc1` |
| Cost | ~$0.013 |

The final molecule has piperidine (basic amine) + aryl ether linker +
halophenyl — textbook D2 pharmacophore territory.

**This is one episode. It is anecdotal, not a probe.** A 9-episode Pro
re-run with the fixed config has not been completed and committed to a JSON
file as of this entry. Treat 0.71 composite as a *plausible upper bound* for
what a flagship LLM with no RL can do on this env, not as a formal baseline.

### Why this number matters as a soft ceiling

A trained Qwen 1.5B with RL on this env should plausibly reach **~0.5-0.6
composite** on DRD2. If a trained 1.5B model significantly exceeds Pro's
single-episode 0.71, suspect reward hacking and re-run
`tests/test_reward_redteam.py` before pitching the number.

---

## How to extend this log

When a new training run, ablation, or model probe completes:

1. Append a new dated section above this block. Do not edit prior entries.
2. Capture: setup table, command to reproduce, quantitative table, qualitative
   findings, implications.
3. If a config bug is found, leave the bugged entry in place and add a
   follow-up entry that explains the bug and links back. Pretending the
   broken run didn't happen is the only thing worse than the bug.
4. Save raw per-episode JSON to `/tmp/<run-name>.json` and reference the
   filename. Do not commit the JSON — it has token-usage data that is
   dev-side artifact.
5. Cost of every paid-API run goes in the entry. Local-first / never-spend
   memory still applies; if you spent $X on this run, say so explicitly.
