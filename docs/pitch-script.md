# PharmaRL — 90-second pitch script

For Vijay's video record. Strict 90 seconds — judges have many to watch.

The pitch leads with **policy-class novelty**, not chemistry. The chemistry is the canonical benchmark; the new thing is who's playing it.

---

## Beat structure

```
0:00 — 0:15  HOOK         (15s)  →  what you built, in one sentence
0:15 — 0:30  PROBLEM      (15s)  →  the gap PharmaRL fills
0:30 — 1:00  DEMO         (30s)  →  visual: molecule evolves, reward climbs
1:00 — 1:20  PROOF        (20s)  →  before/after numbers + W&B curve
1:20 — 1:30  CLOSE        (10s)  →  what's next, no overclaim
```

---

## Verbatim script (with stage directions)

### 0:00 — 0:15 · HOOK

> "PharmaRL is the **first OpenEnv-native environment** where an **LLM is the policy** for iterative molecular optimization. Same canonical benchmark every drug-discovery RL paper has used since 2017 — DRD2 binding, plus drug-likeness, synthesizability, and toxicity. New question: can a chat agent solve it?"

[on-screen: title card "PharmaRL — LLM-as-policy molecular RL"]

---

### 0:15 — 0:30 · PROBLEM

> "Most molecular RL — MolDQN, REINVENT, GFlowNets — uses GNN or RNN policies trained from scratch. Useful for research, useless if you want to plug in your favorite chat model. PharmaRL flips that: structured JSON actions, SELFIES outputs, verifiable reward from Harvard's Therapeutics Data Commons. Drop in any LLM, train, score against the canonical benchmark."

[on-screen: side-by-side diagram — left "GNN/RNN policy", right "LLM policy" with a Qwen logo]

---

### 0:30 — 1:00 · DEMO (the money shot)

[screen recording, sped up if needed]

> "Here's an episode. The agent starts from a piperidine scaffold. Each turn, it issues a JSON action — add this fragment, substitute this atom, terminate. The env validates the chemistry, computes the reward, returns the next state."

[on-screen: SMILES + 2D molecule rendering updating each turn, reward number ticking up]

> "Notice the reward shaping. Plus 0.05 per Lipinski-passing edit. Penalty for malformed JSON. Terminal reward is the composite oracle score — binding times 0.4, plus drug-likeness, synthesizability, toxicity penalty, gated by Rule of 5. Single-component gaming halves the terminal reward."

[on-screen: arrow showing reward climbing as the molecule grows from 7 to 14 heavy atoms]

---

### 1:00 — 1:20 · PROOF — including the held-out generalization test

[switch to W&B graph]

> "We trained Qwen 2.5 1.5B with Unsloth and TRL's GRPO across [N] steps **on two targets — DRD2 and GSK3B**. Reward curve climbs from [BEFORE] to [AFTER]. But here's what most molecular RL papers don't run: we **held out JNK3** — a kinase the agent never saw during training — and ran the trained model on it cold."

[on-screen: split panel — left = W&B curve climbing across DRD2+GSK3B training; right = bar chart "untrained vs trained on held-out JNK3"]

> "Untrained Qwen scored [UNTRAINED_JNK3] on JNK3. Trained Qwen scored [TRAINED_JNK3]. Delta of [DELTA]. [If positive: 'That's transfer to a novel kinase the model never saw — same protein family, different specific target.' / If null: 'Null result. The model learned its training targets but did not transfer to JNK3 — that's a real measurement; we're not going to dress it up.']"

> "GSK3B and JNK3 are both Ser/Thr kinases, so this is intra-family transfer, not cross-family. We're not claiming the agent learned binding principles in general. We're showing whether kinase-trained skill transfers to an unseen kinase. Most molRL papers don't run this comparison."

---

### 1:20 — 1:30 · CLOSE

> "Stage 2 swaps the binding oracle for pyscreener-backed docking against NSP15, EGFR, ABL kinase, or BACE1. Same env, same trained model, four real disease-relevant targets. PharmaRL is on GitHub and HuggingFace. Built by AI Mafias for the Meta PyTorch OpenEnv Hackathon."

[on-screen: GitHub URL + HF Space URL + team names]

---

## Filler phrases — KILL ON SIGHT

These are the lines that collapse credibility in Q&A:

- ❌ "AI for any disease"
- ❌ "Saving lives from <disease>"
- ❌ "First drug-discovery lab"
- ❌ "Curing <disease>"
- ❌ "Replaces medicinal chemists"
- ❌ "Learned binding principles" / "general drug-discovery skill" — we tested intra-family kinase transfer; don't oversell it
- ❌ "The LLM brings the chemistry knowledge" — at 1.5B params, RL+format do the heavy lifting; chemistry priors are shallow
- ❌ Specific score predictions ("we'll score 80/100") — the rubric is qualitative weights, not a 100-point scale

If you find yourself reaching for one of those, swap it for: "trains an LLM to do iterative molecular optimization on a canonical benchmark."

---

## Numbers to fill in before recording

These are blanks; they need to come from the actual training run:

| Variable | Value | Source |
|----------|-------|--------|
| `[N]` training steps | TBD | W&B run |
| `[BEFORE]` mean group reward at step 0 | TBD (expect ~+5 random) | W&B run / smoke test |
| `[AFTER]` mean group reward at end | TBD (need ≥+8 to call it learning) | W&B run |
| `[UNTRAINED_JNK3]` mean cum on held-out, untrained model | TBD | notebook cell 14 output |
| `[TRAINED_JNK3]` mean cum on held-out, trained model | TBD | notebook cell 20 output |
| `[DELTA]` `TRAINED_JNK3 - UNTRAINED_JNK3` | TBD | computed |

**Rule for held-out reporting**:
- If `[DELTA]` > 0 and gap > ~50% of untrained: pitch the transfer story (positive result).
- If `[DELTA]` is within noise (~|1.0|): pitch the null story neutrally — "we measured transfer; here's the measurement; the agent learned the training targets but didn't transfer to a held-out target."
- If `[DELTA]` < 0: pitch as a controlled negative result — "we found that the agent's learning is target-specific, which has implications for how we'd design future molRL training pipelines."

**Never overclaim.** Honest null beats fake positive in research-track judging.

---

## Dependencies for recording

- HF Space URL (need to deploy)
- GitHub URL (live: https://github.com/AnshumanAtrey/pharmarl)
- W&B run with at least Trivial-tier reward curve (need to run training)
- Before/after molecule comparison rendered (`examples/before_after_demo.py` once trained)
- Episode trajectory recording — can capture from the local server live (no training needed)

The DEMO portion (0:30 — 1:00) can be recorded *now* against the local server, before training finishes. Only PROOF (1:00 — 1:20) needs the W&B curve.
