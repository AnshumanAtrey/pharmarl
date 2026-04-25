# PharmaRL Lab Notebook — Decision Log

A running journal of the decisions made during the 18-hour build, with rationale and what we considered but rejected. Reading top-to-bottom should give a reviewer a clear picture of how the env got to its current shape.

Entries are timestamped to the actual commit history.

---

## 2026-04-25 18:04 — Project start (commit `f8ce57b`)

We won Round 1 with SecurityAuditEnv. Round 2 is a fresh build — explicitly *not* a continuation. We picked molecular drug discovery for one reason: it has crisp verifiable rewards (TDC oracles), which the FAQs called out as the highest-EV property of an RL task.

Initial scaffold mirrored the SecurityAuditEnv structure — pydantic models, FastAPI server, episode-keyed state. Initial primary target was SARS-CoV-2 Mpro via TDC's pyscreener-backed docking suite.

**What we considered:**
- AISHA (multi-agent assistance environment) — strong personalized-task fit but harder to validate before deadline
- Patent litigation simulator — novel domain but no off-the-shelf verifier
- Long-horizon code refactoring — overlap with browser RL/coding agent work

We picked PharmaRL because the verifier infrastructure (TDC) was already at Harvard/MIT-grade quality and pip-installable.

---

## 2026-04-25 19:35 — Pivot Stage 1 to DRD2 + session-keyed server (commit `a83e553`)

We initially planned to use SARS-CoV-2 Mpro docking as the primary oracle. After investigating the deployment requirements (OpenBabel + AutoDock Vina native deps in the HF Space container), we pivoted to **DRD2 classifier as the Stage 1 default**.

**Why this matters:** DRD2 is the canonical molRL benchmark — every paper from MolDQN through REINVENT through 2024 GFlowNet variants reports on DRD2. It's pure-Python, ms-latency, no native deps. Stage 2 docking against NSP15 / EGFR / ABL / BACE1 remains opt-in via `PHARMARL_ENABLE_DOCKING=1` for hosts that have the chemistry stack installed.

**Also added:** session-keyed server so a GRPO group of 8 concurrent rollouts each gets isolated state. Without this, parallel rollouts trample each other's `_state`.

---

## 2026-04-25 20:15 — Multi-target oracle routing + held-out JNK3 (commit `56f191b`)

Implemented multi-target binding routing (DRD2 / GSK3B / JNK3) with **JNK3 reserved as held-out**. The pitch claim: train on DRD2 + GSK3B, evaluate transfer on JNK3 — a kinase the agent never sees during training.

**Honest scope:** GSK3B and JNK3 are both Ser/Thr kinases. This is *intra-family* transfer to a novel kinase, not cross-family. We document this caveat clearly in `docs/qa-defense.md` Q13/Q14. Most molRL papers don't run a held-out target measurement at all, so even the intra-family version is a real contribution.

---

## 2026-04-25 21:41 — HF Space deployment + reward red-team + policy regression (commit `bf89b5d`)

Two structural decisions here:

**1. Reward red-team tests.** We added 9 adversarial tests covering empty SMILES, polyaromatic blobs, single carbon, action repetition, disconnected fragments, etc. The OpenEnv FAQ is explicit: *"Do not optimize a reward you have not tried to break yourself first."* We followed that advice and **caught a real bug**: `RDKit.MolFromSmiles("")` returns a 0-atom `Mol` (not `None`), and `QED.qed()` on it returns ~0.34 — so an agent submitting empty output would have gotten composite 0.44. The fix patched all four oracles to reject zero-atom Mols.

**2. `examples/demo.py` regression sanity.** Hand-coded scripted policy vs random uniform — if the scripted policy can't beat random, the reward signal is broken. We ran this against the live HF Space: GSK3B +1.26, JNK3 +0.72, DRD2 +0.27. Reward signal is meaningful. (DRD2 is the smallest margin because the classifier is permissive — random drug-like molecules already score 2-3 on it. Not a bug, just a property of that classifier.)

---

## 2026-04-25 23:49 — Capture untrained LLM baselines (commit `5dd8895`)

Built `examples/llm_baselines/` runners for OpenRouter (Llama 3B/8B/70B) and Gemini (Flash/Pro). Probed 7 policies on the same eval protocol. Total spend: $0.158.

**Most surprising finding: inverted scaling within the Llama family.** 8B (+2.45) > 3B (+1.67) > 70B (+1.19). 70B specifically falls *below random uniform* — it tries to design oversized multi-fragment molecules in single turns and runs head-first into Lipinski + the env's chemistry validator. Parse rate also drops to 97%.

**Why this matters for the pitch:** the env's reward design empirically penalizes capacity-greedy strategies. Random and scripted policies beat 3 of the 4 LLMs. Only Gemini 2.5 Pro clears the scripted baseline cleanly. This is concrete evidence that targeted RL on a small model can outperform raw capacity — the thesis behind training Qwen 1.5B with GRPO on this env.

Full table + reproducing instructions: `docs/baselines.md`.

---

## 2026-04-26 00:25 — Schema drift mechanic (commit `a0c0e34`, PR #5)

Real medicinal-chemistry projects don't have static optimization criteria. A potency push uncovers an ADMET liability; the synthesizability bar tightens before scale-up. Constraints shift mid-development.

We model this directly. With `CurriculumConfig.schema_drift_enabled` (default OFF), the reward weights flip mid-episode according to a sampled drift profile (`static`, `early_admet`, `late_potency`). The agent sees a `drift_warning` in observation when constraints change.

**Rationale for default-OFF:** the headline 200-step GRPO training run targets the Reward Improvement curve on a static reward signal; that's our 20% rubric criterion. Dividing reward signal across drifting weights would make the headline curve noisier. We isolate drift behind a flag — the headline run is reproducible against the static reward, the drift-on demo runs in a separate Colab cell.

**Sub-theme alignment:** This hits the **Patronus AI** sub-theme bonus prize verbatim — *"consumer workflow environments where the underlying data schemas, API contracts, and t&cs/policies/rules change."* To our knowledge, no prior molecular RL env has dynamic reward weights.

---

## 2026-04-26 00:28 — Multi-actor critic (commit `d4b6d25`, PR #6)

Added a deterministic rules-based medicinal-chemist critic. Each post-edit molecule is examined for PAINS substructures (thiocarbonyl, rhodanine, nitroaromatic Michael acceptors), Lipinski-flavored property warnings, reactive group flags (alkyl halide, epoxide, anhydride), and a heavy-atom sanity floor. Verdict is `approve` / `revise` / `reject`; the critique is appended to the next observation's `metadata["critique"]`. The policy can integrate or ignore.

**Why rules-based, not LLM?** Two reasons:
1. **Deterministic** — same molecule always gets the same critique → critic-conditioned training is reproducible. An LLM critic introduces stochasticity that interferes with reward signal cleanliness.
2. **ms latency** — an LLM critic would 10× rollout latency and add API cost. The rules engine is RDKit substructure search; it returns in single-digit ms.

The env contract has a clean seam — a frozen LLM critic can be swapped in here as future work.

**Sub-theme alignment:** Halluminate sub-theme — *"multi-actor environments where an agent interacts with and manages multiple actors."* What matters for the sub-theme is the structured-feedback channel from a separate logical agent, not the implementation of the agent.

---

## 2026-04-26 00:30 — Action-key normalization on /step (commit `eccd98c`)

When Sahil ran a smoke test of the trained Qwen against the live Space, the trained model produced JSON with the wrong key shape: `{"SELFIES": [...], "FRAGMENTS": [...], "ACTION": "TERMINATE"}` instead of the canonical `{"action_type": "TERMINATE"}`. Without intervention, every `/step` call would 422 and the demo would die.

We added **server-side lenient normalization** on `/step`: accept `ACTION` / `Action` / `action` as variants of `action_type`, normalize `POSITION`/`ATOM` parameter keys, strip verbose noise (SELFIES list, FRAGMENTS list, rationale strings) the LLM might wrap around the action. 7 new tests cover the variants. The canonical schema is unchanged; the input boundary is just tolerant.

**Tradeoff:** A strict schema would force the trained model to learn the canonical shape during training. But the training already happened — we can't re-tune the format priming. Lenient input is the pragmatic fix.

---

## 2026-04-26 00:46 — Baseline spectrum doc + Q&A defense (commit `9253a96`)

Folded the LLM baseline finding into the pitch infrastructure. New file `docs/baselines.md` documents the full table + reproducing instructions. New Q18/Q19 in `docs/qa-defense.md` give defenses for the inverted-scaling claim and the empirical reward-not-gameable-by-capacity finding. Pitch script gets a new beat at 1:00–1:10 referencing the baselines before the trained-model proof.

---

## 2026-04-26 01:11 → 01:14 — Stat CI eval, plot infra, more redteam tests

Pre-built the post-training eval pipeline so once the trained Qwen ships, plots are one command away (`examples/eval_with_ci.py` + `examples/plot_results.py`). Added a Colab cell that runs the trained model through 10 episodes per target × 3 targets and saves JSON in the same format the plot script consumes.

Extended redteam tests to 14 total — pinned the Llama 70B failure mode as a regression test: an oversized multi-fragment molecule (verbatim from 70B's actual output) must score modestly. If we ever change the reward function in a way that makes capacity-greedy strategies viable, this test catches it.

---

## 2026-04-26 — Rubric refactor (this commit)

Refactored monolithic `terminal_reward()` into composable rubrics (`server/rubrics.py`). Each oracle is a small `Rubric` class with `score(smiles)`; they compose via `*` (weight) and `+` (sum) operators. The OpenEnv judging guide explicitly rewards *"composable rubrics > monolithic scoring"* — this is structural compliance.

`terminal_reward()` output values are unchanged; the rubric path produces identical numbers. Tests pin this with explicit before/after comparisons on haloperidol and aspirin.

11 new tests in `tests/test_rubrics.py`.

---

## What we rejected — and why

- **"Cure all diseases" framing.** Research-track judges read this and dock instantly. We pitched "first OpenEnv-native LLM-as-policy molecular env" instead — sober, defensible, and what the env actually delivers.
- **LLM-based critic agent.** Latency, API cost, non-determinism. Rules-based critic gets the same Halluminate sub-theme hit at 100× the speed.
- **BBB_Martins as cross-family held-out target.** Auditor recommended this; we declined because BBB correlates strongly with QED + Lipinski (both already in our composite reward). A trained model would automatically score higher on BBB regardless of whether real binding-skill transferred — fake-positive transfer signal. Same critique applies to CYP-toxicity classifiers. Documented in `docs/qa-defense.md` Q15.
- **Lipinski → pharmacophore → binding curriculum stages.** Would contaminate the held-out kinase test (we'd be explicitly teaching kinase pharmacophores then evaluating on a kinase). Existing trivial → easy → hard curriculum is principles-first enough.
- **Foundation model pretraining.** Impossible in 18 hours regardless of compute budget. The Qwen 1.5B already has chemistry priors from web pretraining; our env's job is to elicit and refine them, not to implant chemistry from zero.
- **Domain pivot.** Considered late-night switching from molRL to a less-trodden domain. Rejected because PharmaRL's scaffold + tests + deployment was already working — pivoting would have meant shipping nothing.
- **Reaction-aware actions** (Suzuki coupling, amide formation as action types). Considered as Tier 2 but skipped — it's 6-10h of careful chemistry work that would risk the existing test suite without a clear sub-theme bonus payoff.

---

## What's still TODO at submission

| Item | Owner |
|---|---|
| Final training run + W&B link | Sahil |
| Held-out JNK3 statistical CI numbers | Sahil (Cell 13 in Colab notebook) |
| Plot generation from trained_qwen.json | Anshuman runs `examples/plot_results.py` |
| README results section URL fills | Vijay |
| HF blog post publish | Vijay |
| 90s pitch video record | Vijay |
| Slide deck | Vijay |

The lab notebook will be updated as these land.
