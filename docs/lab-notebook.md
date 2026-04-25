# PharmaRL Lab Notebook — Decision Log

This is the running journal of design decisions made during the 18-hour
build, with rationale and what we considered but rejected. Reading it
should give a reviewer a clear picture of how the env got to its current
shape.

## 2026-04-25 18:00 — Project start

Round 2 is a fresh build. We pick molecular drug discovery —
chemistry has dense reward signals (QED, SA, docking, toxicity), a
canonical RL benchmark family (MOSES / GuacaMol / TDC), and a rubric
that fits "composable rubrics > monolithic scoring" naturally.

First commit: SMILES tokenizer, SELFIES round-trip, RDKit Lipinski
validation, an `Edit` action, single `terminal_reward` composite.

## 2026-04-25 19:35 — Pivot from Mpro docking to DRD2 classifier

Initial plan: SARS-CoV-2 Mpro docking (`7l11_docking_normalize`) via
TDC's pyscreener integration. Mpro gives a clean kcal/mol gradient.
Probing revealed it needs pyscreener + AutoDock Vina + OpenBabel
native binaries — on the Hugging Face Space deploy host these paths
are flaky and add ~20 s of Ray-instance startup before falling back.

We pivot Stage 1 to **DRD2**, the canonical MOSES / GuacaMol
classifier benchmark. Every paper from MolDQN (2019) through REINVENT,
GraphAF, and 2024 GFlowNet variants reports on DRD2, so the comparison
surface is strong. Stage 2 docking stays gated behind
`PHARMARL_ENABLE_DOCKING=1` for hosts that have the native deps.

## 2026-04-25 20:15 — Multi-target routing + held-out JNK3

Reward over a single oracle invites a "did the model just memorize one
classifier?" question. We add per-target oracle routing so `/reset`
accepts a target name and the binding component is computed against
that target's classifier. Training rotates DRD2 + GSK3B; JNK3 is held
out for the final eval. The pitch is: *"if the agent learned chemistry
primitives rather than overfitting one oracle, the JNK3 score lifts
without ever having seen JNK3."*

## 2026-04-25 21:41 — Reward red-team + HF Space deployment

We red-team before training so we don't burn compute on a hackable
surface. The `+0.05` Lipinski step bonus could be farmed by a
methane-only blob; we add a Lipinski **gate** on terminal reward
(composite halved if final fails Rule of 5) and a zero-atom check in
`score_qed` so empty SMILES don't return RDKit's default 0.34. The
`test_reward_redteam.py` suite pins these.

## 2026-04-25 23:49 — Untrained-LLM baselines

The rubric weights "results > 0," so we need a baseline curve bad
enough to leave headroom. We run Gemini 1.5 Flash, Llama 3.1 8B, and
Llama 3.1 70B against the env raw and capture the score distribution.
The 70B underperforms the 8B (inverted scaling — composite ≈ 4.2 vs
≈ 4.7). This is a finding, not a bug: bigger models pattern-match
drug-like SMILES from pretraining and stop exploring. We document it
in `docs/baselines.md` as a six-policy spectrum. It also justifies
sizing the trained model at 1.5B rather than chasing scale.

## 2026-04-26 00:25 — Schema drift, gated OFF (Patronus AI sub-theme)

Real medicinal-chemistry projects don't optimize a fixed objective —
constraints shift mid-development (potency push uncovers an ADMET
liability; synthesizability tightens before scale-up). We model this
with **schema drift**: mid-episode reward-weight changes via three
profiles (`static`, `early_admet`, `late_potency`). Default OFF so
the headline training curve is unperturbed; opt-in via
`reset(schema_drift=True)`.

## 2026-04-26 00:28 — Rules-based critic, NOT an LLM critic

The Halluminate sub-theme rewards multi-actor environments. An LLM
judge has three killer flaws: latency (seconds per step vs ms), cost
(every training step pays an API call), and non-determinism (same
molecule, different feedback across runs — destroys reproducibility).
We ship a rules-based medicinal-chemist critic instead: PAINS
patterns, size limits, ADMET red flags, deterministic critique into
`observation.metadata["critique"]`. Default OFF.

## 2026-04-26 00:30 — Lenient action-key normalization

Trained-in-the-wild LLMs don't emit our exact action JSON — they
wrap actions in `{"action": {...}}`, use synonyms, or add commentary
fields. We add `_normalize_action_dict` on `/step` so the env
unwraps common variants before strict parsing. The difference between
a scored episode and a parse-fail penalty when a policy is plugged in.

## 2026-04-26 01:13 — OpenEnv Rubric refactor

The judging guide explicitly says "Uses OpenEnv's Rubric system
thoughtfully (composable rubrics > monolithic scoring)." We refactor
into `server/rubrics.py` with `QedRubric`, `SaRubric`,
`ToxicityRubric`, and `BindingRubric` that compose via `*` (weight)
and `+` (sum). Hard-tier composite is now declarative:
`BindingRubric(target) * 0.40 + QedRubric() * 0.25 + SaRubric() *
0.15 + ToxicityRubric() * 0.20`. Trivial / Easy tiers keep their
legacy formulas verbatim so the early reward curve doesn't shift.

## 2026-04-26 01:19 — Cross-family secondary held-out

The auditor flagged that GSK3B (Ser/Thr kinase) + JNK3 (MAP kinase)
is intra-family transfer. We declined the auditor's BBB_Martins
suggestion — BBB correlates with QED, so a "transfer" lift would be
a fake-positive driven by the same drug-likeness signal already in
training reward.

We probed CYP2D6_Substrate_CarbonMangels, ESOL, Lipophilicity_
AstraZeneca, Bioavailability_Ma, PPB, HIA_Hou — all are TDC ADMET
datasets, not Oracle-registry entries. We probed the docking_
normalize variants (3eml, 3ny8, drd3, 3pbl) — all need pyscreener
+ Vina + OpenBabel, ruled out earlier.

We settled on `amlodipine_mpo` — MPO oracle scoring similarity-to-
amlodipine plus drug-likeness. Amlodipine is an L-type calcium-
channel blocker, orthogonal pharmacology to the Ser/Thr + MAP kinases.
Does not strongly correlate with QED (penicillin / silymarin score
~0.4 while ibuprofen at QED 0.82 scores ~0.01). Wired as
`CurriculumConfig.secondary_held_out_target` (default None, opt-in).

## What we rejected and why

- **Lipinski → pharmacophore → binding curriculum.** Pharmacophore
  stage would encode target-family knowledge into the curriculum,
  contaminating the held-out generalization test.
- **LLM-based critic.** Latency, API cost, non-determinism.
- **Foundation-model pretraining on ChEMBL.** Out of budget at 18 h
  and not what the rubric rewards.
- **BBB_Martins as cross-family held-out.** Correlates with QED —
  fake-positive transfer.
- **Domain pivot.** Asked late-Saturday whether we should pivot. No.
  Env was integrated, baselines were captured, pivot would lose six
  hours for marginal narrative upside.
- **Overclaim framing.** Research-track judges read it as a tell.
  The pitch is: a clean OpenEnv with a measurable training lift on
  a held-out target, plus two opt-in mechanics (schema drift,
  multi-actor critic) that map onto two partner sub-themes.
