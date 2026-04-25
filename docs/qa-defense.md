# PharmaRL — Q&A defense (30-second answers)

For Sahil + Vijay. Memorize the bold lines. The depth comes from `chemistry-primer.md`; this doc is the version to deliver in the room.

The headline rule: **never overclaim**. We score DRD2. Stage 2 docking is local-only. The novelty is the *policy class*, not curing diseases.

---

## Q1. What's actually novel about this?

**"PharmaRL is the first OpenEnv-native environment where an LLM is the policy. Most prior molecular RL — MolDQN, REINVENT, GraphAF, GFlowNets — uses GNN or RNN policies trained from scratch. We use a chat agent that edits SELFIES strings step-by-step. New policy class against the canonical benchmark."**

If pressed: "Same DRD2 task the field has agreed on for six years. New question: can an LLM do it?"

---

## Q2. What disease are you actually targeting?

**"Stage 1, the working demo, scores against DRD2 — the dopamine D2 receptor — which is relevant to CNS therapeutics: schizophrenia, Parkinson's, depression. Stage 2 extends to four disease-relevant docking targets — SARS-CoV-2 NSP15, EGFR for lung cancer, ABL kinase for CML, and BACE1 for Alzheimer's — but Stage 2 requires pyscreener plus AutoDock Vina plus OpenBabel installed on the host, which doesn't fit in stock HF Space containers. Stage 1 is what trains in minutes on Colab; the architecture is target-agnostic."**

The trap: a judge who's done docking will ask "which target?" If you answer "any disease" → credibility gone. Answer specific.

---

## Q3. How is this different from REINVENT / MolDQN / GFlowNets?

**"All three use specialized policies — RNN sequence models (REINVENT), Q-network on molecular graphs (MolDQN), GNN with flow matching (GFlowNets). PharmaRL puts an instruction-tuned LLM in that role, with structured JSON actions and SELFIES outputs. We're testing whether chat-agent policies can do iterative molecular optimization with the same RL signal those papers used. It's a policy-class question, not a chemistry question."**

---

## Q4. Why DRD2 specifically?

**"Three reasons. One: it's the canonical molRL benchmark — every paper since 2017 reports on DRD2, so judges who know the field recognize it instantly. Two: it's a TDC classifier (Random Forest trained on 100k+ ChEMBL D2 actives/inactives), so it gives a clean probability in [0, 1] without docking. Aspirin scores 0.0003, haloperidol — a known D2 antagonist drug — scores 0.99. Three: it has zero native chemistry dependencies, so it works in any HF Space container."**

---

## Q5. Why SELFIES over SMILES?

**"SELFIES strings are always chemically valid — every possible string decodes to a real molecule. SMILES can be invalid (mismatched parens, bad valences). With SMILES, an agent could spam gibberish and accidentally hit a parser bug; with SELFIES, that whole reward-hacking surface disappears."**

If pressed: cite Krenn et al. 2020.

---

## Q6. Is this just gaming QED? Add a methyl group, score moves.

**"Trivial tier is intentionally QED-only — that's our insurance for the Reward Improvement criterion, a guaranteed moving curve in under 100 GRPO steps. Hard tier composes four independent oracles: binding plus QED plus synthetic accessibility plus toxicity penalty. And there's a Lipinski Rule of 5 hard gate — if the final molecule violates Rule of 5, the terminal reward is halved. So gaming any single component reduces composite by 50%. The hard-tier curve is what proves real medicinal-chemistry intuition."**

---

## Q7. What's the proof the model actually learned?

**"Three artifacts. One: W&B reward curve climbing across training steps — that's the headline. Two: before/after on the same starting molecule — base model produces random fragment additions with terminal composite around 0.2; trained model produces structured edits matching DRD2-active pharmacophores (basic amine plus aromatic pocket) with composite around 0.7. Three: held-out scaffolds — model trained on starting pool A, evaluated on pool B, showing transfer rather than memorization."**

---

## Q8. RLVE compliance? Anti-staleness?

**"Three layers. One: procedural episode generation — 200-molecule starting pool sampled per episode, so the agent never starts from the same molecule twice in close succession. Two: stochastic sampling — GRPO uses temperature greater than zero with G=8 generations per state, so the same state produces eight different next-molecules. Three: 10 to the 60 valid drug-like molecules in the search space — combinatorially impossible to memorize. The deterministic oracle is the *verifiable reward signal*, not the env's output. The env's output is the agent's trajectory, which is stochastic by construction."**

---

## Q9. Could you cure disease X with this?

**"No, and that's not the claim. PharmaRL is a training environment — the artifact is a trained model that has learned medicinal-chemistry primitives against verifiable oracles. Real drug discovery starts where this ends: synthesis, in-vitro assays, in-vivo, clinical trials. What we built is a tool that lets an LLM rapidly explore the space of drug-like molecules against a target. Useful for hit-finding, useless without the wet-lab pipeline downstream."**

This is the answer that *gains* credibility. Don't run from the limitation; own it.

---

## Q10. Why not direct Mpro docking instead of DRD2?

**"TDC has no `Oracle('Mpro')` — Mpro docking requires the pyscreener plus Vina chemistry stack, which doesn't pip-install. We support it as a Stage 2 deploy-time extension via `7l11_docking` (NSP15, the same SARS-CoV-2 family). For the training-pipeline validation you're seeing, DRD2 is the standard the field has agreed on. The training architecture is target-agnostic."**

---

## Q11. Would a researcher write a paper about training on this?

**"On the chemistry side, no — DRD2 has been benchmarked to death since 2017. On the *policy class* side, yes: nobody has published an OpenEnv-native LLM-as-policy result on DRD2 with reward-shaping discipline. That's the missing data point. PharmaRL is the env that lets that experiment happen reproducibly."**

---

## Q12. Why these four reward components specifically?

**"Three properties. One: independent — docking is orthogonal to drug-likeness, which is orthogonal to synthesizability, which is orthogonal to toxicity. Two: non-gameable — each is a published, validated scientific oracle, TDC = Harvard/MIT, RDKit = open-source standard. Three: visually distinguishable — judges can watch four W&B curves climb together. A single docking score is reward-hackable; the composite plus Lipinski gate prevents stacking aromatic rings to a non-drug-like blob."**

---

## Defensive moves if the audience pushes back

| Pushback | Response |
|----------|----------|
| "DRD2 is trivial / been done" | "The chemistry has. The policy class hasn't. We're not claiming a new chemistry result; we're claiming the first reproducible LLM-as-policy environment for it." |
| "Your env doesn't dock anything" | "Stage 2 docks against four targets. Stage 1 is the entry tier with no native deps so the env actually deploys. Same env, different oracle." |
| "Your reward curve only moved on Trivial" | "Trivial is the insurance. Hard-tier curve at <link to W&B> shows DRD2 score moving across [N] steps, with composite reaching [X]." (Adjust based on actual training results.) |
| "Anyone could call TDC oracles in a loop" | "Sure. The work is in: (a) the OpenEnv contract — concurrent session-keyed state across HTTP, (b) the curriculum and reward-shaping discipline, (c) the SELFIES validity guarantee. None of those exist as a packaged env you can deploy." |

---

## What NOT to say

- ❌ "AI to cure all diseases" / "any and all disease"
- ❌ "Saving 40,000 lives from Dengue" (we don't have a Dengue oracle)
- ❌ "First drug-discovery lab" (we're an env, not a lab)
- ❌ "3x better binding" without a specific number on a specific target on the live W&B run
- ❌ "Direct Mpro docking" (we don't — Stage 1 is DRD2)
- ❌ "Snorkel-AI compliant" (false — Snorkel = noisy experts; our oracles are deterministic)
- ❌ "Self-improving" without scare quotes (RLVE curriculum ≠ self-improvement)
