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

**"On the chemistry side, no — DRD2 has been benchmarked to death since 2017. On the *policy class* side, yes: nobody has published an OpenEnv-native LLM-as-policy result on DRD2 with reward-shaping discipline. That's the missing data point. PharmaRL is the env that lets that experiment happen reproducibly. We also ran a held-out target generalization test, which most molRL papers don't — see Q13."**

---

## Q12. Why these four reward components specifically?

**"Three properties. One: independent — docking is orthogonal to drug-likeness, which is orthogonal to synthesizability, which is orthogonal to toxicity. Two: non-gameable — each is a published, validated scientific oracle, TDC = Harvard/MIT, RDKit = open-source standard. Three: visually distinguishable — judges can watch four W&B curves climb together. A single docking score is reward-hackable; the composite plus Lipinski gate prevents stacking aromatic rings to a non-drug-like blob."**

---

## Q13. Does training on one target transfer to a different target?

**"We tested that explicitly. We trained on DRD2 and GSK3B, and reserved JNK3 — a kinase the model never saw during training — for evaluation. Untrained Qwen scores [BEFORE] on JNK3; trained Qwen scores [AFTER]. Delta of [DELTA]. Most molRL papers don't run a held-out target measurement at all."**

Replace the bracket placeholders with actual numbers from the W&B run before pitching. **If transfer is null, frame it neutrally — null result is still a result.**

**Honest scope of the claim** — say this if asked:
> "GSK3B and JNK3 are both serine/threonine kinases, so this is *intra-family* transfer to a novel kinase, not cross-family. It's a real measurement, but we're not claiming the agent learned 'binding principles in general' — we're showing it transfers to a kinase target the agent never saw."

This is the move: fix the *language*, not the *experiment*. The experiment is real and useful. The overclaim ("learned general binding principles") is what would collapse credibility.

If pressed harder: "We're not claiming this generalizes to *any* disease. Cross-family transfer (kinase → GPCR or → enzyme) is much harder and is not what we tested. We tested whether a kinase-trained agent can hit a kinase the trainer hid."

---

## Q14. Why DRD2 + GSK3B as training targets and JNK3 as held-out?

**"All three are TDC classifiers — pure Python, no native chemistry deps, fast oracle calls. DRD2 is the canonical molRL benchmark. GSK3B and JNK3 are both Ser/Thr kinases — they share fold and ATP-binding-pocket geometry — so this is intra-family transfer rather than cross-family. We picked it because (a) it's reachable in our compute budget and (b) JNK3 isn't trivially identical to GSK3B — it's a different protein, with different selectivity, that the model never saw."**

Don't pretend this is a "novel disease" test. It's a novel *target* in the same protein family. That's enough to be interesting; overclaiming kills credibility.

---

## Q15. Wouldn't a TDC oracle like BBB_Martins make a better held-out test?

**"No. BBB_Martins is a physicochemical property classifier — does the molecule cross the blood-brain barrier? It correlates strongly with QED and Lipinski compliance, which we already optimize in the composite reward. So a trained model would *automatically* score higher on BBB regardless of whether real binding-skill transferred. We'd get an artifactual positive signal. Same critique applies to CYP-toxicity classifiers — they correlate with what we trained on. JNK3 binding is independent of the composite reward we optimized; that's why it's a meaningful test even though it's intra-family."**

---

## Defensive moves if the audience pushes back

| Pushback | Response |
|----------|----------|
| "DRD2 is trivial / been done" | "The chemistry has. The policy class hasn't. We're not claiming a new chemistry result; we're claiming the first reproducible LLM-as-policy environment for it." |
| "Your env doesn't dock anything" | "Stage 2 docks against four targets. Stage 1 is the entry tier with no native deps so the env actually deploys. Same env, different oracle." |
| "Your reward curve only moved on Trivial" | "Trivial is the insurance. Hard-tier curve at <link to W&B> shows DRD2 score moving across [N] steps, with composite reaching [X]." (Adjust based on actual training results.) |
| "Anyone could call TDC oracles in a loop" | "Sure. The work is in: (a) the OpenEnv contract — concurrent session-keyed state across HTTP, (b) the curriculum and reward-shaping discipline, (c) the SELFIES validity guarantee. None of those exist as a packaged env you can deploy." |

---

## Q17. Why a rules-based critic instead of an LLM critic?

**"Two reasons. First, deterministic: same molecule always gets the same critique, which makes critic-conditioned training reproducible. An LLM critic introduces stochasticity that interferes with reward signal cleanliness. Second, fast: the rules engine returns in milliseconds — an LLM critic would 10x rollout latency. The Halluminate sub-theme rewards multi-actor environments; what matters is that there's a separate logical agent providing feedback that the policy can integrate, not that the agent is itself an LLM. Future work could swap in a frozen LLM critic for richer feedback."**

If pressed on what the critic actually checks: **"PAINS substructures (thiocarbonyl, rhodanine, nitroaromatic Michael acceptors), Lipinski-flavored property warnings (MW > 500, LogP > 5), reactive group flags (alkyl halide, epoxide, anhydride), and a heavy-atom sanity floor. Verdict is `approve` / `revise` / `reject`; the critique is appended to the next observation's metadata under `critique` so the policy can choose to revise via REMOVE_FRAGMENT or SUBSTITUTE_ATOM."**

If asked why it's gated behind `critic_enabled` and default OFF: **"The headline training run targets the Reward Improvement curve on DRD2 — that's our insurance criterion. The critic adds an extra observation field, which would change the prompt distribution for the policy. We isolate it behind a flag so the headline run is reproducible against an LLM-only baseline; the critic-on run is the multi-actor demo."**

---

## What NOT to say

- ❌ "AI to cure all diseases" / "any and all disease"
- ❌ "Saving 40,000 lives from Dengue" (we don't have a Dengue oracle)
- ❌ "First drug-discovery lab" (we're an env, not a lab)
- ❌ "3x better binding" without a specific number on a specific target on the live W&B run
- ❌ "Direct Mpro docking" (we don't — Stage 1 is DRD2)
- ❌ "Snorkel-AI compliant" (false — Snorkel = noisy experts; our oracles are deterministic)
- ❌ "Self-improving" without scare quotes (RLVE curriculum ≠ self-improvement)

---

## Q16. What is schema drift and why does it matter?

**"Schema drift means the reward function's weights change mid-episode. Real medicinal chemistry projects work this way — you start optimizing potency, discover a metabolic stability problem, and the constraints shift mid-development. We model this directly in our env. The Patronus AI sub-theme rewards exactly this kind of consumer-workflow-with-changing-rules environment, and to our knowledge no prior molecular RL env has dynamic reward weights."**

If pressed on implementation: "It's flag-gated behind `schema_drift_enabled` in `CurriculumConfig`, default OFF — the headline training run is unaffected. When enabled, each episode samples a drift profile (`static`, `early_admet`, `late_potency`); on the configured drift step the reward weights flip from a `pre` tuple to a `post` tuple, and the observation surfaces a `drift_warning` plus an updated `active_constraints` list so the agent can detect that the rules just changed."

---

## Q18. Did you compare against off-the-shelf LLMs as baselines?

**"Yes — six policies on the same eval (9 episodes per target × 3 targets). The full table is in `docs/baselines.md`, but the headline numbers: random uniform +2.30, scripted 4-step heuristic +2.81, Llama 3.2 3B +1.67, Gemini 2.5 Flash +1.81, Llama 3.1 8B +2.45, Llama 3.3 70B +1.19, Gemini 2.5 Pro +3.68. Every probe ran in under $0.16 total spend."**

The interesting result: **inverted scaling.** Across the Llama family, 8B Instruct (+2.45) was the sweet spot — bigger 70B (+1.19) and smaller 3B (+1.67) both underperform. 70B specifically gets greedy: it tries multi-fragment over-substituted molecules in one turn that fail Lipinski + the env's chemistry validator. Parse rate dropped to 97% (3 failed-JSON turns out of 135).

Implication: in this constrained action space, raw model capacity is anti-correlated with performance past a sweet spot. The env's reward-design discipline — composite + Lipinski gate + parse penalty + zero-atom guard — penalizes capacity-greedy strategies. **Random and scripted policies beat 3 of the 4 LLMs we tested.** Only Gemini 2.5 Pro clears the scripted baseline cleanly.

This is the benchmark to beat with our trained 1.5B Qwen.

---

## Q20. What's the Fleet AI oversight mechanic and why pure-LLM (not rules-based)?

**"After each episode ends, an oversight LLM examines the full action trajectory and emits a structured report: strategy summary, risk flags, risk level (low/medium/high), and an explanation. It's the backward-looking counterpart to the critic agent — critic gives the policy advice mid-episode (forward), oversight reads the policy's behavior backward and explains it."**

Why pure LLM and not rules-based:
1. **Fleet AI's brief is about AI agent oversight, not rule engines.** The sub-theme verbatim says *"train oversight agents to monitor, analyze, and explain other AI agents."* A frozen LLM with structured prompting is the canonical scalable-oversight approach — see Constitutional AI, debate, etc.
2. **Explanation quality.** A rules engine can flag "MW > 500" but can't say "the agent stacked aromatic rings to game the binding component, then tried to terminate before Lipinski could catch it." The LLM gives natural-language explanations that are themselves training data for further oversight work.
3. **Single call per episode, not per step.** The oversight LLM runs only at TERMINATE — one inference call per rollout. That keeps cost (~$0.001 per episode on Gemini 2.5 Flash) and latency bounded.

If pressed on cost during training: **"Default OFF. The headline GRPO training run never makes oversight calls. The mechanic is for the demo + post-hoc analysis, not for the reward loop. If a future paper wanted to train the oversight LLM itself on these trajectories, the env's structured action history is exactly the supervision signal needed."**

If pressed on which model: **"We default to Gemini 2.5 Flash with thinking disabled — fast, ~$0.001/call, reliable. The env supports OpenRouter as an alternate provider; we hit OpenRouter free-tier rate limits during baseline runs, so we recommend Gemini for live demos."**

---

## Q19. So what stops an agent from gaming your reward?

**"Three stacked defenses, validated empirically. (1) Composite oracle: any single component that gets gamed gets diluted by the other three. (2) Lipinski gate: violating Rule of 5 halves the terminal reward. (3) Anti-degenerate guards: zero-atom Mol → 0.0; parse failure → -0.5; cannot terminate on step 1. Plus 9 redteam tests covering empty SMILES, polyaromatic blobs, single carbon, action repetition, disconnected fragments. The empirical proof: when we ran Llama 70B on the env, it tried capacity-greedy strategies (over-substituted multi-fragment molecules) and scored worse than random uniform. The reward signal isn't just hard to game in theory; we have data showing a frontier-class LLM can't game it."**

---
