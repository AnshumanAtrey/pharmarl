# Related Work: RL and ML for Molecular Design

This survey covers ~8 years of generative-and-RL work on small-molecule design, from the original SMILES-RNN-RL papers through 2024 transformer-on-SMILES models, and explicitly catalogs the failure modes that motivate an LLM-as-policy paradigm. Each entry is anchored on the primary citation (no survey blogs); numerics are taken from the original papers.

## REINVENT (Olivecrona et al., 2017)

**Citation [1]:** Olivecrona, M., Blaschke, T., Engkvist, O., & Chen, H. (2017). "Molecular de-novo design through deep reinforcement learning." *Journal of Cheminformatics*, 9:48. arXiv:1704.07555.

**Architecture.** A 3-layer GRU recurrent network with 1024 hidden units per layer, operating on canonicalized SMILES tokens. The "Prior" is trained by maximum-likelihood next-token prediction over 1.5M ChEMBL molecules with 10–50 heavy atoms. Adam at lr=1e-3, batch size 128, gradient clip [-3, 3], 50,000 steps.

**Reward / training objective.** The signature contribution is *augmented likelihood*: rather than naive REINFORCE on the scoring function (which collapses to repetitive single-token outputs like all-carbon strings), the Agent network is trained to minimize squared error between its log-probability and an *augmented prior* `log P_aug(A) = log P_prior(A) + sigma * S(A)` where `S(A)` is the task-specific scoring function and `sigma` weights how aggressively to deviate from the Prior. This is functionally a soft-KL anchor that prevents reward hacking while still moving probability mass toward high-reward sequences.

**Reported results.** On a DRD2 (dopamine D2) actives task, >95% of generated molecules were predicted active; the Agent recovered held-out test-set actives with a 250x enrichment over the Prior. A "no-sulfur" toy task hit 98% compliance. Celecoxib analog generation worked even when Celecoxib-similar compounds were stripped from training.

## REINVENT 2.0 (Blaschke et al., 2020)

**Citation [2]:** Blaschke, T., Arús-Pous, J., Chen, H., Margreitter, C., Tyrchan, C., Engkvist, O., Papadopoulos, K., & Patronov, A. (2020). "REINVENT 2.0: An AI Tool for De Novo Drug Design." *Journal of Chemical Information and Modeling*, 60(12), 5918–5922.

**What changed.** Productionized version aimed at industrial drug-discovery workflows. Key additions: (1) a *diversity filter* / molecular memory that penalizes scaffolds the Agent has already exploited, directly attacking the mode-collapse failure mode of the 2017 model; (2) a modular scoring component framework (QED, predictive QSAR models, custom alerts) combinable with arbitrary aggregators; (3) scaffold-decoration and link-design generators built on the same Prior/Agent paradigm but conditioned on input fragments. Code at github.com/MolecularAI/Reinvent.

## REINVENT 4 (Loeffler et al., 2024)

**Citation [3]:** Loeffler, H. H., He, J., Tibo, A., Janet, J. P., Voronov, A., Mervin, L. H., & Engkvist, O. (2024). "REINVENT 4: Modern AI–driven generative molecule design." *Journal of Cheminformatics*, 16:20.

**What changed.** Adds transformer-based generators (Mol2Mol for paired molecule optimization within a Tanimoto-similarity envelope) alongside the legacy RNN. Five generator modes co-exist: Reinvent (de novo), LibInvent (R-group), LinkInvent (linker design), Mol2Mol (transformer), Pepinvent (peptide). The recommended RL objective is renamed *DAP* (Difference between Augmented and Posterior): `log P_aug(T) = log P_prior(T) + sigma * S(T)` — i.e., the same 2017 augmented-likelihood loss, now cast as the default among multiple available RL strategies. Adds *staged learning* (curriculum across multiple RL stages with different scoring weights) and a TOML-configurable plugin scoring subsystem. The legacy Prior is unchanged in spirit; the framing is "REINFORCE-family with KL anchor and curriculum scaffolding."

## MolDQN (Zhou et al., 2019)

**Citation [4]:** Zhou, Z., Kearnes, S., Li, L., Zare, R. N., & Riley, P. (2019). "Optimization of Molecules via Deep Reinforcement Learning." *Scientific Reports*, 9:10752. arXiv:1810.08678.

**Architecture.** A pure RL approach with no pretrained language prior. The MDP operates *directly on molecular graphs* with three chemically-validated action types per step: atom addition, bond addition, bond removal. State = Morgan fingerprint (radius 3, 2048-bit) concatenated with remaining-step counter; fed into a 4-layer MLP `[1024, 512, 128, 32]`. Trained with double Q-learning + bootstrapped DQN (H independent Q-heads as a randomized value function for epistemic exploration). No SMILES anywhere in the loop, so 100% of outputs are by construction valid.

**Reward.** Single-objective reward is the property at the terminal step (penalized logP or QED). Multi-objective variant uses `r = w * Sim(s, s_0) + (1-w) * QED(s)` to enforce similarity to a starting molecule. Reward only delivered at episode end (no shaping).

**Reported results.** Penalized logP: **11.84** (vs. GCPN's 7.98 — at the time, SOTA, but achieved by long thin alkane-like chains, an early example of unconstrained-logP being a degenerate target). QED: **0.948** (matching the global maximum, also matched by GCPN). Constrained optimization at delta=0.4 similarity: 3.37 +/- 1.62 logP improvement, 100% success.

**Why it didn't generalize.** (1) Morgan-fingerprint state is lossy — graph structure is destroyed, so the agent struggles on tasks requiring stereochemistry or fine substructural reasoning. (2) The action space (add atom/add bond/remove bond) cannot reach all valid molecules without long combinatorial sequences, leading to slow convergence on multi-ring drug-like targets. (3) No language prior means the agent re-discovers chemistry from scratch every run; any task that benefits from medicinal-chemistry priors (synthesizable scaffolds, common heterocycles) needs hand-engineered reward terms. (4) Penalized logP itself was widely shown post-hoc to be a degenerate benchmark — an unbounded-chain attractor — and MolDQN's "wins" on it are now considered evidence of reward-hacking the metric rather than chemistry capability.

## GraphAF (Shi et al., 2020)

**Citation [5]:** Shi, C., Xu, M., Zhu, Z., Zhang, W., Zhang, M., & Tang, J. (2020). "GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation." *ICLR 2020*. arXiv:2001.09382.

**Architecture.** Normalizing flow defined autoregressively over graph elements: at each step it samples a node type, then iteratively samples bond types to existing nodes, all under an invertible flow that gives exact likelihood. The flow uses graph convolutions to condition each step on the partial graph. Importantly, valency checks are applied at sampling time as masking, so 100% of outputs are valid when chemical rules are enabled (68% when disabled — a useful ablation showing how much "validity" comes from the model vs. the rule overlay).

**Training & RL fine-tuning.** Pretrained by maximum likelihood on ZINC250k. For property optimization, fine-tuned with policy gradient (REINFORCE-style) on penalized logP / QED rewards. Reports SOTA penalized logP and constrained-optimization scores in 2020, with ~2x training speedup over GCPN.

## MARS (Xie et al., 2021)

**Citation [6]:** Xie, Y., Shi, C., Zhou, H., Yang, Y., Zhang, W., Yu, Y., & Li, L. (2021). "MARS: Markov Molecular Sampling for Multi-objective Drug Discovery." *ICLR 2021* spotlight. arXiv:2103.10432.

**Architecture.** Not RL-as-policy. Instead, an annealed MCMC sampler over molecular graphs where each fragment-level edit is proposed by a learned graph neural network. The GNN is trained on-the-fly using the very samples MCMC accepts, so policy and exploration co-evolve. The MCMC accepts/rejects via a Metropolis-Hastings ratio derived from a multi-objective scalarized reward.

**Reward.** Composite of GSK3-beta activity, JNK3 activity, QED, and SA score — the standard 4-target benchmark. MARS was state-of-the-art on the simultaneous 4-objective task at publication, where pure-RL methods like REINVENT were found to compromise heavily on either bioactivity or drug-likeness. The lesson: explicit MCMC + adaptive proposal beats greedy RL when the reward landscape has many disjoint optima.

## JT-VAE (Jin et al., 2018)

**Citation [7]:** Jin, W., Barzilay, R., & Jaakkola, T. (2018). "Junction Tree Variational Autoencoder for Molecular Graph Generation." *ICML 2018*. arXiv:1802.04364.

**Architecture.** Two-stage hierarchical VAE. Encoder produces dual latents `z_T` (junction-tree topology over chemical substructures) and `z_G` (atom-level graph features). Decoder first samples a junction tree from `z_T` over a learned vocabulary of **780 substructures** (rings + bonds extracted from ZINC), then assembles atoms using `z_G`. Training is standard VAE ELBO with KL term.

**Latent-space optimization paradigm.** JT-VAE introduced the dominant pre-RL paradigm: train a VAE for unconditional molecule generation, then optimize molecular property by Bayesian optimization (Gaussian process surrogate) directly in the continuous latent. This was the standard "no-RL" molecular optimization method for ~2 years.

**Results.** 100% validity from prior sampling (decoder cannot produce invalid molecules by construction). Penalized logP: best 5.30, runner-up 4.93 — beaten later by GCPN/MolDQN/GraphAF, but at the time SOTA. Constrained optimization at delta=0.4: 0.84 +/- avg property improvement, 83.6% success rate. **Failure mode discovered downstream:** latent-space optimization is brittle — most decoded points from optimized latents fail synthesizability filters; <1% pass quality filters in some follow-ups.

## GCPN (You et al., 2018)

**Citation [8]:** You, J., Liu, B., Ying, R., Pande, V., & Leskovec, J. (2018). "Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation." *NeurIPS 2018*. arXiv:1806.02473.

**Architecture.** A graph-convolutional policy that builds molecules atom-by-atom. The action at each step is a tuple `(first_node, second_node, edge_type, stop)`, where the policy attends over both the existing partial molecule and a "scaffold" set of atom candidates. Trained with proximal policy optimization (PPO).

**Reward.** Composite of (1) chemical validity, (2) the property objective (penalized logP, QED, etc.), and (3) an adversarial term from a discriminator trained to separate generated from real ZINC molecules — penalizing distributionally-weird outputs (a partial defense against reward hacking).

**Results.** 61% improvement on chemical property optimization vs. JT-VAE; 184% improvement on constrained property optimization. QED 0.948; penalized logP 7.98 (later beaten by MolDQN).

## ChemBERTa (Chithrananda et al., 2020)

**Citation [9]:** Chithrananda, S., Grand, G., & Ramsundar, B. (2020). "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." arXiv:2010.09885.

**Architecture.** RoBERTa transformer (12 layers, 768 hidden, 12 heads) pretrained with masked-language-modeling on SMILES tokenized via byte-pair encoding. Released curated 77M-SMILES PubChem corpus alongside the model.

**Downstream tasks.** Fine-tuned on MoleculeNet classification benchmarks: BBBP, ClinTox, Tox21, HIV. Competitive with — though not strictly beating — graph-based baselines (D-MPNN, GROVER) at 10M-77M pretraining scale, with positive scaling trends as pretraining corpus grows. The paper's framing was less "we beat graphs" and more "transformers on SMILES are now a credible representation learner" — and it demonstrated attention-map interpretability for chemistry.

**Significance for our paper.** ChemBERTa is the proof-of-concept that *language models read SMILES well enough to do chemistry tasks*. It is the upstream evidence that an LLM-as-policy approach is even plausible.

## MolGPT (Bagal et al., 2021)

**Citation [10]:** Bagal, V., Aggarwal, R., Vinod, P. K., & Priyakumar, U. D. (2021). "MolGPT: Molecular Generation Using a Transformer-Decoder Model." *Journal of Chemical Information and Modeling*, 62(9), 2064–2076.

**Architecture.** GPT-style decoder-only transformer (8 layers, 8 heads, 256 embed dim) trained on next-token prediction over ZINC SMILES. Conditioning on properties (logP, SAS, TPSA, QED) is supported by simple concatenation of property tokens at the prompt prefix.

**Results.** Validity / uniqueness / novelty on par with prior SOTA (REINVENT, JT-VAE, GraphAF) on MOSES and GuacaMol distribution-matching benchmarks. MolGPT's contribution was demonstrating that *decoder-only transformers* — the GPT recipe — work for SMILES with simple conditioning, without elaborate scaffolding or RL.

## MolFormer (Ross et al., 2022)

**Citation [11]:** Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., & Das, P. (2022). "Large-scale chemical language representations capture molecular structure and properties." *Nature Machine Intelligence*, 4(12), 1256–1264. arXiv:2106.09553.

**Architecture.** Linear-attention transformer with rotary positional embeddings (RoPE), pretrained MLM-style on **1.1 billion** SMILES from PubChem + ZINC — by far the largest molecular pretraining at the time. Linear attention enables this scale on a single multi-GPU node.

**Results.** Outperforms supervised and self-supervised graph baselines (D-MPNN, GROVER, several GNN architectures) on 10 MoleculeNet benchmarks including QM9 quantum-chemical regression. Attention analysis recovers spatial / chemical relationships from SMILES alone — supporting the hypothesis that linguistic representations of molecules learn implicit graph structure.

## What Goes Wrong: Honest Limitations of the Pre-LLM Stack

### Mode collapse and diversity loss

**Citation [12]:** Blaschke, T., Engkvist, O., Bajorath, J., & Chen, H. (2020). "Memory-assisted reinforcement learning for diverse molecular de novo design." *Journal of Cheminformatics*, 12:68.

REINVENT-style policy gradient with a KL anchor still suffers policy collapse: once the Agent finds a high-scoring scaffold, it samples that mode repeatedly. Blaschke et al. introduce a memory unit (essentially a discount applied to molecules close to recently-emitted ones) to recover diversity. This problem is not unique to chemistry — Wenkel et al. (arXiv:2510.20817) showed in 2025 that *KL-regularized RL is mathematically designed to mode-collapse* under generic targets, regardless of optimizer choice.

### Reward hacking and metric brittleness

**Citation [13]:** Renz, P., Van Rompaey, D., Wegner, J. K., Hochreiter, S., & Klambauer, G. (2020). "On failure modes in molecule generation and optimization." *Drug Discovery Today: Technologies*, 32–33, 55–63.

Renz et al. demonstrate two distinct failure modes: (a) the **"AddCarbon" trivial baseline** (insert a random carbon atom into a random training molecule) achieves near-perfect novelty + uniqueness + KL-divergence scores on GuacaMol distribution-learning benchmarks, exposing GuacaMol's metrics as gameable; (b) **scoring-function bias** — when generative models are trained to maximize a ligand-based QSAR scoring function, the optimization-model score and an independently-trained control-model score diverge during training. The Agent learns features specific to the scoring model rather than features generalizing across QSAR replicates. This is a textbook overfitting-to-the-reward-model failure.

### Unconstrained logP is a degenerate objective

**Citation [14]:** Brown, N., Fiscato, M., Segler, M. H., & Vaucher, A. C. (2019). "GuacaMol: Benchmarking models for de novo molecular design." *Journal of Chemical Information and Modeling*, 59(3), 1096–1108.

GuacaMol's authors themselves warn that QED has a global ceiling of 0.948 reachable by random sampling, and that simple goal-directed benchmarks "are not suitable for the assessment of generative models" because the unmodified ChEMBL distribution achieves them already. Penalized logP (logP - SA - cycle penalty) is similarly attacked: it has no upper bound and rewards long alkane chains, so MolDQN's 11.84 win is a chain-length artifact, not a chemistry result. The graph genetic algorithm (a non-ML baseline) topped the original GuacaMol leaderboard, generating zero synthetically accessible molecules in follow-up tests — a stark statement that the leaderboard does not measure utility.

### Sample efficiency: simple beats complex

**Citation [15]:** Gao, W., Fu, T., Sun, J., & Coley, C. W. (2022). "Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization." *NeurIPS 2022 Datasets & Benchmarks*. arXiv:2206.12411.

PMO benchmark: 25 algorithms x 23 tasks under a strict 10K-oracle-call budget. Headline finding: most "SOTA" methods *fail to outperform their predecessors* under realistic budgets. Vanilla REINVENT (the 2017 method) ranks as the most sample-efficient generative model on average. Graph-based genetic algorithms are competitive with sophisticated RL. MolDQN, GCPN, JT-VAE all underperform REINVENT in sample efficiency. The plateau is real: 8 years of architectural innovation has not consistently beaten a 3-layer GRU with augmented likelihood under realistic oracle budgets. Augmented Memory (Guo & Schwaller, 2024, JACS Au) extends REINVENT with experience replay and beats it on 19/23 PMO tasks — but the reference frame remains the 2017 GRU.

### Generalization across targets

Renz et al. and follow-ups (Coley group papers from 2022–2024) consistently show that fine-tuning a generative model for one target does not transfer: reward-hacking the QSAR model for target A produces molecules that don't look active under an independently-trained model for the same target. SMILES-RL approaches plateau because their priors are narrow (ZINC/ChEMBL drug-like distribution) and their reward channels are narrow (a single scoring function the Agent can overfit).

## What's Different About LLM-as-Policy

### What the LLM-policy paradigm offers

**Citation [16]:** Anstine, D. M., & Isayev, O. (2024). "Large Language Models as Molecular Design Engines." *Journal of Chemical Information and Modeling*, 64(20), 7747–7757. arXiv preprint at chemRxiv 10.26434/chemrxiv-2024-n0l8q.

**Citation [17]:** Guevorguian, P., et al. (2024). "Small Molecule Optimization with Large Language Models." arXiv:2407.18897.

The LLM-as-policy paradigm — exemplified by Anstine & Isayev's demonstration that Claude 3 Opus generates 97%-valid molecules under natural-language prompting, and Guevorguian et al.'s Chemlactica/Chemma showing 8% improvement on PMO — offers four things prior molecular RL does not:

1. **A vastly broader prior**. Where REINVENT's Prior is 1.5M ChEMBL molecules and MolFormer's is 1.1B SMILES, an LLM has been pretrained on ~all of the chemistry literature: papers, reaction notes, medicinal-chemistry textbooks, retrosynthesis lore. The Prior contains *reasoning* about chemistry, not just structural distributions.

2. **Natural-language reward channels**. A REINVENT Agent maximizes a scalar QSAR output. An LLM Agent can ingest a multi-paragraph design rationale ("avoid CYP3A4 liability, prefer kinase hinge binders, retain the H-bond donor at position 6") and respond at the right level of abstraction. This dissolves the brittle scalar-reward bottleneck Renz et al. identified.

3. **In-context learning instead of fine-tuning**. Anstine & Isayev show iterative editing under prompt control — no gradient steps. This sidesteps mode collapse from KL-regularized RL: the policy is the LLM, and its "exploration" is governed by sampling temperature and prompt diversity rather than entropy regularization.

4. **Free composition with tools**. The LLM-as-policy can call RDKit, dock structures, query PubChem, and read back results — the agent paradigm is native, where REINVENT-family scoring components had to be hand-wired into a Python aggregator.

### What goes wrong (the new failure modes)

1. **Parse-rate failures**. LLMs sometimes emit syntactically invalid SMILES — wrong ring-closure digits, unbalanced parentheses, illegal valences. Anstine & Isayev report 97% validity *with prompt engineering*; raw zero-shot validity is lower. By contrast, a graph-builder like MolDQN is 100% valid by construction. The LLM-policy buys flexibility at the cost of a hard validity floor.

2. **Prompt sensitivity / non-robust elicitation**. The same chemistry task phrased two slightly different ways produces materially different molecule distributions. There is no equivalent of REINVENT's deterministic Prior; output depends on prompt phrasing, temperature, system messages, and even whitespace choices. This makes reproducible benchmarking difficult and suggests the policy is doing pattern-completion on *prompt style* as much as on chemistry.

3. **Inverted scaling on chemistry primitives**. Yu et al. (arXiv:2505.16340) and others show LLMs fail at trivial SMILES-parsing tasks (counting rings, identifying functional groups) even when they ace molecular reasoning at higher levels. There is no guarantee of monotonic capability with model size — bigger LLMs can be *worse* at low-level chemistry primitives if training data emphasized natural-language descriptions over raw SMILES manipulation.

4. **Convergent/divergent creativity trade-off**. Bhatt et al. (arXiv:2604.18031) show LLMs exhibit a systematic negative correlation between validity/success-rate and novelty/diversity in molecular generation. Prompting harder for "valid drug-like molecules" reduces structural diversity; prompting for novelty erodes validity. This is a new mode-collapse failure mode — different in kind from REINVENT's KL-driven collapse, because it's driven by the LLM's own training distribution rather than the optimizer.

5. **Reward-hacking still happens**. If the reward signal is an automated docking or QSAR call, the LLM will exploit it just like REINVENT did — with the additional risk that the LLM can rationalize hacked outputs in fluent natural language, masking the failure from a reviewing chemist.

## Positioning Statement

PharmaRL — our OpenEnv-native environment for LLM-as-policy iterative molecular editing under composite chemistry rewards — sits at the intersection of three lines: (a) the REINVENT family's KL-anchored RL on chemical-language priors [1,2,3], (b) the multi-objective MCMC/RL approaches that explicitly target the multi-target reward landscape (MARS [6], Augmented Memory), and (c) the emerging LLM-as-policy work (Anstine & Isayev [16], Guevorguian et al. [17]). What PharmaRL contributes that the prior lineage does not: a *standardized environment interface* (OpenEnv) that lets any LLM act as policy without bespoke fine-tuning code; a composite reward stack (validity x QED x SA x docking) calibrated against the failure modes Renz et al. [13] documented; and explicit instrumentation for the new failure modes (parse rate, prompt sensitivity, convergent/divergent diversity collapse) that the LLM-as-policy paradigm introduces. The thesis: prior molecular RL plateaued because architecture innovation outpaced reward and prior quality (PMO [15]); LLM-as-policy unlocks a richer prior and a richer reward channel, but only inside an environment that surfaces and scores its native failure modes.

---

## BibTeX Entries

```bibtex
@article{olivecrona2017reinvent,
  title={Molecular de-novo design through deep reinforcement learning},
  author={Olivecrona, Marcus and Blaschke, Thomas and Engkvist, Ola and Chen, Hongming},
  journal={Journal of Cheminformatics},
  volume={9},
  number={1},
  pages={48},
  year={2017},
  doi={10.1186/s13321-017-0235-x},
  archivePrefix={arXiv},
  eprint={1704.07555}
}

@article{blaschke2020reinvent2,
  title={{REINVENT} 2.0: An {AI} Tool for De Novo Drug Design},
  author={Blaschke, Thomas and Ar{\'u}s-Pous, Josep and Chen, Hongming and Margreitter, Christian and Tyrchan, Christian and Engkvist, Ola and Papadopoulos, Kostas and Patronov, Atanas},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  number={12},
  pages={5918--5922},
  year={2020},
  doi={10.1021/acs.jcim.0c00915}
}

@article{loeffler2024reinvent4,
  title={{REINVENT} 4: Modern {AI}-driven generative molecule design},
  author={Loeffler, Hannes H. and He, Jiazhen and Tibo, Alessandro and Janet, Jon Paul and Voronov, Alexey and Mervin, Lewis H. and Engkvist, Ola},
  journal={Journal of Cheminformatics},
  volume={16},
  number={1},
  pages={20},
  year={2024},
  doi={10.1186/s13321-024-00812-5}
}

@article{zhou2019moldqn,
  title={Optimization of Molecules via Deep Reinforcement Learning},
  author={Zhou, Zhenpeng and Kearnes, Steven and Li, Li and Zare, Richard N. and Riley, Patrick},
  journal={Scientific Reports},
  volume={9},
  number={1},
  pages={10752},
  year={2019},
  doi={10.1038/s41598-019-47148-x},
  archivePrefix={arXiv},
  eprint={1810.08678}
}

@inproceedings{shi2020graphaf,
  title={{GraphAF}: a Flow-based Autoregressive Model for Molecular Graph Generation},
  author={Shi, Chence and Xu, Minkai and Zhu, Zhaocheng and Zhang, Weinan and Zhang, Ming and Tang, Jian},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
  archivePrefix={arXiv},
  eprint={2001.09382}
}

@inproceedings{xie2021mars,
  title={{MARS}: Markov Molecular Sampling for Multi-objective Drug Discovery},
  author={Xie, Yutong and Shi, Chence and Zhou, Hao and Yang, Yuwei and Zhang, Weinan and Yu, Yong and Li, Lei},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021},
  archivePrefix={arXiv},
  eprint={2103.10432}
}

@inproceedings{jin2018jtvae,
  title={Junction Tree Variational Autoencoder for Molecular Graph Generation},
  author={Jin, Wengong and Barzilay, Regina and Jaakkola, Tommi},
  booktitle={Proceedings of the 35th International Conference on Machine Learning (ICML)},
  pages={2323--2332},
  year={2018},
  archivePrefix={arXiv},
  eprint={1802.04364}
}

@inproceedings{you2018gcpn,
  title={Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation},
  author={You, Jiaxuan and Liu, Bowen and Ying, Rex and Pande, Vijay and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={31},
  year={2018},
  archivePrefix={arXiv},
  eprint={1806.02473}
}

@article{chithrananda2020chemberta,
  title={{ChemBERTa}: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction},
  author={Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
  journal={arXiv preprint},
  year={2020},
  archivePrefix={arXiv},
  eprint={2010.09885}
}

@article{bagal2021molgpt,
  title={{MolGPT}: Molecular Generation Using a Transformer-Decoder Model},
  author={Bagal, Viraj and Aggarwal, Rishal and Vinod, P. K. and Priyakumar, U. Deva},
  journal={Journal of Chemical Information and Modeling},
  volume={62},
  number={9},
  pages={2064--2076},
  year={2021},
  doi={10.1021/acs.jcim.1c00600}
}

@article{ross2022molformer,
  title={Large-scale chemical language representations capture molecular structure and properties},
  author={Ross, Jerret and Belgodere, Brian and Chenthamarakshan, Vijil and Padhi, Inkit and Mroueh, Youssef and Das, Payel},
  journal={Nature Machine Intelligence},
  volume={4},
  number={12},
  pages={1256--1264},
  year={2022},
  doi={10.1038/s42256-022-00580-7},
  archivePrefix={arXiv},
  eprint={2106.09553}
}

@article{blaschke2020memory,
  title={Memory-assisted reinforcement learning for diverse molecular de novo design},
  author={Blaschke, Thomas and Engkvist, Ola and Bajorath, J{\"u}rgen and Chen, Hongming},
  journal={Journal of Cheminformatics},
  volume={12},
  number={1},
  pages={68},
  year={2020},
  doi={10.1186/s13321-020-00473-0}
}

@article{renz2020failure,
  title={On failure modes in molecule generation and optimization},
  author={Renz, Philipp and Van Rompaey, Dries and Wegner, J{\"o}rg K. and Hochreiter, Sepp and Klambauer, G{\"u}nter},
  journal={Drug Discovery Today: Technologies},
  volume={32-33},
  pages={55--63},
  year={2020},
  doi={10.1016/j.ddtec.2020.09.003}
}

@article{brown2019guacamol,
  title={{GuacaMol}: Benchmarking models for de novo molecular design},
  author={Brown, Nathan and Fiscato, Marco and Segler, Marwin H.S. and Vaucher, Alain C.},
  journal={Journal of Chemical Information and Modeling},
  volume={59},
  number={3},
  pages={1096--1108},
  year={2019},
  doi={10.1021/acs.jcim.8b00839}
}

@inproceedings{gao2022pmo,
  title={Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization},
  author={Gao, Wenhao and Fu, Tianfan and Sun, Jimeng and Coley, Connor W.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2022},
  archivePrefix={arXiv},
  eprint={2206.12411}
}

@article{anstine2024llmmoldesign,
  title={Large Language Models as Molecular Design Engines},
  author={Anstine, Dylan M. and Isayev, Olexandr},
  journal={Journal of Chemical Information and Modeling},
  volume={64},
  number={20},
  pages={7747--7757},
  year={2024},
  doi={10.1021/acs.jcim.4c01396}
}

@article{guevorguian2024smallmolllm,
  title={Small Molecule Optimization with Large Language Models},
  author={Guevorguian, Philipp and Bedrosian, Menua and Fahradyan, Tigran and Avetisyan, Hayk and Khachatrian, Hrant and Aghajanyan, Armen},
  journal={arXiv preprint},
  year={2024},
  archivePrefix={arXiv},
  eprint={2407.18897}
}
```
