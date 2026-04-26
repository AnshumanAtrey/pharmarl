# Section 04: GRPO, Benchmarks, and Infrastructure — Technical Citations

Research compiled for *AI Alchemy In Medicine: A Vision for LLM-as-Policy Molecular Design*
(Meta PyTorch OpenEnv Hackathon Round 2 Grand Finale — April 26, 2026)

This document supplies primary citations, exact mathematical formulations, and methodology
recipes for the methods/infrastructure section of the paper. Equations are written in
LaTeX-friendly inline form so they can be pasted directly into the paper source.

---

## 1. GRPO — Group Relative Policy Optimization

**Primary citation [1]:** Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H.,
Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). *DeepSeekMath: Pushing the Limits of
Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.

GRPO was introduced in DeepSeekMath as a memory-efficient variant of PPO that removes the
value/critic network. Instead of learning a state-value baseline, GRPO samples a *group* of
$G$ candidate completions for each prompt and uses the empirical mean and standard deviation
of group rewards as a per-prompt baseline. This reduces the trainable footprint by roughly a
factor of two (no critic of policy size) and removes a notoriously brittle component of
RLHF-style PPO.

### 1.1 The GRPO Objective

For each question $q \sim P(Q)$, the old policy $\pi_{\theta_{\text{old}}}$ samples a group
$\{o_i\}_{i=1}^{G}$ of $G$ outputs. The clipped surrogate objective (Eq. 21 in the paper) is:

$$
\mathcal{J}_{\text{GRPO}}(\theta) =
\mathbb{E}_{q,\{o_i\}}
\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\Bigl\{
\min\!\bigl(
\rho_{i,t}(\theta)\hat{A}_{i,t},\;
\text{clip}(\rho_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon)\hat{A}_{i,t}
\bigr)
- \beta\, \mathbb{D}_{\text{KL}}\!\left[\pi_\theta\,\Vert\,\pi_{\text{ref}}\right]
\Bigr\}
\right]
$$

where the per-token importance ratio is
$\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$.

### 1.2 Group-Standardized Advantage (Outcome Supervision)

Under outcome supervision (a single scalar reward $r_i$ for each completion $o_i$, as in
PharmaRL where the reward is a pharmacological oracle score), every token in $o_i$ receives
the same advantage:

$$
\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}
$$

Group-relative standardization is the heart of GRPO: it converts noisy absolute oracle scores
into a *relative ranking signal* with zero mean and unit variance per prompt. This automatically
adapts to differently-scaled rewards across prompts and removes the need for reward-model
normalization.

### 1.3 K3 KL Regularization

GRPO applies a *direct* KL penalty in the objective (rather than mixing it into the reward as
classical PPO-RLHF does). The estimator is the unbiased, always-non-negative K3 estimator
(see Section 2.2):

$$
\mathbb{D}_{\text{KL}}\!\left[\pi_\theta\,\Vert\,\pi_{\text{ref}}\right]
= \frac{\pi_{\text{ref}}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
- \log\frac{\pi_{\text{ref}}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
- 1
$$

This token-level penalty anchors the policy to the SFT/instruct reference $\pi_{\text{ref}}$ and
prevents reward-hacking collapse — particularly important for molecular generation where
narrow oracle exploits (degenerate SMILES patterns) exist.

### 1.4 Why Drop the Value Model?

PPO trains a critic of comparable size to the policy to estimate $V^\pi(s_t)$ as a baseline.
For LLMs, this *doubles* GPU memory and introduces a chicken-and-egg learning problem (the
critic must track the moving policy). GRPO replaces $V^\pi$ with the per-prompt empirical mean
$\bar{r} = \frac{1}{G}\sum_j r_j$, which is unbiased *for that prompt* and free of learned
parameters. The trade-off: $G\!-\!1$ extra samples per gradient step for one critic-pass saved.
For a 3B model with $G=8$, this is a clear win.

### 1.5 Successor Work

**[2] DeepSeek-R1** (DeepSeek-AI et al., 2025; arXiv:2501.12948) used GRPO to elicit
chain-of-thought reasoning from DeepSeek-V3-Base with *zero* SFT data (DeepSeek-R1-Zero) and
matched OpenAI-o1 on AIME 2024 (pass@1 71.0% with majority voting at 86.7%). The reward signal
was solely correctness on math/code, with no process supervision. This validates GRPO as a
production training algorithm at scale.

**[3] Chemistry / Molecular GRPO applications (2025-2026):** SMILES-GRPO (in ChemCRAFT, 2026)
applies GRPO with dense chemical rewards for tool-augmented molecular design. Reference-guided
Policy Optimization (RePO, arXiv:2603.05900, 2026) reports GRPO baselines on molecular
optimization tasks. PharmaRL is the first OpenEnv-native realization of GRPO for direct LLM
oracle-driven molecular design.

---

## 2. PPO and the K3 KL Estimator

### 2.1 PPO

**[4] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** *Proximal
Policy Optimization Algorithms.* arXiv:1707.06347.

PPO's clipped surrogate is the parent of GRPO's objective:

$$
\mathcal{L}^{\text{CLIP}}(\theta) =
\mathbb{E}_t\!\left[
\min\!\bigl(
\rho_t(\theta)\hat{A}_t,\;
\text{clip}(\rho_t(\theta),\,1-\varepsilon,\,1+\varepsilon)\hat{A}_t
\bigr)
\right]
$$

with $\rho_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$ and
typically $\varepsilon = 0.2$. The clip prevents destructive policy updates without TRPO's
expensive trust-region machinery. PPO supports multiple epochs of minibatch SGD per rollout —
a sample-efficiency property GRPO inherits.

### 2.2 K3 KL Estimator

**[5] Schulman, J. (2020).** *Approximating KL Divergence.*
http://joschu.net/blog/kl-approx.html

Three Monte-Carlo estimators of $\mathrm{KL}(p\,\Vert\,q)$ for samples $x \sim p$,
with $r = q(x)/p(x)$:

| Estimator | Form | Bias | Variance | Sign |
|-----------|------|------|----------|------|
| $k_1$ | $-\log r$ | unbiased | high | can be negative |
| $k_2$ | $\tfrac{1}{2}(\log r)^2$ | biased | low | non-negative |
| $k_3$ | $(r - 1) - \log r$ | **unbiased** | **low** | **always $\geq 0$** |

The general unbiased family is $-\log r + \lambda(r-1)$ for any $\lambda \in \mathbb{R}$
(since $\mathbb{E}_{x\sim p}[r-1] = 0$). Choosing $\lambda = 1$ recovers $k_3$ and exploits the
inequality $r - 1 \geq \log r$ to guarantee non-negativity. Geometrically, $k_3$ is the
Bregman divergence of $-\log$ at $r=1$ — second-order around the on-policy regime, so variance
is small whenever $\pi_\theta \approx \pi_{\text{ref}}$, which is exactly the regime where the
KL constraint is enforced.

In our paper, we use $k_3$ on a per-token basis as in DeepSeekMath Eq. 4.

---

## 3. Therapeutics Data Commons (TDC)

**[6] Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley, C. W., Xiao, C.,
Sun, J., & Zitnik, M. (2021).** *Therapeutics Data Commons: Machine Learning Datasets and
Tasks for Drug Discovery and Development.* NeurIPS Datasets and Benchmarks Track.
arXiv:2102.09548. https://tdcommons.ai

TDC packages 17 molecule-generation oracles and >200 datasets behind a uniform Python API.
PharmaRL uses four oracles drawn from two TDC families (`Oracle` for activity classifiers,
`single_pred` ADMET for CYP):

| Oracle | Type | Underlying Model | Source Dataset |
|--------|------|------------------|----------------|
| **DRD2** | Bioactivity classifier | SVM (Gaussian kernel) on ECFP6 | ExCAPE-DB |
| **GSK3B** | Bioactivity classifier | Random Forest on ECFP6 | ExCAPE-DB |
| **JNK3** | Bioactivity classifier | Random Forest on ECFP6 | ExCAPE-DB |
| **CYP3A4_Veith** | ADMET — CYP3A4 inhibition | Provided as binary label set; ML head varies (Morgan+MLP, AttentiveFP) | Veith et al., *Nat. Biotechnol.* 27(11), 2009 |

The DRD2 SVM+ECFP6 oracle was constructed by **Olivecrona et al. (2017)** [7] for REINVENT and
adopted by TDC as a community benchmark (Olivecrona, M., Blaschke, T., Engkvist, O., Chen, H.
*Molecular De Novo Design through Deep Reinforcement Learning.* J. Cheminform. 9:48, 2017.
DOI:10.1186/s13321-017-0235-x). PharmaRL inherits this exact predictor for direct
reproducibility against REINVENT-era baselines.

**Splits.** TDC supports random, scaffold (Bemis–Murcko, Bemis & Murcko, *J. Med. Chem.*
39:2887–2893, 1996), and cold-target splits. Bemis–Murcko scaffolds reduce a molecule to its
ring system + linker atoms by iteratively stripping degree-1 atoms; the scaffold split groups
molecules sharing a scaffold into the same partition. Recent work (arXiv:2406.00873) shows
scaffold splits *overestimate* virtual-screening performance vs. UMAP-based splits, but they
remain the canonical convention for cross-paper comparison.

---

## 4. GuacaMol Benchmark

**[8] Brown, N., Fiscato, M., Segler, M. H. S., & Vaucher, A. C. (2019).** *GuacaMol:
Benchmarking Models for de Novo Molecular Design.* *J. Chem. Inf. Model.* 59(3), 1096–1108.
DOI:10.1021/acs.jcim.8b00839. https://github.com/BenevolentAI/guacamol

GuacaMol defines two benchmark families on a curated ChEMBL-derived training set:

- **5 distribution-learning benchmarks**: validity, uniqueness, novelty, KL divergence on
  physicochemical descriptors, Fréchet ChemNet Distance (FCD).
- **20 goal-directed benchmarks**:
  - *Rediscovery (3):* recover Celecoxib, Troglitazone, Thiothixene by Tanimoto-on-ECFP4
  - *Similarity (3):* generate molecules ECFP4-similar (>0.75) to Aripiprazole, Albuterol, Mestranol
  - *Isomers (2):* match formulae C₁₁H₂₄ and C₉H₁₀N₂O₂PF₂Cl
  - *Median molecules (2):* simultaneous similarity to (Camphor, Menthol) and (Tadalafil, Sildenafil)
  - *Multi-property optimization (7):* Osimertinib, Fexofenadine, Ranolazine, Perindopril,
    Amlodipine, Sitagliptin, Zaleplon
  - *SMARTS-based (1):* Valsartan SMARTS substructure with physicochemical constraints
  - *Scaffold hop (1) + decoration (1):* additional structural tasks

Goal-directed scoring is a weighted sum where each task returns a score in $[0,1]$ derived
from oracles that mix Tanimoto similarity, isomer fitness, QED, SA-score, and molecular
property targets. GuacaMol tests *exploitation* — directed optimization toward known molecules
or property targets. The SMILES LSTM and Graph GA baselines are the standard reference points;
Graph GA is widely treated as the high bar.

---

## 5. MOSES Benchmark

**[9] Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov, O.,
Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin, A., Johansson,
S., Chen, H., Nikolenko, S., Aspuru-Guzik, A., & Zhavoronkov, A. (2020).** *Molecular Sets
(MOSES): A Benchmarking Platform for Molecular Generation Models.* *Front. Pharmacol.*
11:565644. DOI:10.3389/fphar.2020.565644. arXiv:1811.12823.

MOSES complements GuacaMol on the *exploration* / distribution-learning side. Metrics:

- **Validity:** fraction RDKit-parseable
- **Uniqueness@k:** fraction of unique SMILES among first $k$ generated (typically $k=1000$ or $10000$)
- **Novelty:** fraction not in training set
- **FCD:** $\text{FCD} = \|\mu_G - \mu_R\|_2^2 + \mathrm{Tr}\!\left(\Sigma_G + \Sigma_R - 2(\Sigma_G \Sigma_R)^{1/2}\right)$
  on ChemNet penultimate features (lower is better)
- **SNN:** average Tanimoto (Morgan r=2, 1024 bits) of each generated molecule to its nearest
  reference neighbor
- **IntDiv:** $1 - \mathbb{E}_{m_1, m_2 \in G}[T(m_1, m_2)]$ — internal diversity, higher is better
- **Filters:** fraction passing PAINS/MCF/lead-likeness filters used during dataset construction
- **Scaff:** cosine similarity of Bemis–Murcko scaffold frequency histograms

Together with GuacaMol, MOSES gives a two-axis evaluation: (a) *can the model
generate diverse, valid, novel molecules?* (MOSES) and (b) *can the model directly optimize
toward a target?* (GuacaMol goal-directed). PharmaRL reports both.

---

## 6. SELFIES — Robust Molecular Strings

**[10] Krenn, M., Häse, F., Nigam, A., Friederich, P., & Aspuru-Guzik, A. (2020).*
*Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation.*
*Mach. Learn. Sci. Technol.* 1(4), 045024. DOI:10.1088/2632-2153/aba947. arXiv:1905.13741.

SELFIES is a context-free formal grammar over molecular tokens. Each symbol's interpretation
depends on a state variable tracking remaining valence, and branch lengths / ring sizes are
encoded *together with their identifiers*. Consequence: **every SELFIES string decodes to a
valid molecule**, and every valid molecule has a SELFIES encoding. Krenn et al. validated this
by round-tripping all 72 million PubChem molecules.

For LLM-as-policy molecular design, SELFIES has two killer properties:
1. **Reward shaping is well-defined** — sampling never produces invalid graphs that crash
   oracles or return undefined rewards, so RL gradients are not polluted by syntax failures.
2. **Action space matches policy outputs** — every token the policy emits is a legal action,
   so KL regularization over $\pi_\theta$ vs. $\pi_{\text{ref}}$ is meaningful at the token
   level.

By contrast, SMILES sampling routinely produces 10–30% invalid strings unless the model has
been heavily trained, and rejection sampling biases the gradient.

**[11] Krenn et al. (2022).** *SELFIES and the future of molecular string representations.*
*Patterns* 3(10), 100588. arXiv:2204.00056. — Outlines 16 future directions including
fragment-aware variants and broader chemistry coverage.

**[12] Cheng, A. H. et al. (2023).** *Group SELFIES: a robust fragment-based molecular string
representation.* *Digital Discovery*, 2, 748–758. DOI:10.1039/D3DD00012E. — Group SELFIES
encodes functional-group fragments as single tokens, blending SELFIES validity with
fragment-based interpretability. Useful for synthesis-aware reward signals (future work).

The current `selfies` Python package (v2.1.x as of 2024) is the reference implementation
([13]; Lo, A., Pollice, R., Nigam, A. K., White, A. D., Krenn, M., Aspuru-Guzik, A., 2023.
*Recent advances in the self-referencing embedded strings (SELFIES) library.* *Digital
Discovery*, 2, 897-908.).

---

## 7. OpenEnv (Meta PyTorch)

**[14] Meta-PyTorch & Hugging Face (2025).** *OpenEnv: Agentic RL Execution Framework.*
Released October 2025 at PyTorch Conference 2025 (San Francisco). Initial spec: OpenEnv 0.1.
Current: v0.2.3 (March 2026).
Repo: https://github.com/meta-pytorch/OpenEnv. Docs: http://meta-pytorch.org/OpenEnv/. Hub:
https://huggingface.co/spaces (OpenEnv-tagged spaces).

**[15] Meta AI Blog (2025).** *The Building Blocks of Agentic AI: From Kernels to Clusters.*
https://ai.meta.com/blog/introducing-pytorch-native-agentic-stack/

OpenEnv is a *PyTorch-native* standardized environment specification for RL post-training of
agentic LLMs. Design principles (RFCs 001–005):

- **Gymnasium-style API:** `reset()`, `step(action)`, `state()`, `close()` — directly familiar
  to RL researchers; agnostic to the training framework.
- **HTTP/REST + Docker isolation:** environments ship as containerized FastAPI servers.
  Strong isolation boundaries are essential for code-execution / tool-use environments and
  for multi-tenant cloud training.
- **Pydantic typed schemas:** every env defines `Action`, `Observation`, `State` dataclasses.
  Server- and client-side validation prevents schema drift between trainer and env. The
  base `Observation` carries `done`, `reward`, and a `metadata: Dict` for env-specific
  telemetry.
- **Session-keyed state:** episodes are tracked server-side via session IDs; the client is
  stateless. This enables pluggable backends (in-memory, Redis, etc.) and parallel rollout.
- **Async-first client:** `HTTPEnvClient` exposes `async` methods with a `.sync()` wrapper.
  Critical for pipelining $G$ parallel rollouts for GRPO group sampling.
- **HF Hub integration:** `openenv push` packages and uploads the Docker image to a
  HuggingFace Space. Any user can `pip install git+https://huggingface.co/spaces/...` to
  pull the env client.
- **Reward stays in the env:** "Reward logic stays with the environment where domain
  knowledge resides" (RFC 002). Avoids client/env drift on reward semantics.

Initial Hub envs: `coding_env`, `atari_env`, `OpenSpiel_env`, `echo_env`. PharmaRL targets the
TDC-oracle gap — there is no published OpenEnv-native chemistry environment as of v0.2.3.

For the paper, this is our primary infrastructure citation: PharmaRL is built directly on the
RFC 002 spec and pushed to HF Hub.

---

## 8. Unsloth — Fast LoRA / QLoRA Fine-Tuning

**[16] Han, D., & Han, M. (2024).** *Unsloth: 2x faster LoRA fine-tuning for LLMs.*
https://unsloth.ai. https://github.com/unslothai/unsloth.

Unsloth is a PyTorch-level kernel library for parameter-efficient fine-tuning. Key engineering:

- **Hand-written Triton kernels** for the LoRA forward/backward path, fused
  matmul–activation–LoRA paths, RoPE, and attention. Reported 2× wall-clock speedup vs.
  HuggingFace `peft` + `transformers` baseline.
- **Dynamic 4-bit quantization** (Unsloth blog, 2024): builds on `bitsandbytes` 4-bit
  (NF4 from QLoRA) but selectively retains higher precision for layers that quantize poorly,
  using ~10% more VRAM than vanilla BnB-4bit while recovering most of the bf16 accuracy gap.
- **4× memory reduction** vs. standard LoRA at zero accuracy loss for typical instruct fine
  tunes; trains a 7B model on a single 24 GB GPU. For Llama-3.2-3B-Instruct + LoRA + GRPO,
  Unsloth fits comfortably on a single A100-40 GB or even an L4-24 GB.

Unsloth integrates with `trl` (HuggingFace's `GRPOTrainer`), so the GRPO loop in our paper
uses Unsloth's `FastLanguageModel` as the policy and reference, with `torch.compile`-friendly
kernels.

---

## 9. LoRA and QLoRA

**[17] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.
(2021).** *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. ICLR 2022.

For a frozen pretrained weight $W_0 \in \mathbb{R}^{d \times k}$, LoRA represents the update as
the product of two low-rank matrices:

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}\, B A,
\quad B \in \mathbb{R}^{d\times r},\;
A \in \mathbb{R}^{r\times k},\;
r \ll \min(d, k)
$$

with $A$ initialized $\mathcal{N}(0, \sigma^2)$ and $B$ initialized to zero (so $\Delta W = 0$
at start). $\alpha$ is a scaling hyperparameter; the `peft` convention is to set
$\text{lora\_alpha} = 2r$.

In the original paper, only $W_q$ and $W_v$ in the attention block are adapted. For Llama
generations and modern recipes, common practice is to adapt
$\{W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}\}$ — i.e., all linear
projections in attention and the SwiGLU MLP. We use this "all-linear" target-modules setting.
Typical hyperparameters for our 3B target: $r=16$, $\alpha=32$, dropout $0.0$, target_modules
all-linear. Trainable parameter count: ~24M (≈0.7% of base).

LoRA's memory wins are dramatic: ~10,000× fewer trainable params and ~3× lower GPU memory at
GPT-3 scale, with on-par or better task performance vs. full FT. No inference latency penalty
once LoRA weights are merged via $W \leftarrow W_0 + \tfrac{\alpha}{r} BA$.

**[18] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).** *QLoRA: Efficient
Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.

QLoRA backpropagates through a **4-bit quantized frozen base model** into bf16 LoRA adapters.
Three innovations:

1. **NF4 (4-bit NormalFloat):** information-theoretically optimal 4-bit dtype for normally
   distributed weights — non-uniform quantization buckets matched to a standard normal.
2. **Double quantization:** quantize the per-block quantization constants themselves, reducing
   the storage overhead from ~0.5 bit/param to ~0.13 bit/param.
3. **Paged optimizers:** NVIDIA unified-memory paging to handle gradient/optimizer-state
   spikes during long sequences without OOM.

QLoRA fine-tuned Guanaco-65B on a single 48 GB GPU and reached 99.3% of ChatGPT on Vicuna
benchmarks. For PharmaRL we use QLoRA-style 4-bit base + LoRA via Unsloth on Llama-3.2-3B.

---

## 10. Llama 3.2 3B Instruct — Base Model

**[19] Meta AI (Sept 25, 2024).** *Llama 3.2: Revolutionizing Edge AI and Vision with Open,
Customizable Models.* https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/.
Model card: meta-llama/Llama-3.2-3B-Instruct on HuggingFace.

Auto-regressive transformer; 3.21B parameters; 128K context; pre-trained on up to 9T tokens;
distilled from Llama-3.1-8B / 70B teacher logits; SFT + RLHF post-training. Officially supports
8 languages. Suitable as a small-footprint policy that fits comfortably with QLoRA + GRPO group
sampling on a single GPU. The 3B size is the sweet spot for hackathon-scale RL: large enough to
have non-trivial chemistry priors from pre-training, small enough that group sampling at $G=8$
is tractable.

---

## BibTeX Entries

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y K and Wu, Y and Guo, Daya},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}

@article{deepseek2025r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={{DeepSeek-AI} and Guo, Daya and Yang, Dejian and others},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}

@article{schulman2017ppo,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}

@misc{schulman2020klapprox,
  title={Approximating {KL} Divergence},
  author={Schulman, John},
  year={2020},
  howpublished={\url{http://joschu.net/blog/kl-approx.html}}
}

@inproceedings{huang2021tdc,
  title={Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, Connor W and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  booktitle={Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2021}
}

@article{olivecrona2017reinvent,
  title={Molecular De-novo Design through Deep Reinforcement Learning},
  author={Olivecrona, Marcus and Blaschke, Thomas and Engkvist, Ola and Chen, Hongming},
  journal={Journal of Cheminformatics},
  volume={9},
  number={1},
  pages={48},
  year={2017},
  doi={10.1186/s13321-017-0235-x}
}

@article{veith2009cyp,
  title={Comprehensive Characterization of Cytochrome {P450} Isozyme Selectivity Across Chemical Libraries},
  author={Veith, Henrike and Southall, Noel and Huang, Ruili and James, Tim and Fayne, Darren and Artemenko, Natasha and Shen, Min and Inglese, James and Austin, Christopher P and Lloyd, David G and Auld, Douglas S},
  journal={Nature Biotechnology},
  volume={27},
  number={11},
  pages={1050--1055},
  year={2009}
}

@article{bemis1996scaffold,
  title={The Properties of Known Drugs. 1. Molecular Frameworks},
  author={Bemis, Guy W and Murcko, Mark A},
  journal={Journal of Medicinal Chemistry},
  volume={39},
  number={15},
  pages={2887--2893},
  year={1996}
}

@article{brown2019guacamol,
  title={GuacaMol: Benchmarking Models for de Novo Molecular Design},
  author={Brown, Nathan and Fiscato, Marco and Segler, Marwin H S and Vaucher, Alain C},
  journal={Journal of Chemical Information and Modeling},
  volume={59},
  number={3},
  pages={1096--1108},
  year={2019},
  doi={10.1021/acs.jcim.8b00839}
}

@article{polykovskiy2020moses,
  title={Molecular Sets ({MOSES}): A Benchmarking Platform for Molecular Generation Models},
  author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and Veselov, Mark and Kadurin, Artur and Johansson, Simon and Chen, Hongming and Nikolenko, Sergey and Aspuru-Guzik, Alan and Zhavoronkov, Alex},
  journal={Frontiers in Pharmacology},
  volume={11},
  pages={565644},
  year={2020},
  doi={10.3389/fphar.2020.565644}
}

@article{krenn2020selfies,
  title={Self-referencing Embedded Strings ({SELFIES}): A 100\% Robust Molecular String Representation},
  author={Krenn, Mario and H{\"a}se, Florian and Nigam, AkshatKumar and Friederich, Pascal and Aspuru-Guzik, Al{\'a}n},
  journal={Machine Learning: Science and Technology},
  volume={1},
  number={4},
  pages={045024},
  year={2020},
  doi={10.1088/2632-2153/aba947}
}

@article{krenn2022selfiesfuture,
  title={{SELFIES} and the Future of Molecular String Representations},
  author={Krenn, Mario and Ai, Qianxiang and Barthel, Senja and Carson, Nessa and Frei, Angelo and Frey, Nathan C and others},
  journal={Patterns},
  volume={3},
  number={10},
  pages={100588},
  year={2022}
}

@misc{metaopenenv2025,
  title={{OpenEnv}: An Interface Library for {RL} Post-training with Environments},
  author={{Meta-PyTorch} and {Hugging Face}},
  year={2025},
  howpublished={\url{https://github.com/meta-pytorch/OpenEnv}},
  note={Released at PyTorch Conference 2025; v0.2.3 March 2026}
}

@misc{unsloth2024,
  title={Unsloth: Fast {LoRA} Fine-tuning for {LLMs}},
  author={Han, Daniel and Han, Michael},
  year={2024},
  howpublished={\url{https://github.com/unslothai/unsloth}}
}

@inproceedings{hu2022lora,
  title={{LoRA}: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{dettmers2023qlora,
  title={{QLoRA}: Efficient Finetuning of Quantized {LLMs}},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@misc{metallama32,
  title={Llama 3.2: Revolutionizing Edge {AI} and Vision with Open, Customizable Models},
  author={{Meta AI}},
  year={2024},
  month={September},
  howpublished={\url{https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/}}
}
```

---

## Methodology Recipe for Our Paper

Drop the following directly into the Methods section of *AI Alchemy in Medicine*:

- **Base policy.** Llama-3.2-3B-Instruct (Meta AI, 2024 [19]) loaded in 4-bit NF4 via QLoRA
  ([18] Dettmers et al., 2023) using Unsloth's `FastLanguageModel` ([16] Han & Han, 2024).
- **LoRA configuration.** Target modules: all linear projections (`q,k,v,o,gate,up,down`).
  Rank $r=16$, $\alpha=32$, dropout $0$, initialization per [17] (Hu et al., 2022).
  Cite the LoRA decomposition $W = W_0 + \tfrac{\alpha}{r}BA$.
- **Action representation.** SELFIES tokens ([10] Krenn et al., 2020) — guarantee 100% valid
  decode and clean RL gradients.
- **Algorithm: GRPO** ([1] Shao et al., 2024). Drop in the full GRPO objective from §1.1, the
  group-standardized advantage from §1.2, and the K3 KL term from §1.3.
- **K3 KL estimator.** Cite [5] (Schulman, 2020). Use the formula
  $D_{\text{KL}} = r - 1 - \log r$ with $r = \pi_{\text{ref}}/\pi_\theta$ on a per-token basis.
- **PPO lineage.** Cite [4] (Schulman et al., 2017) for the clipped surrogate that GRPO
  inherits.
- **Hyperparameters.** Group size $G=8$, clip $\varepsilon = 0.2$, KL coefficient $\beta$ in
  $[0.001, 0.04]$ (DeepSeekMath uses $\beta=0.04$), learning rate $1\!\times\!10^{-5}$, AdamW.
- **Reward.** Convex combination over TDC oracles ([6] Huang et al., 2021): DRD2 (Olivecrona
  SVM [7]), GSK3B, JNK3, CYP3A4_Veith ([Veith2009]). Optionally add QED + SA-score from RDKit.
- **Environment infrastructure.** PharmaRL packaged as an OpenEnv-spec environment
  ([14, 15] Meta-PyTorch & HF, 2025): FastAPI server, Pydantic-typed `MoleculeAction` /
  `MoleculeObservation` / `OracleState`, Docker-isolated, pushed to HF Spaces. Cite the
  Gymnasium-like `step / reset / state / close` API and session-keyed state.
- **Evaluation.** Distribution-learning on MOSES ([9] Polykovskiy et al., 2020): validity,
  uniqueness, novelty, FCD, SNN, IntDiv, Filters, Scaff. Goal-directed on GuacaMol ([8] Brown
  et al., 2019): all 20 tasks, scaffold split per Bemis–Murcko (1996).
- **Successor framing.** Cite [2] DeepSeek-R1 (2025) as proof of GRPO at scale, and [3]
  ChemCRAFT / RePO (2026) as concurrent chemistry GRPO work — PharmaRL is the OpenEnv-native
  bridge.

---

## Sources

- [DeepSeekMath arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- [PPO arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- [Schulman KL blog](http://joschu.net/blog/kl-approx.html)
- [TDC arXiv:2102.09548](https://arxiv.org/abs/2102.09548) | [tdcommons.ai](https://tdcommons.ai)
- [Olivecrona REINVENT 2017](https://link.springer.com/article/10.1186/s13321-017-0235-x)
- [GuacaMol JCIM 2019](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) | [GitHub](https://github.com/BenevolentAI/guacamol)
- [MOSES Front. Pharmacol. 2020](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full) | [GitHub](https://github.com/molecularsets/moses)
- [SELFIES MLST 2020](https://iopscience.iop.org/article/10.1088/2632-2153/aba947) | [SELFIES future arXiv:2204.00056](https://arxiv.org/abs/2204.00056) | [Group SELFIES](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00012e)
- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv) | [OpenEnv docs](http://meta-pytorch.org/OpenEnv/) | [Meta blog](https://ai.meta.com/blog/introducing-pytorch-native-agentic-stack/)
- [Unsloth](https://unsloth.ai/) | [Dynamic 4-bit](https://unsloth.ai/blog/dynamic-4bit)
- [LoRA arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- [QLoRA arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- [Llama 3.2 announcement](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) | [Llama 3.2 3B HF model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
