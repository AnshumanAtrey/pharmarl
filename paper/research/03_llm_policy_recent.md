# Recent Work on LLM-as-Policy for Molecular Design

Compiled for *AI Alchemy In Medicine: A Vision for LLM-as-Policy Molecular Design*. Research cutoff: 2026-04-26. Emphasis on 2024-2026 sources.

---

## 1. Galactica (Meta AI, 2022)

Galactica [1] (Taylor, Kardas, Cucurull, Scialom, Hartshorn, Saravia, Poulton, Kerkez, Stojnic; arXiv:2211.09085) was Meta's flagship attempt at a single dense LLM "for science." Six checkpoints from 125M to 120B parameters were trained on a curated 48M-paper corpus plus millions of compounds and proteins, with a unified `<work>` tag for chain-of-thought reasoning. Headline numbers: **77.6% on PubMedQA** (SOTA at submission), **41.3% on mathematical MMLU** (vs. Chinchilla 35.7%), **20.4% on MATH** (vs. PaLM 540B 8.8%), **68.2% on LaTeX equation completion** (vs. GPT-3 49.0%).

The 120B demo went live on **15 November 2022** and was withdrawn **48 hours later** [2,3]. The collapse was not a benchmark failure but an out-of-distribution failure: Galactica fluently produced *plausibly-formatted but fabricated* citations, wiki entries for nonexistent compounds (the canonical example: "the history of bears in space"), and confident chemistry answers that were chemically nonsensical. Lead author Ross Taylor later attributed the launch failure to a 9-person team being "overstretched" rather than a fundamental scientific design flaw [3]. The lasting technical lesson: **a generative LLM with high lexical fluency in scientific text gives no purchase on the validity-vs-fluency gap that dominates chemistry**, and a one-shot autoregressive generation has no mechanism to query an oracle for ground truth.

For PharmaRL the relevance is direct: Galactica failed *because the LLM was the only loop*. PharmaRL puts an oracle (RDKit, QED, SA, scaffold checks) inside the env so the policy is grounded against deterministic chemistry every step.

---

## 2. ChemCrow (Bran et al., 2024)

ChemCrow [4] (Bran, Cox, Schilter, Baldassari, White, Schwaller; *Nature Machine Intelligence* 6:525–535, May 2024; arXiv:2304.05376) wraps GPT-4 with **18 expert-designed tools** covering web/literature search, RDKit operations, retrosynthesis (RXN, AskCos), reaction prediction, and SMILES utilities, orchestrated through a ReAct (Thought/Action/Observation) loop. The agent autonomously planned syntheses of an insect repellent and three organocatalysts, and guided discovery of a novel chromophore. ChemCrow significantly outperformed tool-less GPT-4 on expert and EvalChemy ratings, especially on tasks requiring grounded chemical reasoning.

Limits acknowledged: ChemCrow inherits GPT-4's reasoning errors when tool selection is ambiguous; it has no fine-tuning loop (the policy is frozen); it pays GPT-4 inference costs per step which makes it impractical as the *training-time* policy in an RL loop. ChemCrow demonstrates the *tool-augmentation hypothesis* but never closes the loop with policy optimization — exactly the gap PharmaRL targets.

---

## 3. ChemLLMBench / What can LLMs do in chemistry? (Guo et al., 2023)

Guo et al. [5] (NeurIPS 2023 Datasets & Benchmarks; arXiv:2305.18365) evaluated GPT-4, GPT-3.5, Davinci-003, LLaMA, and Galactica on **eight tasks**: name prediction, property prediction, yield prediction, reaction prediction, retrosynthesis, text-based molecule design, molecule captioning, reagent selection. Headline pattern: GPT models are competitive on **classification/ranking** tasks (yield, property) but **substantially underperform fine-tuned baselines on tasks requiring precise SMILES manipulation** (reaction prediction, retrosynthesis, name prediction). This pre-2024 result is the foundational empirical evidence for the SMILES-symbolic-fragility problem PharmaRL must defend against.

---

## 4. ChemBench (Mirza, Alampara, Kunchapu, Jablonka, 2025)

ChemBench [6] (Mirza et al., *Nature Chemistry*, May 2025) is the most rigorous current chemistry benchmark: **2,788 question–answer pairs** (1,039 manual, 1,749 semi-automated), spanning 6,202 MCQ + 857 open-ended items. Headline finding: top closed-source LLMs *on average* outperform expert chemists, but the variance is bimodal — models excel at textbook recitation (~71% accuracy on certification-exam-style items) and **fail on tasks requiring grounded reasoning** such as NMR signal prediction (<25% accuracy) and many stereochemistry items. The most damning finding for LLM-as-policy is **calibration failure**: one tested model gave maximum confidence (5/5) on incorrect chemical-safety answers; another's confidence scores barely differentiated correct from incorrect. **Overconfidence on chemistry is the paper's lead conclusion.** A companion piece, "Are large language models superhuman chemists?" [7] (*Nature Chemistry* 2025), argues the "superhuman" framing is misleading.

---

## 5. LlaSMol (Yu, Zhang, Sun, et al., 2024) and SMolInstruct

LlaSMol [8] (Yu et al., COLM 2024; arXiv:2402.09391) demonstrates that **fine-tuning on a high-quality instruction dataset closes the gap to GPT-4 with only 0.58% of parameters trained**. The released dataset SMolInstruct contains **3.3M samples across 14 chemistry tasks**. Trained variants (Galactica, Llama 2, CodeLlama, Mistral; LoRA fine-tuned) significantly outperform GPT-4 and Claude 3 Opus on the included tasks. **Mistral was the strongest base; canonical SMILES outperformed both non-canonical SMILES and SELFIES** — an evidence-backed argument for canonical SMILES as the env-level token format. This is the most direct prior art for "small open base + LoRA SFT" working in chemistry, but it is purely supervised; no RL.

---

## 6. SmileyLlama (Cavanagh, Sun, Gritsevskiy et al., 2024)

SmileyLlama [9] (arXiv:2409.02231) fine-tunes **Llama-3.1-8B-Instruct** on ~2M ChEMBL SMILES with SFT and then DPO against scoring functions (QED, GSK3β, JNK3, DRD2 oracles via TDCommons). Numbers worth quoting: **97.83% validity, 99.94% uniqueness, 97.13% novelty** at temperature 1.1 — comparable to dedicated CLMs (LSTM 98.28%, GPT 91.46%, S4 97.12%). Property-conditioned generation hits 97.1% on H-bond donor constraints, 92.8% on rule-of-three, 84.9% on Lipinski. The paper's honest **failure mode**: "crashing diversity in later epochs due to DPO causing the model to simply memorize the molecules it sees most often." That diversity collapse is exactly what GRPO with a group-relative baseline (and intrinsic-reward shaping à la Mol-AIR) is designed to mitigate.

---

## 7. MOLLEO (Wang et al., ICLR 2025)

MOLLEO [10] (Wang, Skreta, Ser et al., ICLR 2025; arXiv:2406.16976) replaces the random crossover/mutation operators of an evolutionary algorithm with chemistry-aware LLMs (GPT-4, BioT5, MoLFormer). On the **PMO benchmark** [11] (23 oracles including QED, DRD2, GSK3β, JNK3, 19 Guacamol oracles), MOLLEO(GPT-4) **outperformed all baselines on 9 of 12 single-objective tasks**. Limit: the LLM is *not* the policy — it is a black-box mutation operator inside a classical EA, frozen. The optimization signal never reaches the LLM. This is the "LLM-in-the-loop, not LLM-as-policy" failure mode PharmaRL inverts.

---

## 8. DrugAssist (Ye et al., 2024)

DrugAssist [12] (Ye et al., *Briefings in Bioinformatics* 2024; arXiv:2401.10334) fine-tunes **Llama-2-7B-Chat** on a released MolOpt-Instructions dataset for **interactive multi-turn molecule optimization through dialog**. Demonstrates the chat-loop pattern but uses SFT only; no RL post-training, no oracle-grounded reward. PharmaRL's session-keyed env state is the natural extension of DrugAssist's dialog into an RL trajectory.

---

## 9. ChemLLM / ChemDFM (2024)

ChemLLM [13] (Zhang et al., arXiv:2402.06852) fine-tunes **InternLM2-Base-7B** on ChemData and beats GPT-3.5 across the board, surpassing GPT-4 on six of nine ChemBench tasks. ChemDFM [14] (Zhao, Ma et al., arXiv:2401.14818) is a comparable Chinese-led foundation model using a domain-specialization stage on chemistry literature. Both validate the **7B-class open base + chemistry-domain SFT** recipe but neither does RL.

---

## 10. Inverse Scaling and Embers of Autoregression

The McCoy et al. *embers* paper [15] (McCoy, Yao, Friedman, Hardy, Griffiths; arXiv:2309.13638; *PNAS* Oct 2024) provides the theoretical lens. They predict and measure that LLM accuracy depends on three probabilities: task probability, output probability, input probability. The dramatic example: **GPT-4 decodes a simple cipher with 51% accuracy when the output is a high-probability English word sequence and only 13% when the output is low-probability** — a 38-point gap *in a deterministic task*. A follow-up [16] showed even o1 retains these embers under reasoning training.

The chemistry implication is direct. SMILES strings are *not* high-probability sequences in pretraining; stereodescriptors `[C@@H]`, ring closures across distant tokens, and chiral specifications are **systematically low-probability**. Empirical confirmations: the 2024 *Practical Cheminformatics* review [17] documented that GPT-4o1 omits square brackets around stereocenters (yielding invalid SMILES), Claude pathologically extends alkyl chains ("methyl, ethyl, futile"), and ~20% of generated structures contain unparseable tokens like raw "CF3" or "CH3" instead of valid `C(F)(F)F` or `C`. Larger models do not consistently fix this — **the *Measuring Chemical LLM Robustness* framework** [18] (*Journal of Cheminformatics* 2025) documents non-monotonic robustness across SMILES variations even within model families. Combined with the original *Inverse Scaling* prize [19] (McKenzie et al., arXiv:2306.09479), this is the published evidence for *bigger is not always better* in chemistry — which justifies the PharmaRL bet on a 3B base trained against a deterministic oracle.

---

## 11. Mol-AIR and the Diversity-Reward Problem

Mol-AIR [20] (Lim et al., *J. Chem. Inf. Model.* 2024; arXiv:2403.20109) shows that goal-directed molecular RL traps require **adaptive intrinsic rewards** (random distillation + counting-based bonuses) to escape mode collapse; the system reaches the QED theoretical optimum of 0.948 where prior baselines plateaued. This is the strongest prior-art argument for shaping PharmaRL's reward with an exploration term beyond the property oracle alone.

---

## 12. ChemCRAFT and SMILES-GRPO (2026)

The closest contemporaneous concurrent work to PharmaRL is ChemCRAFT [21] (arXiv:2601.17687, Jan 2026), which proposes **SMILES-GRPO**: GRPO with a dense multidimensional reward composed of (i) structural validity via SMILES match + scaffold similarity, (ii) functional fidelity via functional-group / reaction-template alignment, and (iii) optimization success via property-improvement magnitude. Two-stage training: token-level cold start for tool-call format, then GRPO. They report outperforming cloud LLMs on molecular structure analysis, optimization, and synthesis-pathway prediction. **However**: ChemCRAFT is a tool-using *agent* framework with internal sandbox tooling — *not OpenEnv-native, not session-keyed in the OpenEnv sense, not published as a reusable env*. The training pipeline and the env are entangled. This is exactly the gap PharmaRL fills with an OpenEnv-native, hub-publishable env that is decoupled from the trainer. Adjacent: Chennakesavalu et al. [22] (arXiv:2604.16279, Apr 2026) formulate molecular tasks as RL environments and post-train a small model competitive with frontier models — but again do not standardize on OpenEnv.

---

## 13. RL on Proteins (GRPO-adjacent)

ProteinZero [23] (arXiv:2506.07459, Jun 2025) applies RAFT and GRPO to protein inverse-folding LMs and finds GRPO marginally best on most metrics. ProtRL (cited in the 2025 protein-design survey [24]) extends DPO and GRPO to protein engineering, designing low-nanomolar EGFR inhibitors. These are evidence that GRPO transfers to non-math/non-code biological domains, but **not yet to small-molecule drug design as the policy**. PharmaRL's contribution is to bring GRPO to small-molecule chemistry through an OpenEnv-standard env.

---

## 14. OpenEnv (Meta + Hugging Face, October 2025)

OpenEnv [25] (announced 23 Oct 2025 at PyTorch Conf 2025; meta-pytorch/OpenEnv on GitHub; HF blog) is the framework that defines PharmaRL's expected shape. Design philosophy [26]:

- **Three-method Gymnasium-style API**: `step()`, `reset()`, `state()`, returning a typed `StepResult` (observation + reward + done).
- **HTTP isolation**: each env runs as a FastAPI server inside Docker; clients are typed `EnvClient`s; agent code cannot bypass the env's contract.
- **Session-keyed state**: the `state()` method returns a per-session view; multi-turn trajectories belong to the env, not the trainer.
- **Hub publishing**: `openenv push` deploys the container to Hugging Face Spaces under huggingface.co/openenv, integrating with TRL, TorchForge, verl, SkyRL, Lightning.AI.
- **RFC-driven**: RFC 002 (env spec, isolation) and RFC 004 (actions-as-tool-calls + delayed/trajectory rewards) are directly relevant to PharmaRL.

29 reference envs ship in `envs/` (echo, coding, chess, browsergym, atari, reasoning_gym, calendar, kernrl, julia, etc.) — none of them are chemistry. **There is no published OpenEnv-native chemistry env at the time of writing.** That is the empty quadrant PharmaRL claims.

---

## Gap Analysis

The literature splits cleanly into three buckets, and PharmaRL sits at an unfilled intersection. **Bucket 1 — Tool-using LLM agents** (ChemCrow, DrugAssist, ChemDFM): demonstrate that wrapping a frozen LLM with chemistry tools beats the bare LLM, but the policy is never updated against task reward and inference cost is per-step GPT-4. **Bucket 2 — Chemistry-fine-tuned open LLMs** (LlaSMol, SmileyLlama, ChemLLM): show that LoRA-class SFT/DPO on canonical-SMILES instruction data closes the gap to closed frontier models, but their RL signal (where present) is offline DPO against a precomputed pairwise dataset, with documented diversity collapse and no OpenEnv-style env interface. **Bucket 3 — RL for molecular design** (Reinvent, Mol-AIR, MOLLEO, ChemCRAFT, Chennakesavalu et al.): apply REINFORCE/PPO/GFlowNets/EAs and recently GRPO to molecular optimization, but their environments are bespoke trainer-coupled scripts, none are OpenEnv-native, and the training pipeline is entangled with the env logic. PharmaRL's specific contribution is **infrastructure, not training tricks**: an OpenEnv-native session-keyed env that issues RDKit-grounded oracle rewards over a JSON molecular-edit action space, exposes a Llama-3.2-3B-Instruct LoRA+GRPO recipe as a reproducible reference but is decoupled from any specific trainer (TRL, TorchForge, Unsloth all work), and is publishable to the OpenEnv Hub so subsequent chemistry work inherits a standard interface — closing the gap that ChemCRAFT *almost* closes but loses by entangling agent and env. We additionally claim the first published GRPO-on-SMILES recipe distributed as an OpenEnv `pip install`-able artifact, with explicit defenses against the documented failure modes (canonical SMILES per LlaSMol; diversity-aware reward shaping per Mol-AIR; deterministic RDKit grounding per Galactica's negative example).

---

## BibTeX Entries

```bibtex
@article{taylor2022galactica,
  title={Galactica: A Large Language Model for Science},
  author={Taylor, Ross and Kardas, Marcin and Cucurull, Guillem and Scialom, Thomas and Hartshorn, Anthony and Saravia, Elvis and Poulton, Andrew and Kerkez, Viktor and Stojnic, Robert},
  journal={arXiv preprint arXiv:2211.09085},
  year={2022}
}

@article{heaven2022galactica,
  title={Why Meta's Latest Large Language Model Survived Only Three Days Online},
  author={Heaven, Will Douglas},
  journal={MIT Technology Review},
  year={2022}
}

@article{venturebeat2023galactica,
  title={What Meta Learned from Galactica, the Doomed Model Launched Two Weeks Before {ChatGPT}},
  author={Goldman, Sharon},
  journal={VentureBeat},
  year={2023}
}

@article{bran2024chemcrow,
  title={Augmenting Large Language Models with Chemistry Tools},
  author={Bran, Andres M and Cox, Sam and Schilter, Oliver and Baldassari, Carlo and White, Andrew D and Schwaller, Philippe},
  journal={Nature Machine Intelligence},
  volume={6},
  pages={525--535},
  year={2024},
  doi={10.1038/s42256-024-00832-8}
}

@inproceedings{guo2023chemllmbench,
  title={What Can Large Language Models Do in Chemistry? A Comprehensive Benchmark on Eight Tasks},
  author={Guo, Taicheng and Guo, Kehan and Nan, Bozhao and Liang, Zhenwen and Guo, Zhichun and Chawla, Nitesh V and Wiest, Olaf and Zhang, Xiangliang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2023},
  eprint={2305.18365}
}

@article{mirza2025chembench,
  title={A Framework for Evaluating the Chemical Knowledge and Reasoning Abilities of Large Language Models Against the Expertise of Chemists},
  author={Mirza, Adrian and Alampara, Nawaf and Kunchapu, Sreekanth and Jablonka, Kevin Maik and others},
  journal={Nature Chemistry},
  year={2025},
  doi={10.1038/s41557-025-01815-x}
}

@article{mirza2025superhuman,
  title={Are Large Language Models Superhuman Chemists?},
  author={Mirza, Adrian and Alampara, Nawaf and Jablonka, Kevin Maik},
  journal={Nature Chemistry},
  year={2025},
  doi={10.1038/s41557-025-01865-1}
}

@inproceedings{yu2024llasmol,
  title={LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset},
  author={Yu, Botao and Baker, Frazier N and Chen, Ziru and Ning, Xia and Sun, Huan},
  booktitle={Conference on Language Modeling (COLM)},
  year={2024},
  eprint={2402.09391}
}

@article{cavanagh2024smileyllama,
  title={SmileyLlama: Modifying Large Language Models for Directed Chemical Space Exploration},
  author={Cavanagh, Joseph M and Sun, Kunyang and Gritsevskiy, Andrew and Bagni, Dorian and Wang, Yingze and Bannister, Thomas D and Head-Gordon, Teresa},
  journal={arXiv preprint arXiv:2409.02231},
  year={2024}
}

@inproceedings{wang2025molleo,
  title={Efficient Evolutionary Search Over Chemical Space with Large Language Models},
  author={Wang, Haorui and Skreta, Marta and Ser, Cher-Tian and Gao, Wenhao and Kong, Lingkai and Strieth-Kalthoff, Felix and Duan, Chenru and Zhuang, Yuchen and Yu, Yue and Zhu, Yanqiao and Du, Yuanqi and Aspuru-Guzik, Al{\'a}n and Neklyudov, Kirill and Zhang, Chao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  eprint={2406.16976}
}

@inproceedings{gao2022pmo,
  title={Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization},
  author={Gao, Wenhao and Fu, Tianfan and Sun, Jimeng and Coley, Connor W},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2022},
  eprint={2206.12411}
}

@article{ye2024drugassist,
  title={DrugAssist: A Large Language Model for Molecule Optimization},
  author={Ye, Geyan and Cai, Xibao and Lai, Houtim and Wang, Xing and Huang, Junhong and Wang, Longyue and Liu, Wei and Zeng, Xiangxiang},
  journal={Briefings in Bioinformatics},
  volume={26},
  number={1},
  pages={bbae693},
  year={2024},
  doi={10.1093/bib/bbae693}
}

@article{zhang2024chemllm,
  title={{ChemLLM}: A Chemical Large Language Model},
  author={Zhang, Di and Liu, Wei and Tan, Qian and Chen, Jingdan and Yan, Hang and Yan, Yuliang and Li, Jiatong and Huang, Weiran and Yue, Xiangyu and Zhou, Dongzhan and others},
  journal={arXiv preprint arXiv:2402.06852},
  year={2024}
}

@article{zhao2024chemdfm,
  title={Developing {ChemDFM} as a Large Language Foundation Model for Chemistry},
  author={Zhao, Zihan and Ma, Da and Chen, Lu and Sun, Liangtai and Li, Zihao and Xu, Hongshen and Zhu, Zichen and Zhu, Su and Fan, Shuai and Shen, Guodong and Chen, Xin and Yu, Kai},
  journal={arXiv preprint arXiv:2401.14818},
  year={2024}
}

@article{mccoy2024embers,
  title={Embers of Autoregression Show How Large Language Models Are Shaped by the Problem They Are Trained to Solve},
  author={McCoy, R Thomas and Yao, Shunyu and Friedman, Dan and Hardy, Matthew D and Griffiths, Thomas L},
  journal={Proceedings of the National Academy of Sciences (PNAS)},
  volume={121},
  number={41},
  pages={e2322420121},
  year={2024},
  doi={10.1073/pnas.2322420121},
  eprint={2309.13638}
}

@article{mccoy2024embers_o1,
  title={When a Language Model is Optimized for Reasoning, Does It Still Show Embers of Autoregression? An Analysis of {OpenAI} {o1}},
  author={McCoy, R Thomas and Yao, Shunyu and Friedman, Dan and Hardy, Matthew D and Griffiths, Thomas L},
  journal={arXiv preprint arXiv:2410.01792},
  year={2024}
}

@misc{walters2024silly,
  title={Silly Things Large Language Models Do With Molecules},
  author={Walters, Pat},
  howpublished={Practical Cheminformatics Blog},
  year={2024},
  url={https://practicalcheminformatics.blogspot.com/2024/10/silly-things-large-language-models-do.html}
}

@article{robustness2025smiles,
  title={Measuring Chemical {LLM} Robustness to Molecular Representations: A {SMILES} Variation-Based Framework},
  author={Anonymous},
  journal={Journal of Cheminformatics},
  year={2025},
  doi={10.1186/s13321-025-01079-0}
}

@article{mckenzie2023inverse,
  title={Inverse Scaling: When Bigger Isn't Better},
  author={McKenzie, Ian R and Lyzhov, Alexander and Pieler, Michael and Parrish, Alicia and Mueller, Aaron and Prabhu, Ameya and McLean, Euan and Kirtland, Aaron and Ross, Alexis and Liu, Alisa and others},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2023},
  eprint={2306.09479}
}

@article{lim2024molair,
  title={{Mol-AIR}: Molecular Reinforcement Learning with Adaptive Intrinsic Rewards for Goal-Directed Molecular Generation},
  author={Lim, Jinyeong and Park, Yujin and Heo, Donghyeon and Lee, Jaewoong},
  journal={Journal of Chemical Information and Modeling},
  year={2024},
  doi={10.1021/acs.jcim.4c01669},
  eprint={2403.20109}
}

@article{chemcraft2026,
  title={Agentic Reinforcement Learning Empowers Next-Generation Chemical Language Models for Molecular Design and Synthesis},
  author={Anonymous},
  journal={arXiv preprint arXiv:2601.17687},
  year={2026}
}

@article{chennakesavalu2026progression,
  title={Evaluating the Progression of Large Language Model Capabilities for Small-Molecule Drug Design},
  author={Chennakesavalu, Shriram and Shmilovich, Kirill and Weir, Hayley and Grambow, Colin and Bradshaw, John and Suriana, Patricia and Cheng, Chen and Chuang, Kangway},
  journal={arXiv preprint arXiv:2604.16279},
  year={2026}
}

@article{proteinzero2025,
  title={{ProteinZero}: Self-Improving Protein Generation via Online Reinforcement Learning},
  author={Anonymous},
  journal={arXiv preprint arXiv:2506.07459},
  year={2025}
}

@misc{protrl2025survey,
  title={Guiding Generative Models for Protein Design: Prompting, Steering and Aligning},
  author={Anonymous},
  journal={arXiv preprint arXiv:2511.21476},
  year={2025}
}

@misc{openenv2025,
  title={{OpenEnv}: Agentic Execution Environments},
  author={{Meta PyTorch} and {Hugging Face}},
  howpublished={meta-pytorch/OpenEnv (GitHub) and huggingface.co/blog/openenv},
  year={2025},
  note={Announced PyTorch Conference 2025, October 23, 2025}
}

@misc{huggingface2025openenv,
  title={Building the Open Agent Ecosystem Together: Introducing {OpenEnv}},
  author={{Hugging Face} and {Meta PyTorch}},
  howpublished={Hugging Face Blog},
  year={2025},
  url={https://huggingface.co/blog/openenv}
}

@inproceedings{shao2024deepseekmath,
  title={{DeepSeekMath}: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, YK and Wu, Y and Guo, Daya},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024},
  note={Original GRPO formulation}
}

@article{loeffler2024reinvent4,
  title={{Reinvent 4}: Modern {AI}-Driven Generative Molecule Design},
  author={Loeffler, Hannes H and He, Jiazhen and Tibo, Alessandro and Janet, Jon Paul and Voronov, Alexey and Mervin, Lewis H and Engkvist, Ola},
  journal={Journal of Cheminformatics},
  volume={16},
  number={20},
  year={2024},
  doi={10.1186/s13321-024-00812-5}
}
```

---

## References (numbered)

1. Taylor et al., *Galactica: A Large Language Model for Science*, arXiv:2211.09085 (2022).
2. *Why Meta's Latest LLM Survived Only Three Days Online*, MIT Technology Review (Nov 2022).
3. Goldman, *What Meta Learned from Galactica*, VentureBeat (2023).
4. Bran et al., *Augmenting Large Language Models with Chemistry Tools*, Nature Machine Intelligence 6:525–535 (2024).
5. Guo et al., *What Can LLMs Do in Chemistry?*, NeurIPS 2023 D&B Track, arXiv:2305.18365.
6. Mirza, Alampara, Kunchapu, Jablonka, *A framework for evaluating the chemical knowledge and reasoning abilities of LLMs against the expertise of chemists*, Nature Chemistry (2025).
7. Mirza et al., *Are Large Language Models Superhuman Chemists?*, Nature Chemistry (2025).
8. Yu et al., *LlaSMol*, COLM 2024, arXiv:2402.09391.
9. Cavanagh et al., *SmileyLlama*, arXiv:2409.02231 (2024).
10. Wang et al., *Efficient Evolutionary Search Over Chemical Space with LLMs (MOLLEO)*, ICLR 2025, arXiv:2406.16976.
11. Gao, Fu, Sun, Coley, *Sample Efficiency Matters (PMO)*, NeurIPS 2022 D&B, arXiv:2206.12411.
12. Ye et al., *DrugAssist*, Briefings in Bioinformatics (2024), arXiv:2401.10334.
13. Zhang et al., *ChemLLM*, arXiv:2402.06852 (2024).
14. Zhao et al., *ChemDFM*, arXiv:2401.14818 (2024).
15. McCoy, Yao, Friedman, Hardy, Griffiths, *Embers of Autoregression*, PNAS (2024), arXiv:2309.13638.
16. McCoy et al., *Embers of Autoregression in OpenAI o1*, arXiv:2410.01792 (2024).
17. Walters, *Silly Things LLMs Do With Molecules*, Practical Cheminformatics blog (Oct 2024).
18. *Measuring Chemical LLM Robustness*, J. Cheminformatics (2025).
19. McKenzie et al., *Inverse Scaling: When Bigger Isn't Better*, TMLR (2023), arXiv:2306.09479.
20. Lim et al., *Mol-AIR*, J. Chem. Inf. Model. (2024), arXiv:2403.20109.
21. *Agentic RL for Chemical Language Models (ChemCRAFT)*, arXiv:2601.17687 (2026).
22. Chennakesavalu et al., *Evaluating the Progression of LLM Capabilities for Small-Molecule Drug Design*, arXiv:2604.16279 (2026).
23. *ProteinZero*, arXiv:2506.07459 (2025).
24. *Guiding Generative Models for Protein Design (incl. ProtRL)*, arXiv:2511.21476 (2025).
25. Meta PyTorch & Hugging Face, *OpenEnv*, github.com/meta-pytorch/OpenEnv (Oct 2025).
26. Hugging Face Blog, *Introducing OpenEnv*, huggingface.co/blog/openenv (Oct 2025).
