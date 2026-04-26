# Drug Chemistry Foundations for PharmaRL

Technical literature review grounding the chemistry side of "AI Alchemy In Medicine: A Vision for LLM-as-Policy Molecular Design." Each section gives the primary citation, the technical claim, the documented limitations, and what PharmaRL should emphasize when using the metric as part of an RL reward.

---

## 1. Lipinski's Rule of Five (Ro5)

### Primary citations
- **Lipinski, C. A.; Lombardo, F.; Dominy, B. W.; Feeney, P. J.** "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." *Advanced Drug Delivery Reviews* **23** (1997) 3-25 [1]. Reissued in *Adv. Drug Deliv. Rev.* **46** (2001) 3-26 with minor updates [2]. Lipinski's 2012 retrospective: "The role of physicochemical properties in compound 'developability'" [3].
- **Veber, D. F.; Johnson, S. R.; Cheng, H.-Y.; Smith, B. R.; Ward, K. W.; Kopple, K. D.** "Molecular properties that influence the oral bioavailability of drug candidates." *J. Med. Chem.* **45** (2002) 2615-2623. doi:10.1021/jm020017n [4].
- **Hughes, J. D.; Blagg, J.; Price, D. A.; Bailey, S.; DeCrescenzo, G. A.; Devraj, R. V.; Ellsworth, E.; Fobian, Y. M.; Gibbs, M. E.; Gilles, R. W.; et al.** "Physiochemical drug properties associated with in vivo toxicological outcomes." *Bioorg. Med. Chem. Lett.* **18** (2008) 4872-4875 [5].

### Technical claim
Ro5 predicts poor absorption or permeation when a candidate violates two or more of: MW > 500 Da, calculated logP (cLogP) > 5, hydrogen-bond donors > 5, hydrogen-bond acceptors > 10. The cutoffs were derived empirically from a dataset of compounds that had reached Phase II clinical trials at Pfizer in the early 1990s; the rule is heuristic, *not* a physical model. Veber's 2002 supplement adds two flexibility-driven cutoffs (rotatable bonds <= 10, polar surface area <= 140 A^2 or HBD+HBA <= 12) and reports that ~85% of orally administered drugs satisfy them [4]. The Pfizer "3/75" rule [5] flags toxicity risk: cLogP > 3 and TPSA < 75 A^2 increases the probability of in vivo adverse outcomes by ~2.5x. GSK proposed a "4/400" companion (cLogP < 4, MW < 400) tied to compound quality. Ghose, Viswanadhan and Wendoloski (1999) earlier offered a tighter qualifying range used by ZINC and DrugBank: 160 <= MW <= 480, -0.4 <= ALogP <= 5.6, 40 <= MR <= 130, 20 <= n_atoms <= 70 [6].

### Documented limitations / failure modes
The rule was never intended to predict drug efficacy and explicitly excludes natural products and substrates of biological transporters (the original paper says so). It is well documented that 20 of 48 FDA-approved small-molecule kinase inhibitors violate Ro5; a more recent 2023 audit of 74 approved kinase inhibitors finds an even higher violation rate [7]. Concrete violators: **dabrafenib** (MW 519.6, four +H-bond acceptors above the limit and HBA > 10), **ceritinib** (MW 558.1), **lapatinib** (MW 581), **nilotinib** (MW 529), and bosutinib. None of those would have survived a strict Ro5 filter. PROTACs (proteolysis-targeting chimeras) routinely sit at MW 800-1100 with high TPSA and high rotatable-bond counts; nearly all PROTAC chemical matter is "beyond rule of 5" (bRo5) and yet several clinical PROTACs achieve oral exposure via "molecular chameleonicity" - intramolecular hydrogen bonding that masks polarity in membranes [8]. Macrocyclic natural products (rapamycin, cyclosporin A) and modern protein-protein interaction inhibitors live almost entirely outside Ro5. DeGoey et al. 2018 showed that 22 of 105 oral drugs approved 2014-2018 were bRo5 [9]. The current consensus, captured in the 2023 *J. Med. Chem.* "Beyond Rule of Five and PROTACs in Modern Drug Discovery" paper [8]: Ro5 is a useful filter for *most* oral small-molecule projects targeting enzymes with classical active sites, but it is positively misleading for PPI inhibitors, kinase inhibitors with type II/III modes, targeted protein degraders, and macrocycles. Newer multi-parameter scores (CNS MPO, AbbVie AB-MPS, EPSA/TPSA ratio) supplement or replace Ro5 in those contexts [10].

### Relevance to PharmaRL reward
Use Ro5 as a *soft, low-weight* component, not a hard veto. Hard-Ro5 vetoes will collapse the policy onto a tiny island of small, hydrophilic chemistry and bias against any kinase-like or PPI-like exploration. PharmaRL should report Ro5 *compliance* alongside generation, but the composite reward should let high-utility violators through, ideally with target-class-specific bRo5 escapes (e.g., relax MW/HBA when the oracle is a kinase such as DRD2). Veber-style flexibility metrics arguably correlate better with rat oral bioavailability than Ro5 does and are cheap to compute.

---

## 2. QED - Quantitative Estimate of Drug-likeness

### Primary citation
**Bickerton, G. R.; Paolini, G. V.; Besnard, J.; Muresan, S.; Hopkins, A. L.** "Quantifying the chemical beauty of drugs." *Nature Chemistry* **4** (2012) 90-98. doi:10.1038/nchem.1243 [11].

### Technical claim
QED combines eight molecular properties into a single drug-likeness score in [0,1]: (1) molecular weight, (2) ALogP, (3) hydrogen-bond donors, (4) hydrogen-bond acceptors, (5) molecular polar surface area, (6) rotatable bonds, (7) aromatic rings, and (8) count of structural alerts (94 reactive/mutagenic moieties, including the Brenk and PAINS-like lists). Each property is mapped through an *asymmetric double-sigmoidal (ADS)* desirability function `d(x) = a + b*[1+exp(-(x-c+d)/(2e))]*[1-1/(1+exp(-(x-c-d)/(2f)))]` whose six parameters were fit to the empirical distribution of those properties in 771 orally absorbed approved drugs from ChEMBL DrugStore. The composite is the geometric mean of desirabilities: `QED_u = exp((1/n) sum ln d_i)` for the unweighted form, or `QED_w = exp((sum w_i ln d_i)/sum w_i)` weighted [11]. The authors used Shannon-entropy-maximizing weight ensembles to derive `QED_w_max` and `QED_w_mo` (mean of the 1000 highest-entropy weight vectors). At threshold 0.35, QED_w_mo achieved 48% greater specificity than the Ro5 binary at matched sensitivity in their drug-vs-non-drug benchmark.

### Documented limitations
The authors themselves caution that "evaluation of drug-likeness in absolute terms does not adequately reflect the whole spectrum of compound quality." Specific issues that have appeared in the literature since: (a) **Distinguishing drugs from non-drugs is weak.** QED separates approved drugs from generic ChEMBL molecules only with modest AUC (~0.76 in the original paper), and follow-on work has shown it cannot reliably separate clinical-stage compounds from random Enamine-screening-deck molecules [12]. (b) **The 8 properties miss target binding entirely** - QED reports nothing about pharmacology. (c) **It is reward-hackable.** Renz, Van Rompaey, Wegner, Hochreiter and Klambauer 2020 ("On failure modes in molecule generation and optimization") show that QED-optimizing generators converge to small, mid-LogP, low-alert molecules that cluster in a narrow region of chemical space and are not actually drug-like [13]. The structural-alert filter is binary and brittle: removing any single alert flips the score discontinuously. (d) **Training set leakage**: the original benchmark validation overlapped 554 compounds with the positive training set. (e) **pKa is omitted**, by the authors' own admission, which limits its applicability to ionizable molecules. The 2024 *DrugMetric* paper [14] argues that distance-in-chemical-space metrics are a stronger drug-likeness proxy than QED and proposes replacing QED in optimization pipelines.

### Relevance to PharmaRL reward
Treat QED as *the canonical multiproperty drug-likeness component* but be explicit about the failure mode: a policy that maximises QED alone will produce visually drug-like but pharmacologically inert molecules. Couple QED with a target-binding oracle (DRD2/GSK3B/JNK3 from TDC) and a synthesizability term so the policy must trade off across axes the way a med chemist would. The paper should explicitly call out reward-hacking on QED as a motivating problem PharmaRL is designed to expose, rather than something hidden behind composite arithmetic.

---

## 3. SA Score - Synthetic Accessibility

### Primary citation
**Ertl, P.; Schuffenhauer, A.** "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions." *Journal of Cheminformatics* **1** (2009) 8. doi:10.1186/1758-2946-1-8 [15].

### Technical claim
SAscore is the difference between a *fragment score* (built from frequencies of ECFC_4 atom-centred fragments observed in 934,046 PubChem molecules) and a *complexity penalty*. The fragment score is the log ratio of the actual count of each fragment to the count of fragments forming the 80th percentile of the database; common fragments contribute positively, rare fragments negatively. The complexity penalty sums (i) a ring-complexity term (spiro rings, non-standard fusions), (ii) a stereo-complexity term (stereocenter count), (iii) a macrocycle penalty (rings > 8 atoms), and (iv) a size penalty. The combined score is sign-flipped and rescaled to [1, 10] where 1 means easy to make and 10 means very hard. Validated against rankings from 9 experienced Novartis medicinal chemists on 40 molecules with r^2 = 0.89 [15].

### Documented limitations
SAscore is heuristic: it has no notion of *what* synthesis route is required and never invokes a retrosynthesis engine. Coley et al. (SCScore, 2018) and Thakkar et al. (RAscore, 2021) [16] showed that on out-of-distribution molecules - peptidomimetics, macrocycles, novel scaffolds - SAscore disagrees with both expert opinion and AiZynthFinder retrosynthesis success. The 2023 *J. Cheminform.* "Critical assessment of synthetic accessibility scores in computer-assisted synthesis planning" [17] benchmarked SAscore, SYBA, SCScore and RAscore against retrosynthesis-engine outcomes and found that SAscore correlates with chemist intuition for "ordinary" drug-like molecules but breaks down for chiral-dense molecules and for fragments that simply happen to be rare in PubChem despite being trivially synthesizable (e.g., common fluorinated heterocycles introduced after the training cutoff). RAscore (Thakkar 2021) trains a binary classifier on AiZynthFinder outcomes; it is faster than SAscore but inherits AiZynthFinder's blind spots. FSscore (2023) [18] addresses chirality-discrimination failures of SA by fine-tuning on chemist preference pairs. In short: SA is fast and decent on average, but it under-penalises hidden retrosynthesis difficulty and over-penalises modern motifs.

### Relevance to PharmaRL reward
Include SAscore as the synthesizability proxy because it is fast, deterministic and what almost every published RL paper uses; that makes our results comparable. The vision paper should *also* propose an upgrade path to RAscore or AiZynthFinder-in-the-loop as a future-work axis, motivated by exactly the OOD failure modes documented above. The "edit budget" framing of LLM-as-policy maps naturally to retrosynthetic step count if the future SA plug-in is retrosynthesis-aware.

---

## 4. Pharmacophore Modeling

### Primary citation
**Wermuth, C. G.; Ganellin, C. R.; Lindberg, P.; Mitscher, L. A.** "Glossary of terms used in medicinal chemistry (IUPAC Recommendations 1998)." *Pure and Applied Chemistry* **70** (1998) 1129-1143 [19]. The pharmacophore definition: *"the ensemble of steric and electronic features that is necessary to ensure the optimal supramolecular interactions with a specific biological target structure and to trigger (or to block) its biological response."*

### Technical claim
A **2D pharmacophore** captures the connectivity skeleton between key interacting groups (donor, acceptor, hydrophobe, aromatic, positive, negative). A **3D pharmacophore** specifies the geometric arrangement of those features in space (typically as feature spheres with tolerance radii) and is the basis of pharmacophore-based virtual screening. Classical commercial implementations: Catalyst (now part of BIOVIA Discovery Studio), Schrödinger's **Phase**, **LigandScout** (Wolber & Langer 2005, *J. Chem. Inf. Model.* 45, 160) [20], and **MOE**. Recent (2023-2026) work has moved generative pharmacophore design into deep learning. **PharmacoForge** (2025) [21] is a diffusion model that generates 3D pharmacophore queries conditioned on a protein pocket; the resulting query is then used to retrieve commercially-available molecules. **PharmaDiff** (2025) [22] is a transformer-conditioned 3D diffusion model that generates molecules satisfying a supplied pharmacophore hypothesis, outperforming ligand-based methods on docking score even without the protein structure. **MolSnapper** (2024) [23] conditions DiffSBDD-style protein-pocket diffusion on expert-supplied 3D pharmacophore hot spots. **DiffPharma** (2026) [24] generates molecules satisfying interaction constraints derived from structure-based pharmacophore mining.

### Documented limitations
Pharmacophore models are extremely sensitive to the alignment of training ligands and to the "feature dictionary" assumed. Conformational searching is expensive and can miss bioactive conformers. Classical pharmacophore software requires expert hand-curation of feature tolerances; ML diffusion models inherit the statistical bias of their training set (e.g., PDBbind under-represents non-druggable pockets). The 2010 Yang review [25] cautions that 3D pharmacophore screening has high false-positive rates when used without docking refinement.

### Relevance to PharmaRL reward
Pharmacophore matching is *not* in the PharmaRL v1 reward but is the obvious next axis: the LLM-as-policy framing makes "edit this molecule until it satisfies pharmacophore P" a natural RL task, mirroring the conditional generation that PharmaDiff and MolSnapper achieve with diffusion. Cite this body of work as a natural extension and as the chemistry-aware structural prior PharmaRL could ingest in v2.

---

## 5. ADMET Prediction with Machine Learning

### Primary citations
- **Mayr, A.; Klambauer, G.; Unterthiner, T.; Hochreiter, S.** "DeepTox: Toxicity Prediction using Deep Learning." *Frontiers in Environmental Science* **3** (2016) 80 [26]. (Tox21 Challenge winner.)
- **Pires, D. E. V.; Blundell, T. L.; Ascher, D. B.** "pkCSM: Predicting Small-Molecule Pharmacokinetic and Toxicity Properties Using Graph-Based Signatures." *J. Med. Chem.* **58** (2015) 4066-4072. doi:10.1021/acs.jmedchem.5b00104 [27].
- **Swanson, K.; Walther, P.; Leitz, J.; Mukherjee, S.; Wu, J. C.; Shivnaraine, R. V.; Zou, J.** "ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries." *Bioinformatics* **40(7)** (2024) btae416. doi:10.1093/bioinformatics/btae416 [28].
- **Huang, K.; Fu, T.; Gao, W.; Zhao, Y.; Roohani, Y.; Leskovec, J.; Coley, C. W.; Xiao, C.; Sun, J.; Zitnik, M.** Therapeutics Data Commons (TDC) [29]. ICLR 2022 / NeurIPS Datasets & Benchmarks. The reference benchmark for ADMET prediction in 2024-2026.

### Technical claim
DeepTox demonstrated in 2014 that deep multitask neural networks beat random forests, SVMs, and naive Bayes baselines on the Tox21 challenge across 12 nuclear receptor and stress-response endpoints, learning intermediate representations that resemble classical toxicophores [26]. pkCSM (2015) uses graph-based atomic distance signatures plus general molecular properties to train 30 ADMET predictors (7 absorption, 4 distribution, 7 metabolism, 2 excretion, 10 toxicity) [27]; it achieves 83.8% accuracy on AMES mutagenicity. ADMET-AI (Stanford, 2024) is the current state-of-the-art end-user platform: 41 Chemprop-RDKit graph neural networks trained on TDC ADMET datasets, achieving the highest average TDC leaderboard rank, 45% faster than the next public web tool, and uniquely contextualizing predictions against DrugBank-approved drug distributions [28]. ADMET-AI was extended in 2025 to interpretable cardiotoxicity prediction.

### Documented limitations
The honest summary: ADMET prediction works well for endpoints with abundant well-curated data (CYP450 inhibition, hERG cardiotoxicity at the binary level, AMES mutagenicity, basic logP/solubility) and poorly for endpoints that are noisy or sparse (hepatotoxicity, idiosyncratic toxicity, clinical drug-drug interactions, BBB permeability for novel scaffolds). MoleculeNet, TDC and DeepChem benchmarks consistently show AUROCs of 0.85-0.90 on Tox21 endpoints but 0.70-0.80 on DILI and similar clinical endpoints. Data leakage between TDC train/test splits is a known issue; "scaffold split" performance is typically 5-15 AUROC points worse than random split. ADMET-AI itself notes that out-of-distribution molecules (large macrocycles, PROTACs) are flagged as low-confidence by design - the predictions only mean what the training distribution says they mean.

### Relevance to PharmaRL reward
TDC integrates naturally with PharmaRL because both speak the same molecule-as-SMILES interface. Use TDC oracles (DRD2, GSK3B, JNK3 are targets, but Caco-2, BBB, hERG, CYP3A4 can join the composite reward) as differentiable-ish (actually black-box but cheap-to-call) rewards. Be explicit in the paper that ADMET predictions are confidence-bounded, motivate why an LLM policy that *understands* uncertainty is more appropriate than an RL policy that gradient-descents through it, and cite ADMET-AI as the de-facto ADMET reward backend.

---

## 6. Molecular Descriptors, Fingerprints, and Representations

### Primary citations
- **Rogers, D.; Hahn, M.** "Extended-Connectivity Fingerprints." *J. Chem. Inf. Model.* **50** (2010) 742-754. doi:10.1021/ci100050t [30].
- **Durant, J. L.; Leland, B. A.; Henry, D. R.; Nourse, J. G.** "Reoptimization of MDL keys for use in drug discovery." *J. Chem. Inf. Comput. Sci.* **42** (2002) 1273-1280 [31] (the MACCS-166 paper).
- **Krenn, M.; Hase, F.; Nigam, A.; Friederich, P.; Aspuru-Guzik, A.** "Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation." *Machine Learning: Science and Technology* **1** (2020) 045024. doi:10.1088/2632-2153/aba947 [32]. arXiv:1905.13741.
- **Weininger, D.** "SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules." *J. Chem. Inf. Comput. Sci.* **28** (1988) 31-36 [33] (the SMILES paper).

### Technical claim
**ECFP/Morgan** circular fingerprints iterate over atom neighborhoods up to a fixed radius (typically 2 or 3, giving ECFP4 / ECFP6) using a Morgan-like update; each iteration's atom invariants are hashed and the union is folded into a fixed-length bit vector (commonly 1024 or 2048 bits) [30]. ECFP4 is the default similarity measure in pharmaceutical informatics. **MACCS** keys are 166 expert-curated SMARTS substructure flags [31]; widely used in similarity searches but limited in expressiveness. **RDKit topological fingerprints** are Daylight-style hashed path fingerprints. RDKit also exposes ~200 physicochemical descriptors covering Lipinski/Veber properties, Crippen LogP, TPSA, BertzCT, Kier shape indices, and the Mordred package extends this to ~1800 2D/3D descriptors. **SMILES** (1988) [33] is the de-facto string format but ~10-20% of randomly-generated SMILES strings are syntactically invalid and ~5% more are chemically invalid; this is fatal for unconstrained generation. **SELFIES** (Krenn et al. 2020) [32] solves this with a small context-free grammar plus self-referencing branch/ring counters, guaranteeing that *any* token sequence parses to a valid molecular graph. SELFIES-based GANs produce 78.9% diverse valid molecules vs 18.6% for SMILES GANs [32]. **Graph neural networks** (Gilmer et al. 2017 "Neural Message Passing"; Yang et al. 2019 D-MPNN/Chemprop) bypass the string-vs-string question entirely and operate on molecular graphs directly; they excel at per-atom property prediction (e.g., site of metabolism) and are the backbone of ADMET-AI and most modern property predictors.

### Documented limitations
Bit collisions in folded ECFP fingerprints destroy interpretability: at 1024 bits, several million distinct fragments collide into the same bit. MACCS-166 misses any structural feature not in the original 166-key dictionary - it cannot distinguish a fluorinated heterocycle the dictionary did not anticipate. SMILES is non-canonical (multiple valid strings for the same molecule unless canonicalized) which complicates sequence-model training; data augmentation by SMILES randomization helps but is not a clean fix. SELFIES, while valid-by-construction, can encode "weird" molecules that are valid graphs but synthetically nonsense - validity does not imply drug-likeness. Graph representations are permutation-invariant only if the GNN is properly equivariant; many early GNNs were not.

### Relevance to PharmaRL reward
PharmaRL's core argument is *LLM as policy*, which means the policy is tokenizer-bound - an LLM-as-policy on SMILES inherits SMILES validity problems unless we use SELFIES, post-hoc validation, or constrained decoding. We should be explicit in the paper about which representation we chose and why; if SELFIES, cite Krenn 2020 and emphasize the validity guarantee as why LLM-as-policy is even tractable. ECFP fingerprints stay relevant for the *similarity-to-known-actives* secondary reward and for any nearest-neighbor lookup in the action space.

---

## Key Takeaways for Our Paper

- **Composite rewards are necessary but reward-hackable.** Cite Renz/Klambauer 2020 [13] explicitly. Frame PharmaRL not as "we add up Lipinski + QED + SA + TDC oracle and call it good" but as a deliberate vehicle to *expose* and *study* multi-objective reward hacking with an LLM policy whose edits are interpretable.
- **Lipinski is a heuristic, not a law.** Mention dabrafenib, ceritinib, PROTACs explicitly. Argue that PharmaRL must allow controlled bRo5 exploration when the target class warrants it - a hard Ro5 veto would have rejected ~40% of approved kinase inhibitors.
- **QED has known structural-alert and reward-hacking failure modes.** Use it but report alongside QED a chemical-space-distance check (DrugMetric-style) to detect mode collapse. The vision paper should call out that "high QED" is necessary but profoundly insufficient.
- **SAscore is the standard, but it is a heuristic.** Position RAscore / AiZynthFinder-in-the-loop as the v2 upgrade, especially for non-drug-like chemical space (macrocycles, PROTACs). Frame this as an axis where LLM-as-policy edits map naturally to retrosynthesis steps.
- **TDC + ADMET-AI are the right benchmark substrate.** Both are open, both are LLM-tokenizer-friendly, both are what the field will demand we compare against. Position PharmaRL squarely in the TDC ecosystem with the standard DRD2/GSK3B/JNK3 oracles plus a small set of ADMET-AI calls (Caco-2, hERG, BBB) to argue for "drug-like *and* developable."
- **Representation choice is policy-defining.** SELFIES guarantees molecular validity at the token level and is the cleanest fit for an LLM-as-policy. If the paper uses SMILES, justify it; if SELFIES, cite Krenn 2020. Either way the discussion belongs in the methods section, not buried.

---

## BibTeX Entries

```bibtex
@article{lipinski1997,
  author    = {Lipinski, Christopher A. and Lombardo, Franco and Dominy, Beryl W. and Feeney, Paul J.},
  title     = {Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings},
  journal   = {Advanced Drug Delivery Reviews},
  volume    = {23},
  number    = {1-3},
  pages     = {3--25},
  year      = {1997},
  doi       = {10.1016/S0169-409X(96)00423-1}
}

@article{lipinski2001,
  author    = {Lipinski, Christopher A. and Lombardo, Franco and Dominy, Beryl W. and Feeney, Paul J.},
  title     = {Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings},
  journal   = {Advanced Drug Delivery Reviews},
  volume    = {46},
  number    = {1-3},
  pages     = {3--26},
  year      = {2001},
  doi       = {10.1016/s0169-409x(00)00129-0}
}

@article{lipinski2004,
  author    = {Lipinski, Christopher A.},
  title     = {Lead- and drug-like compounds: the rule-of-five revolution},
  journal   = {Drug Discovery Today: Technologies},
  volume    = {1},
  number    = {4},
  pages     = {337--341},
  year      = {2004},
  doi       = {10.1016/j.ddtec.2004.11.007}
}

@article{veber2002,
  author    = {Veber, Daniel F. and Johnson, Stephen R. and Cheng, Hung-Yuan and Smith, Brian R. and Ward, Keith W. and Kopple, Kenneth D.},
  title     = {Molecular Properties That Influence the Oral Bioavailability of Drug Candidates},
  journal   = {Journal of Medicinal Chemistry},
  volume    = {45},
  number    = {12},
  pages     = {2615--2623},
  year      = {2002},
  doi       = {10.1021/jm020017n}
}

@article{hughes2008,
  author    = {Hughes, Jason D. and Blagg, Julian and Price, David A. and Bailey, Simon and DeCrescenzo, Gary A. and Devraj, Rajesh V. and Ellsworth, Edmund and Fobian, Yvette M. and Gibbs, Michael E. and Gilles, Roy W. and others},
  title     = {Physiochemical drug properties associated with in vivo toxicological outcomes},
  journal   = {Bioorganic \& Medicinal Chemistry Letters},
  volume    = {18},
  number    = {17},
  pages     = {4872--4875},
  year      = {2008},
  doi       = {10.1016/j.bmcl.2008.07.071}
}

@article{ghose1999,
  author    = {Ghose, Arup K. and Viswanadhan, Vellarkad N. and Wendoloski, John J.},
  title     = {A Knowledge-Based Approach in Designing Combinatorial or Medicinal Chemistry Libraries for Drug Discovery},
  journal   = {Journal of Combinatorial Chemistry},
  volume    = {1},
  number    = {1},
  pages     = {55--68},
  year      = {1999},
  doi       = {10.1021/cc9800071}
}

@article{bhullar2023ro5kinase,
  author    = {Roskoski, Robert},
  title     = {Properties of FDA-approved small molecule protein kinase inhibitors: A 2023 update},
  journal   = {Pharmacological Research},
  volume    = {187},
  pages     = {106552},
  year      = {2023},
  doi       = {10.1016/j.phrs.2022.106552}
}

@article{degoey2023bro5,
  author    = {DeGoey, David A. and Chen, Hwan-Jung and Cox, Philip B. and Wendt, Michael D.},
  title     = {Beyond Rule of Five and PROTACs in Modern Drug Discovery: Polarity Reducers, Chameleonicity, and the Evolving Physicochemical Landscape},
  journal   = {Journal of Medicinal Chemistry},
  volume    = {67},
  pages     = {2624--2643},
  year      = {2024},
  doi       = {10.1021/acs.jmedchem.3c02332}
}

@article{degoey2018,
  author    = {DeGoey, David A. and Chen, Hwan-Jung and Cox, Philip B. and Wendt, Michael D.},
  title     = {Beyond the Rule of 5: Lessons Learned from AbbVie's Drugs and Compound Collection},
  journal   = {Journal of Medicinal Chemistry},
  volume    = {61},
  number    = {7},
  pages     = {2636--2651},
  year      = {2018},
  doi       = {10.1021/acs.jmedchem.7b00717}
}

@article{wager2010cnsmpo,
  author    = {Wager, Travis T. and Hou, Xinjun and Verhoest, Patrick R. and Villalobos, Anabella},
  title     = {Moving beyond Rules: The Development of a Central Nervous System Multiparameter Optimization (CNS MPO) Approach To Enable Alignment of Druglike Properties},
  journal   = {ACS Chemical Neuroscience},
  volume    = {1},
  number    = {6},
  pages     = {435--449},
  year      = {2010},
  doi       = {10.1021/cn100008c}
}

@article{bickerton2012qed,
  author    = {Bickerton, G. Richard and Paolini, Gaia V. and Besnard, J{\'e}r{\'e}my and Muresan, Sorel and Hopkins, Andrew L.},
  title     = {Quantifying the chemical beauty of drugs},
  journal   = {Nature Chemistry},
  volume    = {4},
  number    = {2},
  pages     = {90--98},
  year      = {2012},
  doi       = {10.1038/nchem.1243}
}

@article{kohlbacher2024drugmetric,
  author    = {Cao, Bing and Wang, Hao and Du, Mengkai and Yang, Lan and others},
  title     = {DrugMetric: quantitative drug-likeness scoring based on chemical space distance},
  journal   = {Briefings in Bioinformatics},
  volume    = {25},
  number    = {4},
  pages     = {bbae321},
  year      = {2024},
  doi       = {10.1093/bib/bbae321}
}

@article{renz2020failure,
  author    = {Renz, Philipp and Van Rompaey, Dries and Wegner, Joerg Kurt and Hochreiter, Sepp and Klambauer, G{\"u}nter},
  title     = {On failure modes in molecule generation and optimization},
  journal   = {Drug Discovery Today: Technologies},
  volume    = {32-33},
  pages     = {55--63},
  year      = {2020},
  doi       = {10.1016/j.ddtec.2020.09.003}
}

@article{ertl2009sa,
  author    = {Ertl, Peter and Schuffenhauer, Ansgar},
  title     = {Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions},
  journal   = {Journal of Cheminformatics},
  volume    = {1},
  number    = {1},
  pages     = {8},
  year      = {2009},
  doi       = {10.1186/1758-2946-1-8}
}

@article{thakkar2021rascore,
  author    = {Thakkar, Amol and Chadimov{\'a}, Veronika and Bjerrum, Esben Jannik and Engkvist, Ola and Reymond, Jean-Louis},
  title     = {Retrosynthetic accessibility score (RAscore) -- rapid machine learned synthesizability classification from AI driven retrosynthetic planning},
  journal   = {Chemical Science},
  volume    = {12},
  pages     = {3339--3349},
  year      = {2021},
  doi       = {10.1039/D0SC05401A}
}

@article{skoraczynski2023sacritical,
  author    = {Skoraczy{\'n}ski, Grzegorz and Kitlas, Mateusz and Miasojedow, B{\l}a{\.z}ej and Gambin, Anna},
  title     = {Critical assessment of synthetic accessibility scores in computer-assisted synthesis planning},
  journal   = {Journal of Cheminformatics},
  volume    = {15},
  pages     = {6},
  year      = {2023},
  doi       = {10.1186/s13321-023-00678-z}
}

@article{neeser2023fsscore,
  author    = {Neeser, Rebecca M. and Akdel, Mehmet and Kovtun, Daria and Naef, Luca},
  title     = {FSscore: A Machine Learning-based Synthetic Feasibility Score Leveraging Human Expertise},
  journal   = {arXiv preprint arXiv:2312.12737},
  year      = {2023},
  doi       = {10.48550/arXiv.2312.12737}
}

@article{wermuth1998iupac,
  author    = {Wermuth, Camille G. and Ganellin, C. Robin and Lindberg, P. and Mitscher, Lester A.},
  title     = {Glossary of terms used in medicinal chemistry (IUPAC Recommendations 1998)},
  journal   = {Pure and Applied Chemistry},
  volume    = {70},
  number    = {5},
  pages     = {1129--1143},
  year      = {1998},
  doi       = {10.1351/pac199870051129}
}

@article{wolber2005ligandscout,
  author    = {Wolber, Gerhard and Langer, Thierry},
  title     = {LigandScout: 3-D pharmacophores derived from protein-bound ligands and their use as virtual screening filters},
  journal   = {Journal of Chemical Information and Modeling},
  volume    = {45},
  number    = {1},
  pages     = {160--169},
  year      = {2005},
  doi       = {10.1021/ci049885e}
}

@article{pharmacoforge2025,
  author    = {Sunseri, Jocelyn and Ragoza, Matthew and Koes, David Ryan},
  title     = {PharmacoForge: pharmacophore generation with diffusion models},
  journal   = {Frontiers in Bioinformatics},
  volume    = {5},
  pages     = {1628800},
  year      = {2025},
  doi       = {10.3389/fbinf.2025.1628800}
}

@article{pharmadiff2025,
  author    = {Anonymous},
  title     = {Pharmacophore-Conditioned Diffusion Model for Ligand-Based De Novo Drug Design},
  journal   = {arXiv preprint arXiv:2505.10545},
  year      = {2025}
}

@article{molsnapper2024,
  author    = {Ziv, Yuval and Marsden, Brian and Deane, Charlotte M.},
  title     = {MolSnapper: Conditioning Diffusion for Structure-Based Drug Design},
  journal   = {Journal of Chemical Information and Modeling},
  volume    = {64},
  pages     = {8398--8408},
  year      = {2024},
  doi       = {10.1021/acs.jcim.4c02008}
}

@article{diffpharma2026,
  author    = {Anonymous},
  title     = {Interaction-constrained 3D molecular generation using a diffusion model enables structure-based pharmacophore modeling for drug design},
  journal   = {npj Drug Discovery},
  year      = {2026},
  doi       = {10.1038/s44386-026-00040-x}
}

@article{yang2010pharmacophore,
  author    = {Yang, Sheng-Yong},
  title     = {Pharmacophore modeling and applications in drug discovery: challenges and recent advances},
  journal   = {Drug Discovery Today},
  volume    = {15},
  number    = {11-12},
  pages     = {444--450},
  year      = {2010},
  doi       = {10.1016/j.drudis.2010.03.013}
}

@article{mayr2016deeptox,
  author    = {Mayr, Andreas and Klambauer, G{\"u}nter and Unterthiner, Thomas and Hochreiter, Sepp},
  title     = {DeepTox: Toxicity Prediction using Deep Learning},
  journal   = {Frontiers in Environmental Science},
  volume    = {3},
  pages     = {80},
  year      = {2016},
  doi       = {10.3389/fenvs.2015.00080}
}

@article{pires2015pkcsm,
  author    = {Pires, Douglas E. V. and Blundell, Tom L. and Ascher, David B.},
  title     = {pkCSM: Predicting Small-Molecule Pharmacokinetic and Toxicity Properties Using Graph-Based Signatures},
  journal   = {Journal of Medicinal Chemistry},
  volume    = {58},
  number    = {9},
  pages     = {4066--4072},
  year      = {2015},
  doi       = {10.1021/acs.jmedchem.5b00104}
}

@article{swanson2024admetai,
  author    = {Swanson, Kyle and Walther, Parker and Leitz, Jeremy and Mukherjee, Souhrid and Wu, Joseph C. and Shivnaraine, Rabindra V. and Zou, James},
  title     = {ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries},
  journal   = {Bioinformatics},
  volume    = {40},
  number    = {7},
  pages     = {btae416},
  year      = {2024},
  doi       = {10.1093/bioinformatics/btae416}
}

@inproceedings{huang2021tdc,
  author    = {Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, Connor W. and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  title     = {Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
  booktitle = {Proceedings of NeurIPS Datasets and Benchmarks},
  year      = {2021}
}

@article{rogers2010ecfp,
  author    = {Rogers, David and Hahn, Mathew},
  title     = {Extended-Connectivity Fingerprints},
  journal   = {Journal of Chemical Information and Modeling},
  volume    = {50},
  number    = {5},
  pages     = {742--754},
  year      = {2010},
  doi       = {10.1021/ci100050t}
}

@article{durant2002maccs,
  author    = {Durant, Joseph L. and Leland, Burton A. and Henry, Douglas R. and Nourse, James G.},
  title     = {Reoptimization of MDL Keys for Use in Drug Discovery},
  journal   = {Journal of Chemical Information and Computer Sciences},
  volume    = {42},
  number    = {6},
  pages     = {1273--1280},
  year      = {2002},
  doi       = {10.1021/ci010132r}
}

@article{krenn2020selfies,
  author    = {Krenn, Mario and H{\"a}se, Florian and Nigam, AkshatKumar and Friederich, Pascal and Aspuru-Guzik, Al{\'a}n},
  title     = {Self-referencing embedded strings (SELFIES): A 100\% robust molecular string representation},
  journal   = {Machine Learning: Science and Technology},
  volume    = {1},
  number    = {4},
  pages     = {045024},
  year      = {2020},
  doi       = {10.1088/2632-2153/aba947}
}

@article{weininger1988smiles,
  author    = {Weininger, David},
  title     = {SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules},
  journal   = {Journal of Chemical Information and Computer Sciences},
  volume    = {28},
  number    = {1},
  pages     = {31--36},
  year      = {1988},
  doi       = {10.1021/ci00057a005}
}

@inproceedings{gilmer2017mpnn,
  author    = {Gilmer, Justin and Schoenholz, Samuel S. and Riley, Patrick F. and Vinyals, Oriol and Dahl, George E.},
  title     = {Neural Message Passing for Quantum Chemistry},
  booktitle = {ICML},
  year      = {2017}
}

@article{yang2019chemprop,
  author    = {Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and Eiden, Philipp and Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy and Kelley, Brian and Mathea, Miriam and others},
  title     = {Analyzing Learned Molecular Representations for Property Prediction},
  journal   = {Journal of Chemical Information and Modeling},
  volume    = {59},
  number    = {8},
  pages     = {3370--3388},
  year      = {2019},
  doi       = {10.1021/acs.jcim.9b00237}
}
```

---

### Citation key (in-body numbering)

[1] Lipinski 1997; [2] Lipinski 2001; [3] Lipinski 2004; [4] Veber 2002; [5] Hughes 2008; [6] Ghose 1999; [7] Roskoski 2023 kinase inhibitor audit; [8] DeGoey 2024 *J. Med. Chem.* bRo5/PROTAC review; [9] DeGoey 2018; [10] Wager 2010 CNS MPO; [11] Bickerton 2012 QED; [12] DrugMetric 2024; [13] Renz/Klambauer 2020 failure modes; [14] DrugMetric (same as [12]); [15] Ertl/Schuffenhauer 2009 SA; [16] Thakkar 2021 RAscore; [17] Skoraczynski 2023 SA critical assessment; [18] Neeser 2023 FSscore; [19] Wermuth 1998 IUPAC; [20] Wolber/Langer 2005 LigandScout; [21] PharmacoForge 2025; [22] PharmaDiff 2025; [23] MolSnapper 2024; [24] DiffPharma 2026; [25] Yang 2010 pharmacophore review; [26] Mayr 2016 DeepTox; [27] Pires 2015 pkCSM; [28] Swanson 2024 ADMET-AI; [29] Huang 2021 TDC; [30] Rogers 2010 ECFP; [31] Durant 2002 MACCS; [32] Krenn 2020 SELFIES; [33] Weininger 1988 SMILES.
