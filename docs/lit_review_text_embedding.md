# Literature Review: Learning the Text-Side (Rank) Embeddings of CLIP for Few-Shot Ordinal Medical Grading

*Scope:* methods that produce and learn the **text-side class/rank embeddings** of CLIP-style models beyond plain CoOp/CoCoOp/MaPLe prompt learning. The emphasis is on ordinal classification/regression and medical imaging, especially diabetic retinopathy (DR) grading. Most papers are from 2023–2026; a few earlier foundational papers are included for context.
*Compiled:* 2026-10-17. *Status of claims:* entries marked ✔ were checked against the arXiv/proceedings page during this review. Entries marked ○ are summarised from prior knowledge of the paper and should be re-checked before they are cited in a manuscript.

---

## 1. Where our current model sits

Our current text branch is `RankFormerPromptLearner`. It uses 16 shared context tokens and 5 rank tokens, which pass through MHSA and then the **frozen** CLIP RN50 text encoder to give the prototypes t_k. It is trained with CE, the CLIP-DR ranking-aware loss, and an L2RCLIP-style pairwise loss, with SMS calibration at inference. In the taxonomy below this is a **"prompt tokens → frozen encoder"** method, the same family as OrdinalCLIP, L2RCLIP and CLIP-DR. The question for the next phase is how to move to a text branch that is **learned end-to-end in embedding space** and is still novel.

## 2. Taxonomy of how text embeddings are produced and learned

| Family | What is learned | Gradient path | Representative work |
|---|---|---|---|
| **A. Prompt tokens** | Context/rank token vectors in the word-embedding space | through the frozen text encoder | CoOp, CoCoOp, MaPLe, KgCoOp, TCP; OrdinalCLIP, L2RCLIP, CLIP-DR |
| **B. Direct embedding / classifier-weight learning** | The class prototypes t_k ∈ ℝ^d themselves (or a residual on them), initialised from zero-shot text | none through the encoder (the encoder is used only once, for initialisation) | TaskRes, CLAP, LP++/LP+text, ProKeR, SS-Text, ProLIP-style |
| **C. Text-encoder adaptation** | LoRA or partial fine-tuning of the text encoder weights | through the encoder | CLIP-LoRA, MMLoP, LoRA-adapted DR VLMs |
| **D. Generated descriptions / LLM priors** | Usually nothing; prototypes are averaged embeddings of LLM-written descriptions. Sometimes these are combined with A–C | n/a or as in A–C | CuPL, DCLIP, WaffleCLIP; concept-guided retinal prompting; KOAL anchors; VL-OrdinalFormer |
| **E. Numeric/verbal re-labelling** | Ranks re-expressed as language concepts, followed by regularisation | as in A | NumCLIP |

Most ordinal-CLIP work sits in **A** (+E). Most of the strong *few-shot* general-domain and medical work from 2023–2025 sits in **B** or **C**. **No paper we found combines B with explicit ordinal constraints on the prototypes.** Section 5 develops this point.

---

## 3. Ordinal / ranking CLIP methods (general domain)

### OrdinalCLIP — NeurIPS 2022 ✔ (foundational, pre-2023)
- **Text embedding:** learnable context tokens + **learnable rank embeddings**. The rank embeddings are not free vectors. They are built by **linear interpolation from a small set of base rank embeddings**, which gives continuity between ranks. They pass through the frozen CLIP text encoder, and the resulting language prototypes can be cached after training.
- **Ordinal structure:** enforced *implicitly* by the interpolation parameterisation (neighbouring ranks share bases). There is no explicit ordinal loss on the prototypes.
- **Few-shot:** few-shot and distribution-shift experiments on MORPH II age estimation. Full-data MAE 2.32 on MORPH II (vs. CoOp 2.39 and zero-shot CLIP 14.45).
- **Code:** <https://github.com/xk-huang/OrdinalCLIP>
- **Relevance:** CLIP-DR notes that base-embedding interpolation is hard to use with only 5 grades. This is one motivation for working in embedding space with explicit constraints instead.

### L2RCLIP — "Learning-to-Rank Meets Language", NeurIPS 2023 ✔
- **Text embedding:** **RankFormer**, which applies token-level attention with *residual-style prompt blending in the word-embedding space* over the rank prompts. Prompts go through the frozen text encoder. (Our `RankFormerPromptLearner` follows this design.)
- **Ordinal structure:** a **cross-modal ordinal pairwise loss**, derived by revisiting the approximate bound optimisation of cross-entropy, that refines the joint space so image–text similarities respect rank order.
- **Few-shot:** reports gains in few-shot and distribution-shift settings (age, HCI dating, aesthetics).
- **Code:** <https://github.com/raywang335/L2RCLIP>

### NumCLIP — "Teach CLIP to Develop a Number Sense for Ordinal Regression", ECCV 2024 ✔
- **Text embedding:** two stages. In the coarse stage, numeric bins are **re-expressed as everyday language concepts** (e.g. "teenager", "elderly") so they match CLIP's pre-training. The fine stage uses learnable prompts through the frozen encoder (ViT-B/16), following the OrdinalCLIP setup.
- **Ordinal structure:** a **cross-modal ranking-based feature regularisation** that keeps label order in the feature space.
- **Few-shot:** 1-shot: −1.33 MAE on age estimation, +10% accuracy on historical image dating, +3.83% on aesthetics. Full-data MORPH II MAE 2.08.
- **Code:** <https://github.com/xmed-lab/NumCLIP>
- **Relevance:** comes from a medical-imaging lab (XMed, HKUST) but is evaluated on natural images. The "verbal concepts instead of numbers" idea transfers directly to DR ("no / mild / moderate / severe / proliferative" already *is* verbal).

### Ranking-aware adapter for text-driven image ordering — ICLR 2025 ✔
- **Text embedding:** learnable prompts driven by a ranking instruction. A lightweight adapter adds a **ranking-aware attention branch** that uses text-conditioned visual differences between images.
- **Ordinal structure:** learning-to-rank objective on image pairs (relative ordering, not fixed grades).
- **Code:** <https://github.com/uynaes/RankingAwareCLIP>
- **Relevance:** shows that ICLR-level work frames CLIP ordering as pairwise LTR. This is a possible pairwise baseline.

### HiMo-CLIP (arXiv 2511.06653, 2025) ✔ (title/abstract only)
- Models *semantic hierarchy and monotonicity* in vision–language alignment. It is not ordinal classification, but the **monotonicity-of-similarity constraint** idea is related.

---

## 4. Medical / DR-specific ordinal VLM work

### CLIP-DR — MICCAI 2024 ✔
- **Text embedding:** **ranking-aware prompting**. Learnable prompts are placed between *neighbouring* grade text–image pairs in **two ranking directions** (ascending and descending), starting from a template like "This image is {label}". The text encoder is frozen.
- **Ordinal structure:** a ranking loss that forces similarity to fall off monotonically from the true grade, plus **Similarity Matrix Smoothing (SMS)** for class imbalance. (We already use both.)
- **Few-shot:** none. Evaluated as **domain generalisation on GDRBench** (6 DR datasets): average F1 45.5, AUC 82.7 for the full model.
- **Code:** <https://github.com/Qinkaiyu/CLIP-DR>
- **Relevance:** the direct predecessor of our project. Its prompts still go through the frozen encoder and it has no few-shot protocol.

### CLIP-based Ranking-Aware Adaptation for DR: a comparative study — arXiv 2603.13403 (Mar 2026) ✔
- Compares zero-shot CLIP, a hybrid FCN-CLIP + CBAM model, and a CLIP-DR-style ranking-aware prompting model on APTOS-2019 + Messidor-2 (5,406 images). Ranking-aware prompting reaches 93.42% accuracy and AUROC 0.9845; zero-shot reaches 55.17%.
- No new text-embedding mechanism and no few-shot results. Code is "planned". Mainly useful as evidence that ranking-aware prompting is now a standard DR baseline.

### Text-guided lesion-mining VL ordinal classification for DR — J. Imaging (MDPI) 2025 ✔ (search snippet; full text blocked)
- Builds on the **RetiZero** retinal VLM. It uses global grade prompts ("A fundus photograph of {}") plus **local lesion prompts**, a conditional ordinal-regression branch, and **ordinal distribution alignment**.
- Text embeddings come from hand-written prompts. The ordinal structure lives in the regression head and the distribution alignment, not in the text prototypes. No few-shot results found.
- Link: <https://www.mdpi.com/2313-433X/12/9/412>

### Cross-Modal Semantic-Enhanced Diffusion for DR grading — arXiv 2605.09242 (May 2026) ✔
- Grade text-description features come from a domain VLM **LoRA-adapted** to the target domain (family C). No ordinal constraint and no few-shot results; APTOS only (acc. 87.5%, macro-F1 0.731). No code.

### Interpretable Few-Shot Retinal Disease Diagnosis with Concept-Guided Prompting — IPMI 2025 ✔
- **Text embedding:** a GPT-derived **concept bank** of retinal-disease findings used as the language component of CoOp/CoCoOp-style prompt learning (families D + A).
- **Ordinal:** none, and it covers multi-disease classification rather than DR grading.
- **Few-shot:** +≈5.8 mAP at 16-shot and +≈2.7 mAP for zero-shot novel classes.
- **Code:** none linked.

### Few-shot Adaptation of Medical VLMs — MICCAI 2024 (Shakeri et al.) ✔
- First structured few-shot benchmark for medical VLMs, including fundus with FLAIR. It proposes **LP+text**: a linear probe whose class weights **blend learnable visual prototypes with the pre-trained text embeddings** through class-wise multipliers (family B). This is competitive with prompt learning and adapters, much faster, and works in a black-box setting. 1-shot histology: 55.6% vs. CoOp 46.05%.
- **Ordinal:** none. **Code:** <https://github.com/FereshteShakeri/few-shot-MedVLMs>

### Few-Shot, Now for Real (SS-Text) — MICCAI 2025 ✔
- Shows that with **imbalanced support sets and no validation set** (the realistic case for DR, where grades 3–4 are rare), many few-shot adapters fall *below zero-shot*. Proposes a **training-free linear probe** that adaptively combines visual and text supervision (family B).
- **Ordinal:** none. **Code:** <https://github.com/jusiro/SS-Text>
- **Relevance:** this is the protocol critique that reviewers in this community apply. Any new method should be robust under it.

### VL-OrdinalFormer — arXiv 2601.00879 (Dec 2025), knee OA ✔
- ViT-L/16 + **CORAL** ordinal head + a CLIP-driven semantic-alignment module using fixed text embeddings of radiographic concepts (joint-space narrowing, osteophytes, sclerosis). The text is *not* learned.
- Ordinal structure comes from CORAL on the image side. No few-shot results and no code.

### KOAL — MICCAI 2026 (arXiv 2607.06019), prostate Gleason grading ✔
- An LLM extracts grade-specific findings from reports and guidelines to make **offline semantic anchors** (family D). Ordinality comes from Hierarchical Ordinal-aware Constraints and a differentiable mapping to Grade Group. It is unclear from the abstract whether the anchors are refined during training. No few-shot results.
- **Code:** <https://github.com/Gother-GZ/KOAL>
- **Relevance:** the closest *medical* work to "LLM-initialised grade anchors + ordinal constraints". It is fully supervised, though, and the anchors are not optimised as free embeddings with ordinal geometry.

### Retinal foundation VLMs (backbones, not text-learning methods) ✔
FLAIR (MedIA 2025, expert-knowledge text supervision), KeepFIT / MM-Retinal V1–V2, ViLReF, RET-CLIP, RetiZero, EyeCLIP. All report zero-/few-shot transfer. None learns rank-structured text prototypes for grading. Several (ViLReF, RET-CLIP, RetiZero) were pre-trained on private data. Swapping our RN50 OpenAI CLIP for FLAIR or KeepFIT is an orthogonal (and strong) baseline change.

---

## 5. Text-side learning beyond prompts (general few-shot CLIP)

| Paper | Venue | Text embedding mechanism | Constraint/prior | Code |
|---|---|---|---|---|
| **TaskRes** ✔ | CVPR 2023 | Frozen zero-shot text classifier **+ learnable residual** (prior-independent) | residual scale α | <https://github.com/geekyutao/TaskRes> |
| **CLAP** ✔ | CVPR 2024 | Linear probe **initialised with zero-shot text embeddings**. Class-adaptive penalty on deviation from that init, optimised with an Augmented Lagrangian; no validation set needed | per-class ℓ2 anchor to the zero-shot text | <https://github.com/jusiro/CLAP> |
| **LP++** ✔ | CVPR 2024 | Blends fixed text embeddings with visual prototypes through class-wise learnable multipliers | — | <https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline> ○ |
| **ProKeR** ✔ | CVPR 2025 | Kernel/RKHS view; the proximal regulariser keeps the classifier near the zero-shot text prior | RKHS proximity | ○ |
| **ProLIP** (arXiv 2410.05270) ✔ | 2024 | Fine-tunes CLIP's last visual projector; argues that **text embeddings used directly as classifier weights** is good practice | distance to pre-trained projector | ○ |
| **CLIP-LoRA** ✔ | CVPR-W 2024 | LoRA on **both encoders**, including the text encoder (family C); beats prompt and adapter methods on 11 datasets with fixed hyper-parameters | — | <https://github.com/MaxZanella/CLIP-LoRA> ○ |
| **KgCoOp** ○ | CVPR 2023 | Prompt tokens + loss keeping the learned text embedding close to the hand-crafted one | ℓ2 to zero-shot text | <https://github.com/htyao89/KgCoOp> |
| **TCP** ○ | CVPR 2024 | Class-aware *textual knowledge embedding* injected into prompt tokens | — | <https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning> |
| **CuPL / DCLIP / WaffleCLIP** ○ | ICCV 2023 / ICLR 2023 / ICCV 2023 | LLM-generated (or random-word) descriptions averaged into class prototypes | — | yes |

**What carries over:** in general-domain few-shot CLIP, the 2023–2025 trend has moved *away* from prompt tokens toward (i) **directly learning prototypes initialised from zero-shot text, with a proximity prior** (TaskRes → CLAP → ProKeR), and (ii) LoRA on the encoders (CLIP-LoRA). These methods are faster and more stable, and they win in few-shot settings. **None of them has any notion of ordering among classes.** Their regularisers are per-class anchors (ℓ2 to the init), not structural constraints *between* classes.

---

## 6. Gap analysis

**Question:** is there published work that learns the rank text embeddings **fully end-to-end in embedding space**, initialised from a text prompt and then optimised directly with ordinal constraints, for **few-shot ordinal medical grading**?

**Answer: no, as far as this search found** (arXiv, CVF, MICCAI 2024–2026, NeurIPS/ICLR/ECCV, MDPI; searched 2026-10). The pieces exist separately:

| Ingredient | Exists in | Missing piece |
|---|---|---|
| Rank prototypes learned for ordinal tasks | OrdinalCLIP, L2RCLIP, NumCLIP, CLIP-DR | all learned **through prompt tokens + frozen encoder**; ordinality comes from the parameterisation (interpolation, attention) or from image–text ranking losses, **not from geometric constraints on the prototypes themselves** |
| Prototypes learned directly, initialised from text, with a prior | TaskRes, CLAP, ProKeR, LP+text, SS-Text | **no ordinal structure**; prior is per-class and isotropic |
| Ordinal medical grading with text | CLIP-DR, lesion-mining DR, VL-OrdinalFormer, KOAL | fully supervised or DG, **no few-shot protocol**; text fixed or prompt-learned |
| Few-shot medical VLM adaptation | Shakeri'24, SS-Text'25, concept-guided IPMI'25 | **nominal classes only** |

So the defensible novelty claim is the *intersection*: **direct, text-initialised, ordinally-constrained prototype learning for few-shot (and imbalanced) ordinal medical grading**. Reviewers will ask why this is more than "CLAP + a ranking loss". The method therefore needs a principled ordinal structure on the prototypes (geometry, not just a loss), and it should be evaluated under the realistic imbalanced, no-validation protocol of SS-Text.

Caveat: MICCAI 2026 and late-2026 arXiv are only partly indexed. Re-run a targeted search (`"ordinal" AND ("CLIP" OR "vision-language") AND ("prototype" OR "anchor")`) right before submission.

---

## 7. Candidate novel directions

### Direction 1: Ordinally-constrained direct prototype learning (an "Ordinal CLAP")
- **Idea:** encode the 5 hand-written / LLM-described grade prompts **once** with the frozen encoder to get the init T⁰ = {t⁰_k}. Then treat T = {t_k} as free parameters (or T⁰ + ΔT residual) and optimise them end-to-end with the LoRA image encoder. Add **structural ordinal constraints between prototypes**:
  - *monotone geodesic distance:* d(t_i, t_j) increasing in |i − j|, enforced through hinge constraints or an Augmented-Lagrangian solve as in CLAP, with no validation set;
  - *path/curve prior:* the prototypes lie near a 1-D curve on the hypersphere (e.g. a slerp/Bezier path with learnable control points), so grade order is built into the geometry;
  - *anchor prior:* class-adaptive ℓ2 to T⁰ (CLAP-style) so language knowledge survives at 1–2 shots.
- **Why novel:** CLAP/TaskRes have no between-class structure. OrdinalCLIP's interpolation lives in *token* space and passes through the frozen encoder. Here the ordinal geometry is enforced *in the joint embedding space* where similarities are computed.
- **Closest prior to beat:** **CLAP** (CVPR 2024) and **LP+text / SS-Text** (MICCAI 2024/2025) on the few-shot side; **L2RCLIP** and **CLIP-DR** (our current baseline) on the ordinal side. Use the same 1–32-shot DeepDRiD protocol, plus an imbalanced SS-Text-style protocol.

### Direction 2: Language-initialised, order-preserving text-encoder adaptation (LoRA-T with an ordinal manifold loss)
- **Idea:** keep text prompts discrete and clinically meaningful (ICDR grade definitions, optionally LLM-expanded with lesion findings as in CuPL or KOAL), but put **LoRA on the text encoder** (CLIP-LoRA style). Train it end-to-end with (a) the ranking-aware and pairwise losses already in use and (b) an ordinal-manifold regulariser on the *outputs*: neighbouring-grade prototypes stay closer than far grades, and per-grade description clouds stay ordered. At inference no prompt tokens are needed, and new descriptions (e.g. other ICDR wordings) can be encoded zero-shot.
- **Why novel:** CLIP-LoRA is nominal and natural-image. The 2026 diffusion-DR paper uses text LoRA but no ordinal constraint and no few-shot. CLIP-DR / L2RCLIP adapt prompts, not encoder weights.
- **Closest prior to beat:** **CLIP-LoRA** (both-encoder LoRA, with our ordinal losses added as a fair baseline) and **CLIP-DR**. Ablate text-LoRA vs. prompt tokens vs. direct prototypes (Direction 1).

### Direction 3: Probabilistic / distributional rank embeddings (uncertainty-aware ordinal text prototypes)
- **Idea:** model each grade as a **distribution** in embedding space (e.g. a von Mises–Fisher or Gaussian on the sphere) with mean initialised from the grade prompt and a learned concentration. Constrain the means to be ordered (as in Direction 1) and let concentrations capture inter-grader ambiguity, which is large between DR grades 1–2 and 2–3. The image–text score becomes an expected similarity or likelihood; ordinal smoothing (SMS) falls out of the overlap of neighbouring distributions instead of a post-hoc Gaussian kernel.
- **Why novel:** probabilistic ordinal embeddings exist on the image side (e.g. POE, CVPR 2021 ○) and RetiZero uses Dirichlet uncertainty for zero-shot recognition. However, no work found learns **distributional text-side rank prototypes** with ordinal constraints for few-shot grading. It also gives calibrated uncertainty, which is clinically attractive and directly comparable to our SMS step.
- **Closest prior to beat:** **CLIP-DR + SMS** (our calibration baseline), **NumCLIP** (ordinal regularisation), and for uncertainty, **RetiZero** zero-shot. Report ECE/QWK alongside accuracy.

**Suggested order:** Direction 1 first. It is the cleanest gap, is cheap to implement (it replaces `RankFormerPromptLearner` with a parameter tensor plus constraints), and gives an ablation axis (token-space vs. embedding-space vs. encoder-LoRA) that Directions 2 and 3 can reuse.

---

## 8. References

1. Li, W. et al. **OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression.** NeurIPS 2022. <https://arxiv.org/abs/2206.02338> · code <https://github.com/xk-huang/OrdinalCLIP>
2. Wang, R. et al. **Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification (L2RCLIP).** NeurIPS 2023. <https://arxiv.org/abs/2306.13856> · code <https://github.com/raywang335/L2RCLIP>
3. Du, Y., Zhai, Q., Dai, W., Li, X. **Teach CLIP to Develop a Number Sense for Ordinal Regression (NumCLIP).** ECCV 2024. <https://arxiv.org/abs/2408.03574> · code <https://github.com/xmed-lab/NumCLIP>
4. Yu, Q. et al. **CLIP-DR: Textual Knowledge-Guided Diabetic Retinopathy Grading with Ranking-aware Prompting.** MICCAI 2024. <https://arxiv.org/abs/2407.04068> · <https://papers.miccai.org/miccai-2024/141-Paper1493.html> · code <https://github.com/Qinkaiyu/CLIP-DR>
5. Yu, W.-H., Lin, Y.-Y., Yang, M.-H., Tsai, Y.-H. **Ranking-aware Adapter for Text-driven Image Ordering with CLIP.** ICLR 2025. <https://arxiv.org/abs/2412.06760> · code <https://github.com/uynaes/RankingAwareCLIP>
6. Cho, S. **Diabetic Retinopathy Grading with CLIP-based Ranking-Aware Adaptation: A Comparative Study on Fundus Images.** arXiv 2026. <https://arxiv.org/abs/2603.13403>
7. **A Text-Guided Lesion Mining Vision–Language Ordinal Classification Framework for Diabetic Retinopathy Grading.** J. Imaging 12(9):412, 2025. <https://www.mdpi.com/2313-433X/12/9/412>
8. Wang, Y. **Cross-Modal Semantic-Enhanced Diffusion Framework for Diabetic Retinopathy Grading.** arXiv 2026. <https://arxiv.org/abs/2605.09242>
9. Mehta, D. et al. **Interpretable Few-Shot Retinal Disease Diagnosis with Concept-Guided Prompting of Vision-Language Models.** IPMI 2025. <https://arxiv.org/abs/2503.02917>
10. Shakeri, F. et al. **Few-shot Adaptation of Medical Vision-Language Models.** MICCAI 2024. <https://arxiv.org/abs/2409.03868> · code <https://github.com/FereshteShakeri/few-shot-MedVLMs>
11. Silva-Rodríguez, J., Shakeri, F., Bahig, H., Dolz, J., Ben Ayed, I. **Few-Shot, Now for Real: Medical VLMs Adaptation without Balanced Sets or Validation.** MICCAI 2025. <https://arxiv.org/abs/2506.17500> · code <https://github.com/jusiro/SS-Text>
12. Ullah, Z., Kim, J. **VL-OrdinalFormer: Vision Language Guided Ordinal Transformers for Interpretable Knee Osteoarthritis Grading.** arXiv 2025. <https://arxiv.org/abs/2601.00879>
13. Guo, Z. et al. **KOAL: Knowledge-Driven Prostate Cancer Grading with Ordinal-Aware Learning.** MICCAI 2026. <https://arxiv.org/abs/2607.06019> · code <https://github.com/Gother-GZ/KOAL>
14. Yu, T. et al. **Task Residual for Tuning Vision-Language Models (TaskRes).** CVPR 2023. <https://arxiv.org/abs/2211.10277> · code <https://github.com/geekyutao/TaskRes>
15. Silva-Rodríguez, J., Hajimiri, S., Ben Ayed, I., Dolz, J. **A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models (CLAP).** CVPR 2024. <https://arxiv.org/abs/2312.12730> · code <https://github.com/jusiro/CLAP>
16. Huang, Y. et al. **LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP.** CVPR 2024. <https://arxiv.org/abs/2404.02285>
17. Bendou, Y. et al. **ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models.** CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/papers/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.pdf>
18. Fahes, M. et al. **Fine-tuning CLIP's Last Visual Projector: A Few-shot Cornucopia (ProLIP).** arXiv 2024. <https://arxiv.org/abs/2410.05270>
19. Zanella, M., Ben Ayed, I. **Low-Rank Few-Shot Adaptation of Vision-Language Models (CLIP-LoRA).** CVPR Workshops 2024. <https://arxiv.org/abs/2405.18541>
20. **MMLoP: Multi-Modal Low-Rank Prompting.** arXiv 2026. <https://arxiv.org/abs/2602.21397>
21. **HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment.** arXiv 2025. <https://arxiv.org/abs/2511.06653>
22. Yao, H., Zhang, R., Xu, C. **Visual-Language Prompt Tuning with Knowledge-guided Context Optimization (KgCoOp).** CVPR 2023. <https://arxiv.org/abs/2303.13283> ○
23. Yao, H., Zhang, R., Xu, C. **TCP: Textual-based Class-aware Prompt Tuning for Visual-Language Model.** CVPR 2024. <https://arxiv.org/abs/2311.18231> ○
24. Pratt, S. et al. **What does a platypus look like? Generating customized prompts for zero-shot image classification (CuPL).** ICCV 2023. <https://arxiv.org/abs/2209.03320> ○
25. Menon, S., Vondrick, C. **Visual Classification via Description from Large Language Models (DCLIP).** ICLR 2023. <https://arxiv.org/abs/2210.07183> ○
26. Roth, K. et al. **Waffling around for Performance: Visual Classification with Random Words and Broad Concepts (WaffleCLIP).** ICCV 2023. <https://arxiv.org/abs/2306.07282> ○
27. Silva-Rodríguez, J. et al. **A Foundation Language-Image Model of the Retina (FLAIR).** Medical Image Analysis 2025. <https://www.sciencedirect.com/science/article/abs/pii/S1361841524002822>
28. **MM-Retinal V2: Transfer an Elite Knowledge Spark into Fundus Vision-Language Pretraining (KeepFIT V2).** arXiv 2025. <https://arxiv.org/abs/2501.15798>
29. **ViLReF: An Expert Knowledge Enabled Vision-Language Retinal Foundation Model.** arXiv 2024. <https://arxiv.org/abs/2408.10894>
30. Zhou, K. et al. **Learning to Prompt for Vision-Language Models (CoOp)** IJCV 2022 / **CoCoOp** CVPR 2022; Khattak, M. U. et al. **MaPLe** CVPR 2023. <https://arxiv.org/abs/2109.01134> · <https://arxiv.org/abs/2203.05557> · <https://arxiv.org/abs/2210.03117> ○
31. Li, W. et al. **Learning Probabilistic Ordinal Embeddings for Uncertainty-Aware Regression (POE).** CVPR 2021. <https://arxiv.org/abs/2103.13629> ○
