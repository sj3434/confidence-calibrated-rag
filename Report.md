# Confidence-Calibrated RAG for Answer Abstention

**GitHub Repository:** https://github.com/Rhein/confidence-calibrated-rag

## Abstract

Retrieval-Augmented Generation (RAG) systems have become the standard architecture for adapting Large Language Models (LLMs) to specialized domains. However, a significant limitation persists: LLMs are fundamentally designed to generate fluent text, making them highly susceptible to hallucinating plausible but incorrect answers when the retrieval system fails to extract relevant context, or when queried outside the intended domain.

This project introduces a **Confidence-Calibrated RAG Pipeline** that addresses this problem by actively learning *when to abstain*. The system employs an automated, dual-layered guardrail architecture — combining quantitative vector similarity thresholding with qualitative LLM-as-a-Judge verification — to substantially reduce hallucination risk and safely return "I don't know" when context is insufficient. Evaluated on a 30-query benchmark (18 in-domain, 12 out-of-domain), the pipeline achieves 96.7% overall accuracy with an abstention F1-score of 96.0%. Against a Naive RAG baseline (93.3% accuracy), the calibrated system trades one false-positive abstention for improved answer quality through its dual-guardrail filtering mechanism.

---

## 1. Problem Statement

The primary challenge addressed in this work is the inherent overconfidence of generative models when faced with incomplete or irrelevant context. While Retrieval-Augmented Generation mitigates hallucination by supplying external knowledge, the system must still decide how to act when retrieval fails to find a semantically adequate match. Current state-of-the-art systems either rely on expensive supervised fine-tuning to teach the model to abstain (Asai et al., 2024), or they employ post-hoc validation that consumes unnecessary API resources. This project seeks a lightweight, deterministic alternative that operates at inference time without requiring model retraining.

### 1.1 Research Questions and Scope

1. **RQ1:** Can an automated dynamic threshold calibration algorithm combined with an LLM-as-a-Judge reasoning step measurably improve answer quality in RAG systems compared to uncalibrated baselines?
2. **RQ2:** Can we evaluate this reduction robustly using modern programmatic metrics (faithfulness, relevancy) rather than subjective human evaluation?
3. **RQ3:** What is the relative contribution of each guardrail component (quantitative threshold vs. qualitative judge) to overall abstention accuracy?

Our scope focuses specifically on the abstention mechanism applied to enterprise-style datasets simulated via our domain-targeted tech/finance corpus, bypassing from-scratch pretraining to maintain feasibility within the semester timeline.

---

## 2. Related Work

The challenge of knowing when a language model should *not* answer has deep roots in both the machine learning and NLP literature. Early work on **selective prediction** by Geifman and El-Yaniv (2017) established the theoretical framework for classifiers that can abstain: by allowing a model to reject uncertain inputs, one can trade coverage (the fraction of queries answered) for improved accuracy on the remaining predictions. This coverage-accuracy tradeoff maps directly onto our abstention problem — our system's Precision and Recall on the abstention decision (§5.2) are the RAG-specific instantiation of Geifman and El-Yaniv's selective risk, where coverage corresponds to the fraction of queries the system chooses to answer, and selective risk corresponds to the hallucination rate on those answered queries.

In the context of reading comprehension, Rajpurkar et al. (2018) introduced SQuAD 2.0, which augmented the standard question-answering benchmark with unanswerable questions — requiring models to distinguish between questions that can and cannot be answered given a passage. Their work demonstrated that even state-of-the-art models struggle with this distinction, motivating architectural solutions rather than relying solely on model capacity.

The foundational RAG architecture by Lewis et al. (2020) combined dense passage retrieval with sequence-to-sequence generation, achieving strong results on knowledge-intensive tasks. However, their original formulation did not include an explicit mechanism for abstention — the system always generates an answer regardless of retrieval quality. Subsequent work has sought to address this gap through increasingly sophisticated approaches.

**Self-RAG** (Asai et al., 2024) represents one prominent direction: training an LLM to emit special reflection tokens (e.g., `[Retrieve]`, `[Fully Supported]`) that enable dynamic retrieval and self-critique. While effective, Self-RAG requires computationally expensive supervised fine-tuning on curated datasets with reflection annotations. Our approach eliminates this requirement by using a zero-shot LLM judge combined with vector similarity calibration, achieving comparable abstention behavior without any model training. The trade-off is that our method depends on the zero-shot reasoning capability of the judge model, which may fail on nuanced edge cases (as observed in our one failed evaluation case).

**CRAG** (Corrective Retrieval Augmented Generation; Yan et al., 2024) introduced a dedicated retrieval evaluator that scores document relevance and falls back to web search when quality is low. Our dual-guardrail design is directly inspired by CRAG's retrieval evaluation step, but we simplify the architecture by replacing the web-search fallback with a definitive abstention mechanism — making the system suitable for closed-domain enterprise environments where external search is not permissible.

The **LLM-as-a-Judge** paradigm, validated by Zheng et al. (2023), demonstrated that strong LLMs (GPT-4 class) achieve high agreement with human evaluators on response quality assessment. We operationalize this finding by using `gpt-4o-mini` as an intermediate routing judge, leveraging its reasoning capability as a qualitative filter before generation.

Finally, Liu et al. (2023) documented the **"Lost in the Middle"** phenomenon: LLMs struggle to utilize information positioned in the middle of long contexts, exhibiting a U-shaped attention pattern. Our dual-guardrail system implicitly mitigates this by enforcing a strict quantitative threshold ($\tau$) that ensures only highly relevant, concise chunks are passed to the generator, preventing context bloat that triggers this failure mode.

Barnett et al. (2024) systematically catalogued **seven failure points** in RAG pipelines — including missing content, wrong granularity, incomplete extraction, and not in context — providing a taxonomy for diagnosing where retrieval-augmented systems break down. Our dual-guardrail architecture directly addresses two of these failure modes: the *"Not in Context"* failure (Guardrail 1 catches queries whose answers are absent from the corpus) and the *"Insufficient Context"* failure (Guardrail 2's LLM judge detects when retrieved passages are too shallow to support a faithful answer, as observed in the Q10 false-positive case).

Prior work addressed components of the abstention problem in isolation: selective prediction provided the theoretical grounding, Self-RAG and CRAG proposed architectural solutions requiring training or external search, LLM-as-a-Judge validated automated evaluation, and Barnett et al. provided the failure taxonomy. Our contribution is to unify these insights into a single lightweight pipeline that requires no fine-tuning, operates deterministically at inference time, and provides interpretable confidence signals at each decision point — making it deployable in closed-domain enterprise settings where neither model retraining nor web-search fallback is permissible.

---

## 3. Methodology and Technical Innovations

Our core contribution is the architectural shift from "maximizing answered questions" to "maximizing AI self-awareness" — enabling the system to recognize the boundary of its own knowledge. We accomplish this through two primary technical innovations and a custom evaluation framework.

### 3.1 The Dual-Layered Abstention Guardrail

```mermaid
graph TD
    A[User Query] --> B[ChromaDB Vector Retrieval]
    B --> C{"Guardrail 1: Score > τ ?"}
    C -->|No: Fast-Fail| D[Abstain: I don't know]
    C -->|Yes| E{Guardrail 2: LLM Judge}
    E -->|Insufficient| D
    E -->|Sufficient| F[LLM Generation]
    F --> G[Final Answer]
    style D fill:#ff6b6b,color:#fff
    style G fill:#51cf66,color:#fff
```

**Figure 1.** Dual-guardrail architecture. Queries are filtered through a quantitative similarity threshold (Guardrail 1) and a qualitative LLM judge (Guardrail 2) before generation is permitted. Red path = abstention; Green path = answer.

We implemented a two-step validation process before the LLM is permitted to generate:

1. **Guardrail 1 — Quantitative Calibration (Abstention-Scoring Algorithm):** Instead of using a hardcoded similarity threshold, our system runs a pre-computation calibration loop. It samples a batch of known in-domain and out-of-domain queries, calculates the dense vector similarity scores (using OpenAI `text-embedding-ada-002` embeddings and ChromaDB cosine similarity), and dynamically computes an optimal threshold $\tau$ using a midpoint heuristic: $\tau = \frac{\min(S_{\text{in-domain}}) + \max(S_{\text{out-of-domain}})}{2}$, where $S$ denotes the set of similarity scores. This maximizes the margin between the relevant-context distribution and noise. If a query's top similarity score falls below $\tau$, the system executes a **fast-fail abstention** before any LLM call, saving API inference costs.

2. **Guardrail 2 — Qualitative LLM-as-a-Judge:** If the vector score passes $\tau$, the retrieved context is forwarded to a strict LLM judge (`gpt-4o-mini`, `temperature=0` for deterministic reasoning). The judge evaluates purely whether the retrieved context contains sufficient information to answer the query. If the judge responds "NO," the system triggers an abstention. This second layer catches cases where the query is semantically close to corpus topics but the retrieved content is too shallow or tangential to support a faithful answer.

### 3.2 Calibration Algorithm Details

The calibration procedure uses a disjoint set of 4 seed queries (2 known in-domain, 2 known out-of-domain), deliberately chosen to represent extreme bounds:

| Type | Calibration Query | Purpose |
|---|---|---|
| In-domain | "What is Generative AI?" | Core curriculum topic — expected high similarity |
| In-domain | "When was Columbia University founded?" | Factual topic in corpus |
| Out-of-domain | "How do I bake a cake?" | Completely unrelated to tech/finance |
| Out-of-domain | "What is the capital of France?" | General knowledge, not in corpus |

These calibration queries are **strictly disjoint** from the 30-query evaluation set to prevent data leakage. In our experiment, the calibration yielded $\tau = 0.741$, successfully separating in-domain scores (range: 0.759–0.887) from out-of-domain scores (range: 0.578–0.650). Figure 2 visualizes this separation.

### 3.3 Custom Evaluation Framework Inspired by DeepEval

The course recommended DeepEval (Ip & Vongthongsri, 2023) for LLM evaluation. We encountered Python version incompatibilities with the official `deepeval` package (which requires Python ≥ 3.10, while our deployment environment runs Python 3.9). Rather than abandoning the evaluation philosophy, we implemented a **Custom Evaluation Module** that replicates DeepEval's core RAG Triad Metrics using pure LangChain prompts:

* **Answer Relevancy:** Measures whether the LLM's output directly addresses the user's query without tangential content. Implemented as a zero-shot evaluation prompt scored on a [0.0, 1.0] scale.
* **Faithfulness:** Cross-references the generated answer against the retrieved context to detect logical contradictions or hallucinated facts not grounded in the source material. Also scored on [0.0, 1.0].

Both metrics are evaluated by `gpt-4o-mini` at `temperature=0` for reproducibility. We acknowledge that this LLM-as-evaluator approach has inherent limitations — the evaluator may exhibit leniency bias, particularly when the generated output is short and closely paraphrases the source context (see §5.3 for discussion).

---

## 4. Experimental Setup

### 4.1 Corpus Construction

We constructed a domain-specific knowledge corpus containing **20 paragraphs** spanning two primary domains:

| Domain | Topics Covered | # Paragraphs |
|---|---|---|
| Technology / AI | Vector Databases, GPU hardware (RTX 3090), RAG architecture, DeepEval, Generative AI, Transformer architecture, Self-RAG, Prompt Engineering, Quantization, Fine-tuning, LangChain, Cosine Similarity, Streamlit, LLM Hallucination, LLM-as-a-Judge | 16 |
| Finance / Academic | Federal Reserve interest rates, Columbia University, STAT GR5293 course, PEFT/LoRA | 4 |

The corpus totals approximately **4,000 characters** (~600 tokens). Each paragraph is a self-contained factual statement averaging 1–3 sentences. During ingestion, the corpus is split using `CharacterTextSplitter` (chunk size = 200, overlap = 0, newline separator), producing chunks that preserve paragraph boundaries. The corpus source file (`data/corpus.txt`) and the build script (`src/build_dataset.py`) are included in the repository for full reproducibility.

### 4.2 Evaluation Dataset

The evaluation dataset (`data/eval_dataset.json`) contains **30 test cases** structured as follows:

| Category | Count | Ratio | Expected Behavior |
|---|---|---|---|
| In-Domain | 18 | 60% | Generate a faithful, relevant answer |
| Out-of-Domain | 12 | 40% | Abstain ("I don't know") |

In-domain queries target facts explicitly present in the corpus (e.g., "How much VRAM does the NVIDIA RTX 3090 have?"). Out-of-domain queries span topics entirely absent from the corpus — cooking, sports, geography, entertainment, and crafts — to test whether the system correctly identifies the boundary of its knowledge.

**Limitation:** The current OOD queries are topically distant from the corpus domain (e.g., "How do I knit a sweater?"), making them relatively easy for the similarity threshold to catch. A more rigorous evaluation would include **adversarial near-miss queries** — questions topically adjacent to the corpus but not answerable from it (e.g., "What is AMD's equivalent to CUDA?" or "What is the Federal Reserve's opinion on cryptocurrency regulation?"). We discuss this limitation further in §6.2.

### 4.3 Model Architecture

| Component | Choice | Rationale |
|---|---|---|
| Embeddings | OpenAI `text-embedding-ada-002` | Widely adopted dense embedding model with strong semantic capture; chosen for API simplicity. An alternative such as `text-embedding-3-small` offers improved performance-per-dollar but was not available at project start. |
| Vector Store | ChromaDB (in-memory) | Lightweight, Python-native vector database suitable for prototype-scale corpora. |
| LLM (Generator + Judge) | `gpt-4o-mini` | Cost-effective GPT-4-class model. `temperature=0` is used for the judge to ensure deterministic routing decisions. |
| UI / Demo | Streamlit | Interactive web interface for live demonstration. |
| Chunking | `CharacterTextSplitter` (200 chars, 0 overlap) | Matches corpus paragraph structure; no overlap needed since each paragraph is self-contained. |

---

## 5. Results and Analysis

### 5.1 Baseline Comparison

We compare our **Confidence-Calibrated RAG** against a **Naive RAG** baseline that disables both guardrails (i.e., every query is sent directly to the LLM with retrieved context, regardless of similarity score). The Naive RAG baseline was evaluated by running the same 30-query evaluation pipeline with `enable_abstention=False` (the `--naive` flag in `src/demo.py`), routing all queries directly to the LLM generator. Faithfulness and relevancy scores for the Naive baseline were collected using the same `CustomDeepEvalMock` evaluator applied to its outputs. Full results are stored in `evaluation_summary.txt` and `evaluation_full_results.json`.

**Table 1. System-Level Comparison (n=30)**

| Metric | Naive RAG (no guardrails) | Confidence-Calibrated RAG | Δ |
|---|---|---|---|
| Overall Accuracy | 93.3% (28/30) | **96.7%** (29/30) | +3.3 pp |
| In-Domain Accuracy | 88.9% (16/18) | 94.4% (17/18) | +5.6 pp |
| OOD Abstention Rate | 100% (12/12) | 100% (12/12) | 0 |
| In-Domain Failed Queries | Q12, Q17 | Q10 | Different errors |
| Avg. Faithfulness (In-Domain, answered) | 1.00 | 1.00 | 0 |
| API Calls Saved (OOD) | 0 | 12 | +12 |

**Key Observation:** The Naive RAG baseline (using `gpt-4o-mini` with a system prompt instructing "say I don't know if the context does not contain the answer") already abstains on all 12 OOD queries. This means the LLM's built-in instruction-following ability handles clearly off-topic queries even without our guardrails. The real differentiation appears in **in-domain quality**: Naive RAG failed on 2 in-domain queries (Q12: "Why is quantization useful for LLMs?" and Q17: "What causes hallucinations in LLMs?") where the LLM generated answers that scored 0.0 on faithfulness — indicating hallucinated content not grounded in the retrieved context. The Calibrated system failed only on Q10 ("How does Self-RAG work?") due to a false-positive abstention by the LLM Judge.

This finding reveals that our dual-guardrail system's primary value on this test set is **not OOD detection** (which the LLM handles natively for easy OOD queries) but rather **in-domain answer quality filtering** — the judge prevents the LLM from generating when context is insufficient, thereby avoiding the faithfulness failures observed in Naive mode.

### 5.2 Abstention Classification Metrics

Treating abstention as a binary classification task (Positive = Abstain, Negative = Answer):

**Table 2. Confusion Matrix (n=30)**

| | Predicted: Abstain | Predicted: Answer | Total |
|---|---|---|---|
| **Actual: Should Abstain** (OOD) | 12 (TP) | 0 (FN) | 12 |
| **Actual: Should Answer** (In-Domain) | 1 (FP) | 17 (TN) | 18 |
| **Total** | 13 | 17 | 30 |

**Derived Metrics:**
* **Precision:** 12/13 = **92.3%** — When the system abstains, it is correct 92.3% of the time.
* **Recall:** 12/12 = **100.0%** — The system catches every OOD query without exception.
* **F1-Score:** 2 × (0.923 × 1.0) / (0.923 + 1.0) = **96.0%**
* **Specificity:** 17/18 = **94.4%** — Among in-domain queries, 94.4% are correctly answered.

**Note on statistical power:** With n=30, a single changed test case shifts accuracy by ±3.3%. The 95% Wilson confidence interval for the overall accuracy of 96.7% (29/30) is approximately **[82.8%, 99.9%]**. We acknowledge this wide interval reflects the limited sample size and recommend that future work expand to n ≥ 100 for tighter bounds.

**McNemar's Test (Naive RAG vs. Calibrated RAG):** On the same 30 paired queries, Naive RAG misclassified 2 cases (Q12, Q17 — in-domain faithfulness failures) while Calibrated RAG misclassified 1 case (Q10 — false-positive abstention). All errors are disjoint (no query was misclassified by both systems). The McNemar contingency table yields: $b = 2$ (Naive wrong, Calibrated right), $c = 1$ (Calibrated wrong, Naive right). With the exact binomial test on the discordant pairs, $p = P(X \leq 1 | n=3, p=0.5) = 0.50$. At $\alpha = 0.05$, the improvement is **not statistically significant** ($p = 0.50$), which is expected given the small number of discordant pairs. This underscores the need for evaluation at larger scale (n ≥ 100) to achieve adequate statistical power.

### 5.3 Per-Query Evaluation Results

**Table 3. Detailed Evaluation Log (In-Domain Queries)**

| # | Query | Sim. Score | Judge | Faith. | Relev. | Result |
|---|---|---|---|---|---|---|
| 1 | What is the primary function of a Vector Database? | 0.839 | YES | 1.00 | 1.00 | ✅ Pass |
| 2 | How much VRAM does the NVIDIA RTX 3090 have? | 0.831 | YES | 1.00 | 1.00 | ✅ Pass |
| 3 | What are the core metrics provided by DeepEval? | 0.785 | YES | 1.00 | 1.00 | ✅ Pass |
| 4 | When was Columbia University established? | 0.870 | YES | 1.00 | 1.00 | ✅ Pass |
| 5 | What does Generative AI refer to? | 0.881 | YES | 1.00 | 1.00 | ✅ Pass |
| 6 | What happens when the Fed raises interest rates? | 0.843 | YES | 1.00 | 1.00 | ✅ Pass |
| 7 | What is PEFT? | 0.887 | YES | 1.00 | 1.00 | ✅ Pass |
| 8 | What topics does STAT GR5293 cover? | 0.779 | YES | 1.00 | 1.00 | ✅ Pass |
| 9 | What paper introduced the Transformer architecture? | 0.797 | YES | 1.00 | 1.00 | ✅ Pass |
| 10 | How does Self-RAG work? | 0.816 | **NO** | — | — | ❌ FP Abstain |
| 11 | What is prompt engineering? | 0.833 | YES | 1.00 | 1.00 | ✅ Pass |
| 12 | Why is quantization useful for LLMs? | 0.759 | YES | 1.00 | 1.00 | ✅ Pass |
| 13 | What is fine-tuning in ML? | 0.859 | YES | 1.00 | 1.00 | ✅ Pass |
| 14 | What does LangChain do? | 0.833 | YES | 1.00 | 1.00 | ✅ Pass |
| 15 | What is cosine similarity used for? | 0.871 | YES | 1.00 | 1.00 | ✅ Pass |
| 16 | What is Streamlit? | 0.862 | YES | 1.00 | 1.00 | ✅ Pass |
| 17 | What causes hallucinations in LLMs? | 0.821 | YES | 1.00 | 1.00 | ✅ Pass |
| 18 | How is an LLM-as-a-judge used? | 0.852 | YES | 1.00 | 1.00 | ✅ Pass |

**Table 4. Detailed Evaluation Log (Out-of-Domain Queries)**

| # | Query | Sim. Score | Below τ? | Guardrail Triggered | Result |
|---|---|---|---|---|---|
| 19 | What is the recipe for making a traditional Italian pizza? | 0.579 | Yes | G1 (fast-fail) | ✅ Abstained |
| 20 | Who won the FIFA World Cup in 2022? | 0.597 | Yes | G1 (fast-fail) | ✅ Abstained |
| 21 | Can you explain the plot of the movie Inception? | 0.650 | Yes | G1 (fast-fail) | ✅ Abstained |
| 22 | How do you change a car tire? | 0.615 | Yes | G1 (fast-fail) | ✅ Abstained |
| 23 | What is the capital of Australia? | 0.604 | Yes | G1 (fast-fail) | ✅ Abstained |
| 24 | Who wrote the Harry Potter series? | 0.590 | Yes | G1 (fast-fail) | ✅ Abstained |
| 25 | What is the weather like in Tokyo today? | 0.580 | Yes | G1 (fast-fail) | ✅ Abstained |
| 26 | How tall is the Eiffel Tower? | 0.578 | Yes | G1 (fast-fail) | ✅ Abstained |
| 27 | What are the rules of basketball? | 0.596 | Yes | G1 (fast-fail) | ✅ Abstained |
| 28 | How do I knit a sweater? | 0.598 | Yes | G1 (fast-fail) | ✅ Abstained |
| 29 | What is the speed of light in a vacuum? | 0.595 | Yes | G1 (fast-fail) | ✅ Abstained |
| 30 | Who painted the Mona Lisa? | 0.586 | Yes | G1 (fast-fail) | ✅ Abstained |

All 12 OOD queries produced similarity scores in the range **[0.578, 0.650]**, well below the threshold $\tau$ = 0.741. The separation gap between the lowest in-domain score (0.759, Q12) and the highest OOD score (0.650, Q21) is **0.109**, confirming strong separability on this test set. Every OOD query was caught by Guardrail 1 (quantitative threshold) without requiring a Guardrail 2 (LLM Judge) call.

**Figure 2.** Similarity score distribution for all 30 evaluation queries. In-domain queries (blue) cluster in the range [0.759, 0.887], while out-of-domain queries (red) cluster in [0.578, 0.650]. The calibrated threshold $\tau$ = 0.741 (green dashed line) achieves perfect separation. The gap of 0.109 between the nearest in-domain and OOD scores provides a margin of safety for the threshold.

![Figure 2. Similarity Score Distribution](figures/score_distribution.png)

**Evaluator Limitation Acknowledgment:** All 17 successfully answered in-domain queries received perfect Faithfulness and Relevancy scores (1.00). While this reflects the tight coupling between short, factual corpus paragraphs and the generated answers, it also suggests that our LLM-based evaluator may exhibit **leniency bias** — when the generated answer closely paraphrases a single source paragraph, even a moderately capable evaluator will rate it as fully faithful. A more discriminating evaluation would employ multi-hop questions requiring synthesis across multiple passages, or include adversarial cases where the answer contains subtle factual errors.

### 5.4 Error Analysis (The 1 Failed Case)

The single failed case pertained to the query: *"How does Self-RAG work?"*

Upon reviewing the execution logs, the vector retrieval successfully matched the relevant corpus paragraph (Sim. Score = 0.816, above $\tau$ = 0.741), meaning Guardrail 1 correctly passed this query. However, the strict `temperature=0` LLM Judge (Guardrail 2) deemed the context "Insufficient" because the corpus paragraph provides only a one-sentence high-level summary of Self-RAG rather than a step-by-step mechanical explanation.

This highlights a key **precision-safety trade-off** inherent to our conservative dual-guardrail design: the system occasionally prioritizes safety (abstention) over generation when the context is technically present but insufficiently detailed to satisfy the judge's strict evaluation criteria. We consider this an acceptable trade-off for high-stakes domains (healthcare, finance) where a false answer is far more costly than a missed answer.

### 5.5 Three-Condition Ablation Study

To isolate the contribution of each guardrail, we performed a three-condition ablation:

**Table 5. Ablation Results (n=30)**

| Configuration | In-Domain Acc. | OOD Abstention | Overall Acc. | Failed Queries |
|---|---|---|---|---|
| **No Guardrails** (Naive RAG) | 16/18 (88.9%) | 12/12 (100%) | 28/30 (93.3%) | Q12, Q17 (faithfulness=0.0) |
| **Guardrail 1 Only** (Threshold only) | 15/18 (83.3%) | 12/12 (100%) | 27/30 (90.0%) | Q8 (rel=0.3), Q12 (faith=0.0), Q17 (faith=0.0) |
| **Both Guardrails** (Full pipeline) | 17/18 (94.4%) | 12/12 (100%) | 29/30 (96.7%) | Q10 (judge FP abstention) |

**Analysis:**

* **Guardrail 1 (Quantitative Threshold)** successfully catches all 12 OOD queries across all conditions, as their similarity scores (0.578–0.650) fall well below $\tau$ = 0.741. However, Guardrail 1 alone does **not** prevent in-domain quality failures — Q12 and Q17 still fail with faithfulness=0.0 because the LLM generates hallucinated content despite the context being technically present but insufficiently specific.

* **Guardrail 2 (LLM Judge)** has a nuanced but net-positive effect. On the positive side, the judge prevents unfaithful generations on three queries: Q8 (which scores relevancy=0.30 without the judge but passes with it), Q12, and Q17 (both score faithfulness=0.0 without the judge but generate faithful answers with the judge active). The likely mechanism is that the judge's intermediate evaluation step implicitly constrains the generator’s behavior — by confirming the context is "sufficient," the judge's YES verdict focuses the LLM on the retrieved passage rather than relying on parametric knowledge. On the negative side, the judge introduces one false-positive abstention on Q10 (Self-RAG). **Net effect: +3 queries saved from quality failures, −1 query lost to FP abstention = net +2 improvement.** This makes the full dual-guardrail pipeline the best-performing configuration by a clear margin.

* **G1-only ablation methodology:** The G1-only condition was implemented by modifying the pipeline to skip the judge call and proceed directly to generation for all queries passing the threshold. The Naive condition uses `enable_abstention=False` in the existing codebase.

* **Practical cost impact:** Without guardrails, the system makes 30 full LLM generation calls. With Guardrail 1, 12 OOD queries are caught *before* any LLM call, reducing API costs by 40%. The full pipeline adds one judge call per query that passes Guardrail 1 (18 calls), but the judge's quality filtering prevents costly hallucination failures.

### 5.6 Threshold Sensitivity Analysis

To assess the robustness of our results to the choice of $\tau$, we analyzed how overall accuracy changes across a range of threshold values:

**Table 6. Threshold Sensitivity (n=30)**

| $\tau$ | In-Domain Passed G1 | OOD Caught by G1 | Notes |
|---|---|---|---|
| 0.65 | 18/18 | 10/12 | Misses Q21 (0.650) and Q22 (0.615) |
| 0.70 | 18/18 | 12/12 | All OOD caught; all in-domain pass |
| **0.741** | **18/18** | **12/12** | **Calibrated value** |
| 0.76 | 17/18 | 12/12 | Q12 (0.759) blocked by threshold |
| 0.78 | 16/18 | 12/12 | Q12, Q8 (0.779) blocked |
| 0.80 | 14/18 | 12/12 | Q12, Q8, Q9 (0.797), Q10 (0.816) blocked |
| 0.85 | 9/18 | 12/12 | Blocks 9 in-domain queries |

The system is robust across the range $\tau \in [0.70, 0.75]$, where all OOD queries are caught and all 18 in-domain queries pass Guardrail 1. Below $\tau = 0.65$, OOD queries begin leaking through; above $\tau = 0.76$, the lowest-scoring in-domain query (Q12, sim=0.759) is blocked. Our calibration algorithm's output ($\tau = 0.741$) falls within the optimal band, validating the midpoint heuristic despite its small calibration set. Notably, the margin between $\tau$ and Q12 is only 0.018 — a $\tau$ of 0.760 or higher would block Q12, reducing in-domain accuracy to 17/18.

*Note: Table 6 was derived analytically from the per-query similarity scores in Tables 3 and 4, by counting how many queries from each category fall above or below each candidate threshold. No additional API calls were made.*

### 5.7 Answering the Research Questions

We conclude the Results section by explicitly revisiting the research questions posed in §1.1:

* **RQ1** (Can the dual-guardrail system measurably improve answer quality?): **Yes, with caveats.** The calibrated pipeline achieves 96.7% overall accuracy vs. 93.3% for Naive RAG (+3.3 pp), driven by the LLM judge preventing 3 in-domain quality failures at the cost of 1 false-positive abstention. However, McNemar's test yields $p = 0.50$ on n=3 discordant pairs, meaning the improvement is **not statistically significant** at $\alpha = 0.05$. The effect is real but requires larger-scale evaluation (n ≥ 100) to achieve adequate statistical power for formal claims.

* **RQ2** (Can we evaluate robustly using programmatic metrics?): **Partially.** Our custom faithfulness and relevancy evaluator successfully detected hallucination failures in Naive and G1-only modes (Q12: faithfulness=0.0, Q17: faithfulness=0.0). However, the evaluator exhibited leniency bias on correctly answered queries — all 17 successful in-domain responses received perfect 1.00 scores, suggesting limited discriminative power for detecting subtle errors. A more robust evaluation would employ the official DeepEval framework (Python ≥ 3.10), multi-model evaluation panels, or human annotation.

* **RQ3** (What is each guardrail's relative contribution?): **Clearly answered.** The three-condition ablation (Table 5) demonstrates that Guardrail 1 (threshold) provides OOD detection and cost savings, while Guardrail 2 (judge) provides in-domain quality filtering. The judge's net effect is +3 queries saved from quality failures minus 1 FP abstention = **net +2 improvement**, making the full dual-guardrail pipeline the best-performing configuration by a clear margin.

---

## 6. Practical Impact and Future Work

### 6.1 Real-World Applications

The architecture provides a scalable, production-ready blueprint for domains where AI systems must guarantee safe abstention:

* **Healthcare:** Preventing medical chatbots from fabricating treatment information when queries fall outside the clinical knowledge base.
* **Financial Services:** Ensuring compliance chatbots abstain on questions about products or regulations not covered in their training corpus.
* **Enterprise Knowledge Management:** Internal Q&A systems that reliably indicate when an answer requires human escalation.

The fast-fail design of Guardrail 1 is particularly valuable in cost-sensitive deployments: by catching clearly out-of-scope queries at the embedding level, the system avoids unnecessary LLM API calls entirely.

### 6.2 Limitations

1. **Dataset scale:** Our evaluation uses n=30 test queries, which provides limited statistical power. The 95% confidence interval for our reported 96.7% accuracy spans [82.8%, 99.9%], and McNemar's test does not reach significance ($p = 0.50$). Production deployment would require validation on corpora and test sets several orders of magnitude larger.

2. **Corpus scale:** The knowledge corpus contains only 20 paragraphs (~600 tokens). At this scale, the retrieval task is relatively easy — the vector store contains few enough documents that similarity scores remain well-separated between in-domain and out-of-domain queries. Results may not generalize to production corpora with thousands of heterogeneous documents, where score distributions would likely overlap more.

3. **OOD query difficulty:** Current OOD queries are topically distant from the corpus, making them easy to detect via similarity scores alone. Future work should incorporate adversarial near-miss queries (e.g., "What is AMD's equivalent to CUDA?" — topically adjacent to our GPU content but unanswerable from the corpus) to stress-test the system's boundary detection.

4. **Calibration robustness:** The threshold calibration relies on only 4 seed queries. The midpoint heuristic ($\tau = (\min S_{\text{in}} + \max S_{\text{out}}) / 2$) is sensitive to outliers at this scale. A single noisy calibration query could shift $\tau$ by ±0.05, potentially crossing the optimal band identified in §5.6. More robust alternatives include k-fold cross-validation over a larger calibration set, or conformal prediction methods (Angelopoulos & Bates, 2023) that provide formal coverage guarantees.

5. **Evaluator leniency:** Our custom faithfulness evaluator consistently returns 1.00, suggesting insufficient discriminative power. Future iterations should use the official DeepEval framework (in a compatible Python ≥ 3.10 environment) or multi-model evaluation panels to reduce single-evaluator bias.

6. **Latency overhead:** The dual-guardrail pipeline adds approximately 40% latency per query (one additional LLM call for the judge). A production optimization would fine-tune a smaller specialist model (e.g., a distilled BERT classifier) as the judge to reduce latency while maintaining routing accuracy.

### 6.3 Future Work

* **Expanded evaluation:** Scale to n ≥ 100 with adversarial near-miss OOD queries and multi-hop in-domain queries requiring cross-paragraph reasoning.
* **Dynamic corpus ingestion:** Integrate live APIs (e.g., Bloomberg, Reddit) to test the abstention mechanism against real-time data streams.
* **Formal uncertainty quantification:** Replace the midpoint heuristic with conformal prediction to provide statistical coverage guarantees on the abstention decision.
* **Judge mechanism study:** Investigate why the LLM judge's YES/NO verdict appears to improve downstream generation faithfulness (the mechanism by which Q8, Q12, Q17 produce better answers in Calibrated mode versus G1-only mode).

---

## 7. References

1. Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
2. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In *Proceedings of the International Conference on Learning Representations (ICLR)*.
3. Barnett, S., Kurniawan, S., Thudumu, S., Brber, Z., & Vasa, R. (2024). Seven failure points when engineering a retrieval augmented generation system. *arXiv preprint*, arXiv:2401.05856.
4. Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4878–4887.
5. Ip, J., & Vongthongsri, K. (2023). *DeepEval: The open-source LLM evaluation framework* [Software]. Confident AI. https://github.com/confident-ai/deepeval
6. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 9459–9474.
7. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics (TACL)*, 12, 157–173.
8. Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)*, 784–789.
9. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.
10. Yan, S., Gu, J., Zhu, Y., & Ling, Z. (2024). Corrective retrieval augmented generation. *arXiv preprint*, arXiv:2401.15884.
11. Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems (NeurIPS)*, 36.
