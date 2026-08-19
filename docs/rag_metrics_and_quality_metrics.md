# RAG System Metrics and Quality Metrics

**Research date:** 2026-08-19
**Scope:** Text-based retrieval-augmented generation (RAG), from retrieval through answer generation, citations, reliability, and production operation.

## Executive summary

A RAG system cannot be represented faithfully by one score. It has at least four independently failing layers:

1. **Retrieval:** Did the system find the evidence?
2. **Generation:** Did it answer the question correctly and completely?
3. **Grounding and attribution:** Is every material claim supported by the supplied evidence, and do the citations point to that evidence?
4. **Operation:** Did it do so reliably, quickly, and at an acceptable cost?

The three maturity levels in this guide are a practical implementation taxonomy, not a formal standard:

- **Fundamentals** are deterministic or directly observed metrics that establish a trustworthy baseline.
- **Intermediate** metrics diagnose the interaction between retriever, generator, evidence, and citations.
- **Advanced** metrics evaluate claim-level failure modes, robustness, calibration, judge reliability, and quality-cost trade-offs.

The most important distinction is:

> **Faithfulness is not correctness.** A response can accurately repeat a false or outdated retrieved passage and therefore be faithful but incorrect. Conversely, a response can be factually correct from model memory but ungrounded in the retrieved evidence. Measure both.

## Measurement prerequisites

Before selecting metrics, create an evaluation record for every test query with as many of these fields as possible:

| Field | Why it is needed |
|---|---|
| Query and query ID | Stable unit of evaluation and regression tracking. |
| Answerability label | Separates answerable, unanswerable, ambiguous, and out-of-domain behavior. |
| Gold answer or atomic gold claims | Enables correctness, completeness, and claim-recall measurement. |
| Gold document/chunk IDs | Enables deterministic retrieval precision, recall, MRR, and nDCG. |
| Retrieved ranked chunks and scores | Enables retriever and ranking diagnosis. |
| Context actually sent to the model | Distinguishes retrieval from truncation or prompt-assembly failures. |
| Generated answer and citations | Enables answer, grounding, and citation evaluation. |
| Corpus/version timestamp | Enables reproducibility, freshness, and staleness analysis. |
| Language, domain, question type, and risk tier | Enables slice-level metrics instead of misleading global averages. |
| Retrieval, reranking, generation, and total timings | Enables stage-level latency diagnosis. |
| Model/provider, token usage, retries, fallbacks, and cost | Enables operational and economic evaluation. |

When complete gold passages are expensive to annotate, use gold answer claims as a proxy for evidence coverage, but label the metric accordingly. RAGAS explicitly supports claim-based context recall from a reference answer, while RAGChecker uses atomic claims to compare references, retrieved context, and responses ([RAGAS context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/), [RAGChecker tutorial](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md)).

---

## Level 1 — Fundamentals

Fundamental metrics should be implemented first because they are inexpensive, reproducible, and make obvious failures visible. They require a small gold test set for retrieval and answer metrics, plus production telemetry for operational metrics.

### 1. Retrieval Hit Rate@k (Success@k)

- **Definition:** The fraction of queries for which at least one relevant document or chunk appears in the top `k` results.
- **Formula:** `HitRate@k = (1/N) × Σ 1[relevant item exists in top k]`.
- **Role:** A coarse retrieval gate: can the generator see any usable evidence?
- **Use:** Track at several cutoffs such as `k = 1, 3, 5, 10`; use it during embedding, chunking, query-rewrite, and reranker comparisons.
- **Direction:** Higher is better.
- **Limitation:** One hit may be insufficient for multi-hop or multi-claim questions, and the metric ignores rank after the first hit.

### 2. Precision@k

- **Definition:** The fraction of the top `k` retrieved items that are relevant.
- **Formula:** `Precision@k = relevant items in top k / k`.
- **Role:** Measures retrieval focus and the amount of noise sent to the generator.
- **Use:** Optimize prompt-space efficiency and reduce distractors; compare different `k`, filters, and rerankers.
- **Direction:** Higher is better.
- **Limitation:** Requires relevance labels and does not penalize missing relevant evidence outside the top `k`.

### 3. Recall@k

- **Definition:** The fraction of all known relevant items retrieved in the top `k`.
- **Formula:** `Recall@k = relevant items in top k / all relevant items`.
- **Role:** Measures evidence coverage and the risk that the generator never receives required information.
- **Use:** Prioritize for multi-source, synthesis, legal, clinical, or other completeness-sensitive questions.
- **Direction:** Higher is better.
- **Limitation:** An incomplete relevance pool underestimates the system; increasing `k` can improve recall while worsening noise and latency.

### 4. Mean Reciprocal Rank (MRR)

- **Definition:** The mean inverse rank of the first relevant result.
- **Formula:** `MRR = (1/N) × Σ 1/rank(first relevant result)`; use zero when no relevant item is retrieved.
- **Role:** Measures how quickly the first useful item appears.
- **Use:** Best for questions usually answerable from one passage and for comparing ranking quality.
- **Direction:** Higher is better.
- **Limitation:** Ignores every relevant result after the first. NIST/TREC describes MRR as the average inverse rank of the first relevant suggestion ([TREC evaluation example](https://trec.nist.gov/pubs/trec25/papers/DUTH-CX.pdf)).

### 5. Mean Average Precision (MAP)

- **Definition:** Average Precision (AP) averages precision at each rank containing a relevant item; MAP averages AP over queries.
- **Role:** Measures whether *all* relevant binary-labeled results are ranked early.
- **Use:** Use when several relevant passages may exist and their order matters.
- **Direction:** Higher is better.
- **Limitation:** Treats relevance as binary and requires reasonably complete relevance judgments. TREC defines AP per topic from precision at relevant-document ranks and MAP as its mean across topics ([TREC ranked-list evaluation](https://trec.nist.gov/presentations/TREC9/overview/tsld014.htm)).

### 6. nDCG@k

- **Definition:** Normalized Discounted Cumulative Gain rewards high graded relevance near the top of a ranked list.
- **Formula:** `DCG@k = Σ(rel_i / log2(i + 1))`; `nDCG@k = DCG@k / ideal_DCG@k`.
- **Role:** Captures both ranking position and multiple relevance grades such as fully supporting, partially useful, and irrelevant.
- **Use:** Prefer it over Precision@k when relevance is graded or when highly supportive evidence must rank above merely related passages.
- **Direction:** Higher is better; normally 0–1.
- **Limitation:** The discount assumes rank-sensitive consumption. Recent work notes that an LLM may consume the whole context differently from a human scanning a result list, so nDCG may not predict downstream RAG quality by itself ([UDCG paper](https://aclanthology.org/2026.eacl-long.391/)).

### 7. Exact Match (EM)

- **Definition:** The fraction of answers that exactly equal an accepted reference after a documented normalization step.
- **Role:** A strict correctness metric for entities, dates, identifiers, or short factoid answers.
- **Use:** Use for closed-form tasks with unambiguous outputs; publish the normalization rules and accepted aliases.
- **Direction:** Higher is better.
- **Limitation:** Penalizes correct paraphrases and is inappropriate as the sole metric for explanatory answers. KILT uses EM for extractive and short abstractive tasks while using other measures for long answers ([KILT](https://aclanthology.org/2021.naacl-main.200/)).

### 8. Token Precision, Recall, and F1

- **Definition:** Lexical overlap between normalized answer tokens and reference tokens. Precision measures how much of the response matches the reference; recall measures how much of the reference is covered; F1 is their harmonic mean.
- **Formula:** `F1 = 2PR / (P + R)`.
- **Role:** Gives partial credit when short answers are not exact matches.
- **Use:** Pair with EM for extractive or short-answer QA.
- **Direction:** Higher is better.
- **Limitation:** Word overlap can miss semantic equivalence and reward unsupported copying.

### 9. End-to-end task accuracy or success rate

- **Definition:** The fraction of test cases that satisfy a task-specific pass condition, such as the correct value, valid structured output, or fully answered required fields.
- **Role:** Answers the business question: did the complete pipeline complete the requested task?
- **Use:** Define the pass rubric before testing and report results by answerability, language, domain, and question type.
- **Direction:** Higher is better.
- **Limitation:** A single success number cannot locate whether failure came from retrieval, generation, grounding, or formatting.

### 10. Latency percentiles

- **Definition:** Wall-clock duration distributions for retrieval, reranking, generation, and end-to-end response; report `p50`, `p95`, and `p99`, not only the mean.
- **Role:** Measures user experience and exposes tail latency.
- **Use:** Instrument each stage; separate time-to-first-token (TTFT) from total completion time for streaming systems.
- **Direction:** Lower is better under the required quality level.
- **Limitation:** Compare under the same load, region, hardware, cache state, corpus size, and output-length distribution. Retrieval research recommends reporting query latency and cost alongside accuracy ([Santhanam et al., 2023](https://aclanthology.org/2023.findings-acl.738/)).

### 11. Error rate and availability

- **Definition:** `Error rate = failed requests / total requests`; `Availability = successful service time or requests / total eligible service time or requests`.
- **Role:** Separates model quality from service reliability.
- **Use:** Track HTTP failures, timeouts, malformed outputs, retrieval failures, provider failures, and total user-visible failures separately.
- **Direction:** Lower error rate and higher availability are better.
- **Limitation:** A technically successful but incorrect response is not an operational error; quality failures need separate metrics.

### 12. Token usage and cost per request

- **Definition:** Input, retrieved-context, and output tokens and the attributable infrastructure/provider cost per request.
- **Role:** Measures resource efficiency and identifies oversized context or verbose generation.
- **Use:** Report mean and tail percentiles, then segment by route, model, query type, and success status.
- **Direction:** Lower is better only at matched quality.
- **Limitation:** Cheapest is not best if answer quality or coverage falls. OpenTelemetry defines GenAI duration and token-usage telemetry for model operations ([OpenTelemetry GenAI observability](https://opentelemetry.io/blog/2026/genai-observability/)).

---

## Level 2 — Intermediate

Intermediate metrics connect the modules. They answer *why* a RAG response succeeded or failed and normally require the exact retrieved context, a reference answer or claims, citations, and sometimes an NLI or LLM judge.

### 1. Context Precision

- **Definition:** Measures whether relevant retrieved chunks are ranked above irrelevant chunks. In RAGAS it is an average-precision-style score over the ranked context.
- **Role:** Diagnoses retriever/reranker focus and prompt contamination.
- **Use:** Compare chunking, hybrid search, metadata filters, reranking, and `k`; inspect queries with low score and high Recall@k to identify ranking/noise problems.
- **Inputs:** Query, ranked retrieved context, and either a reference answer/context or a judge.
- **Direction:** Higher is better.
- **Caution:** “Context precision” is not implemented identically by every framework. RAGChecker defines it as the proportion of relevant chunks, while RAGAS includes rank order ([RAGAS definition](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/), [RAGChecker definition](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md)). Record the implementation, judge, and version.

### 2. Context Recall / Claim Recall

- **Definition:** The fraction of reference evidence or gold-answer claims supported by the retrieved context.
- **Formula:** `supported reference claims / all reference claims`.
- **Role:** Measures whether retrieval supplied enough evidence to construct a complete answer.
- **Use:** Diagnose missing evidence in multi-claim and multi-hop questions; pair with Context Precision to tune `k` without hiding noise.
- **Inputs:** Retrieved context plus reference contexts or atomic reference claims.
- **Direction:** Higher is better.
- **Caution:** Claim recall based on a reference answer measures information coverage, not necessarily exact gold-document retrieval ([RAGAS context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)).

### 3. Faithfulness / Groundedness

- **Definition:** The fraction of claims in the generated response that are supported or entailed by the context actually supplied to the model.
- **Formula:** `supported response claims / all response claims`.
- **Role:** Detects unsupported generation and is the central hallucination-control metric for evidence-bound RAG.
- **Use:** Run at claim level; preserve the unsupported claims for debugging rather than storing only the aggregate score.
- **Inputs:** Generated response and the exact model context.
- **Direction:** Higher is better.
- **Caution:** It does not prove that the source is true, current, authoritative, or the best source. RAGAS defines faithfulness as factual consistency between response and retrieved context ([RAGAS faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)).

### 4. Answer Relevance

- **Definition:** How directly and completely the response addresses the user’s intent, independent of factual correctness.
- **Role:** Detects tangents, evasive responses, excessive irrelevant detail, and partially answered questions.
- **Use:** Score against the query using a rubric or a validated evaluator; inspect separately from correctness and faithfulness.
- **Inputs:** Query and response.
- **Direction:** Higher is better.
- **Caution:** A relevant answer can still be false. The RAGAS implementation reverse-generates questions from the answer and compares their embeddings with the original query, so its exact score depends on both evaluator and embedding models ([RAGAS answer relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)).

### 5. Answer Correctness (claim precision)

- **Definition:** The fraction of claims in the response that agree with the gold answer or verified ground truth.
- **Formula:** `correct response claims / all response claims`.
- **Role:** Measures factual accuracy, including errors that may be faithfully copied from faulty context.
- **Use:** Use atomic reference claims and retain contradiction/error labels for analysis.
- **Inputs:** Response and verified reference answer/claims.
- **Direction:** Higher is better.
- **Caution:** Do not call retrieval relevance or faithfulness “correctness.” They compare against different targets.

### 6. Answer Completeness (claim recall)

- **Definition:** The fraction of required gold-answer claims present in the response.
- **Formula:** `gold claims covered by response / all gold claims`.
- **Role:** Detects answers that are accurate but omit required facts.
- **Use:** Important for compound questions, procedures, comparisons, and safety-critical required disclosures.
- **Inputs:** Response and atomic gold claims.
- **Direction:** Higher is better.
- **Caution:** A verbose answer may raise completeness while lowering relevance or correctness; combine with claim precision and F1. RAGChecker calls these overall claim recall and precision ([RAGChecker metrics](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md)).

### 7. Answer claim F1

- **Definition:** Harmonic mean of claim-level answer correctness and completeness.
- **Role:** Provides an end-to-end factual balance when both invented and missing claims matter.
- **Use:** Use as a summary only after inspecting precision and recall separately.
- **Direction:** Higher is better.
- **Caution:** The score inherits errors from claim extraction and entailment judges.

### 8. Context Utilization

- **Definition:** The proportion of relevant information in the supplied context that the generator actually uses in its response.
- **Role:** Distinguishes “retrieval succeeded, generation ignored it” from retriever failure.
- **Use:** Compare prompts, context ordering, compression, and model choices when Context Recall is high but answer completeness is low.
- **Inputs:** Retrieved context, response, and relevance/claim judgments.
- **Direction:** Higher is generally better, subject to answer relevance and concision.
- **Caution:** Not every relevant detail should appear in every answer; define the denominator around evidence required for the query.

### 9. Citation Recall / Citation Completeness

- **Definition:** The fraction of externally verifiable or attribution-required answer claims that have at least one supporting citation.
- **Role:** Measures whether the answer is fully attributable and auditable.
- **Use:** Decompose the response into claims, determine which require citation, and test whether cited passages entail them.
- **Direction:** Higher is better.
- **Caution:** Merely placing a citation marker does not count; the cited evidence must support the claim.

### 10. Citation Precision / Citation Correctness

- **Definition:** The fraction of emitted citations that genuinely support the claim to which they are attached.
- **Role:** Detects decorative, irrelevant, or misplaced citations.
- **Use:** Evaluate citation-to-claim entailment and whether each citation contributes support; preserve bad citation examples.
- **Direction:** Higher is better.
- **Caution:** Citation precision and recall must be reported together. ALCE evaluates answer correctness and citation quality, including whether cited passages support the answer ([ALCE](https://aclanthology.org/2023.emnlp-main.398/)).

### 11. Provenance accuracy

- **Definition:** The fraction of citations whose document ID, page/section/span, and corpus version correctly resolve to the evidence displayed or used.
- **Role:** Validates traceability independently of semantic support.
- **Use:** Deterministically verify identifiers, URLs, page numbers, offsets, permissions, and source versions.
- **Direction:** Higher is better; high-risk systems should target 100% resolvability.
- **Caution:** A resolvable citation may still be irrelevant; pair with citation precision. KILT explicitly evaluates both downstream output and provenance in a shared knowledge source ([KILT](https://aclanthology.org/2021.naacl-main.200/)).

### 12. Negative rejection / Abstention quality

- **Definition:** Performance on unanswerable or out-of-domain queries. Useful measures include true rejection rate, false answer rate, abstention precision, and abstention recall.
- **Role:** Measures whether the system admits insufficient evidence instead of hallucinating.
- **Use:** Build a labeled mix of answerable, unanswerable, ambiguous, and adversarially related questions; tune the abstention threshold on validation data.
- **Direction:** Higher true rejection and lower false answer rate are better.
- **Caution:** Always report answer coverage alongside rejection quality; a system that refuses everything is safe-looking but useless. RGB includes negative rejection as a core RAG ability ([RGB](https://arxiv.org/abs/2309.01431)).

### 13. Human rubric score and preference win rate

- **Definition:** Expert or user ratings on explicit dimensions such as correctness, completeness, grounding, relevance, clarity, and actionability; pairwise win rate is the fraction of comparisons preferred over a baseline.
- **Role:** Provides the criterion validity that automated evaluators approximate.
- **Use:** Maintain a stratified human-audited set; randomize answer order, blind system identity, provide written rubrics, and report inter-annotator agreement.
- **Direction:** Higher is better.
- **Caution:** Human evaluation is not automatically unbiased; adjudication, domain expertise, and agreement reporting matter.

### 14. Stage latency, fallback rate, and cost per successful answer

- **Definition:** Retrieval/rerank/generation timing; provider retry or fallback frequency; and total cost divided by answers meeting the quality gate.
- **Role:** Makes reliability and efficiency trade-offs visible at the component level.
- **Use:** Correlate each stage with answer quality; alert on rising fallback rate even when availability remains high.
- **Direction:** Lower latency, fallback rate, and cost are better at matched success quality.
- **Caution:** Cost per raw request can improve by serving cheap failures. Use `cost / quality-passing answer` or a Pareto analysis.

---

## Level 3 — Advanced

Advanced metrics are appropriate after the basic scorecard is stable. They require controlled perturbations, fine-grained claims, calibrated evaluators, statistical analysis, or longitudinal production data.

### 1. Noise Sensitivity

- **Definition:** The fraction of response claims that become incorrect because the generator uses misleading information in relevant or irrelevant retrieved chunks.
- **Role:** Measures susceptibility to distractors rather than ordinary retrieval relevance.
- **Use:** Inject related-but-irrelevant, redundant, stale, and subtly conflicting chunks while holding the answerable evidence constant; report separate sensitivity to noise in relevant and irrelevant chunks.
- **Direction:** Lower is better.
- **Caution:** Keep perturbations realistic and preserve a clean control. RAGAS and RAGChecker both expose noise-sensitivity diagnostics ([RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/), [RAGChecker](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md)).

### 2. Hallucination rate and self-knowledge rate

- **Definition:** RAGChecker distinguishes response claims that are neither supported by context nor correct against the reference (**hallucination**) from correct claims that came from outside the supplied context (**self-knowledge**).
- **Role:** Separates harmful fabrication from useful but policy-violating external knowledge.
- **Use:** Apply when the system must be evidence-bound; inspect both rates alongside correctness and faithfulness.
- **Direction:** Lower hallucination is better. The desired self-knowledge rate depends on product policy.
- **Caution:** Attribution to “model memory” is inferred from comparisons, not observed causally.

### 3. Counterfactual Robustness

- **Definition:** The ability to resist or appropriately flag retrieved passages containing plausible but false facts that conflict with verified truth.
- **Role:** Tests whether retrieval grounding makes the system gullible to poisoned or erroneous sources.
- **Use:** Insert controlled counterfactual evidence, measure error acceptance/detection/correction, and stratify by source authority and conflict visibility.
- **Direction:** Higher detection/correction and lower counterfactual acceptance are better.
- **Caution:** This requires an authoritative truth set and careful safety controls. RGB evaluates counterfactual robustness as one of four core RAG abilities ([RGB](https://arxiv.org/abs/2309.01431)).

### 4. Multi-source Information Integration

- **Definition:** The rate at which the system correctly combines complementary evidence distributed across multiple passages or documents.
- **Role:** Tests synthesis beyond single-passage lookup.
- **Use:** Create multi-hop cases whose answer cannot be obtained from one chunk; measure final correctness, required-evidence coverage, and unsupported bridge claims.
- **Direction:** Higher is better.
- **Caution:** Recall@k can be high while integration fails; verify that every required evidence hop was retrieved and used. RGB treats information integration as a separate RAG capability ([RGB](https://arxiv.org/abs/2309.01431)).

### 5. Claim-level diagnostic matrix

- **Definition:** A matrix comparing response claims and gold claims against the response, retrieved context, and reference answer.
- **Role:** Localizes failure to missing retrieval, unused evidence, hallucination, self-knowledge, contradiction, or omission.
- **Use:** Store per-claim labels and aggregate into claim recall, context precision, context utilization, hallucination, self-knowledge, and faithfulness.
- **Direction:** Depends on the derived measure.
- **Caution:** Claim boundary and entailment errors propagate into all derived scores. RAGChecker formalizes this diagnostic approach and reports stronger human correlation than coarser metrics ([RAGChecker paper](https://arxiv.org/abs/2408.08067)).

### 6. Calibrated automated-judge score with confidence interval

- **Definition:** A context relevance, answer relevance, or faithfulness estimate produced by a domain-adapted judge and corrected using a small human-labeled set, with statistical uncertainty.
- **Role:** Enables scalable evaluation while preserving a quantified connection to human labels.
- **Use:** Validate on held-out human annotations, report confidence intervals, freeze judge/prompts during comparisons, and recalibrate after domain or model changes.
- **Direction:** Higher quality scores and narrower valid intervals are better.
- **Caution:** A raw LLM score is not calibrated evidence. ARES trains lightweight judges and uses prediction-powered inference with human annotations to estimate context relevance, answer faithfulness, and answer relevance with confidence intervals ([ARES](https://aclanthology.org/2024.naacl-long.20/)).

### 7. Judge validity, agreement, and bias

- **Definition:** Correlation or agreement between automated and human judgments, test-retest consistency, pair-order sensitivity, and subgroup bias.
- **Role:** Measures the quality of the evaluator itself before trusting its RAG scores.
- **Use:** Report Pearson/Spearman correlation for continuous ratings, accuracy/F1 for labels, Cohen/Fleiss kappa or Krippendorff alpha for agreement, and explicit order/verbosity/self-preference bias tests.
- **Direction:** Higher agreement and lower bias/sensitivity are better.
- **Caution:** G-Eval found stronger human alignment than older automatic metrics but also warned of bias toward LLM-generated text; later work shows superficial quality and verbosity can bias judges ([G-Eval](https://aclanthology.org/2023.emnlp-main.153/), [LLM-judge bias study](https://aclanthology.org/2024.ccl-1.101/)).

### 8. Confidence calibration: ECE and Brier score

- **Definition:** **Expected Calibration Error (ECE)** compares predicted confidence with observed accuracy across bins. **Brier score** is the mean squared error between predicted probability and binary outcome.
- **Role:** Determines whether a reported 80% confidence corresponds to roughly 80% correctness.
- **Use:** Calibrate on held-out data; report reliability diagrams and results by domain/language rather than only one aggregate.
- **Direction:** Lower is better.
- **Caution:** Open-ended answer correctness must first be mapped to a reliable outcome label; ECE is bin-sensitive.

### 9. Selective risk, coverage, and risk–coverage curve

- **Definition:** **Coverage** is the fraction of queries answered rather than abstained. **Selective risk** is the error rate among answered queries. The risk–coverage curve evaluates their trade-off across thresholds.
- **Role:** Finds a defensible abstention operating point for high-risk systems.
- **Use:** Choose thresholds on validation data to meet a maximum tolerated error; report the full curve and performance on rejected cases.
- **Direction:** Lower risk at a given coverage, or higher coverage at a fixed risk, is better.
- **Caution:** Never report selective accuracy without coverage.

### 10. Source conflict resolution accuracy

- **Definition:** The fraction of conflicting-source cases in which the system identifies the conflict, prefers the source selected by the evidence policy, and communicates residual uncertainty.
- **Role:** Evaluates authority, recency, and evidence-quality reasoning rather than simple entailment.
- **Use:** Build controlled pairs varying publication date, source authority, study quality, and contradiction type.
- **Direction:** Higher is better.
- **Caution:** The source-selection policy must be explicit and reviewed by domain experts.

### 11. Freshness and staleness metrics

- **Definition:** Measures such as corpus age, stale-hit rate, freshness-weighted accuracy, and time-to-index after a source update.
- **Role:** Detects answers that are well-grounded in outdated evidence.
- **Use:** Timestamp sources and queries, create time-sensitive evaluation cases, and measure whether newer authoritative evidence is retrieved and cited.
- **Direction:** Lower stale-hit/time-to-index and higher freshness-weighted accuracy are better.
- **Caution:** Newer does not always mean more authoritative; use a domain-specific source policy.

### 12. Slice disparity and worst-group quality

- **Definition:** The gap between overall performance and the lowest-performing language, domain, user group, query type, document format, or risk tier.
- **Role:** Prevents majority slices from hiding severe local failures.
- **Use:** Report every core metric by predeclared slice, worst-group score, max–min gap, and confidence interval; include English/Arabic parity where applicable.
- **Direction:** Higher worst-group quality and smaller unjustified gaps are better.
- **Caution:** Small slices produce unstable estimates; publish sample sizes and intervals.

### 13. Drift and regression rate

- **Definition:** Change in metric distributions over time or across corpus, embedding, reranker, prompt, model, or provider versions; regression rate is the fraction of previously passing cases that now fail.
- **Role:** Detects silent degradation after deployments or source updates.
- **Use:** Run a frozen regression set plus a refreshed current-data set; compare paired per-query deltas and alert on practical and statistical thresholds.
- **Direction:** Lower harmful drift and regression are better.
- **Caution:** A changing benchmark can imitate model drift; version all data and evaluators.

### 14. Quality–latency–cost Pareto frontier

- **Definition:** The set of configurations for which no other configuration is simultaneously better in quality, latency, and cost.
- **Role:** Replaces misleading single-metric optimization with an explicit deployment trade-off.
- **Use:** Evaluate combinations of `k`, reranker, context compression, model, provider, and fallback route under the same workload; discard dominated configurations.
- **Direction:** Prefer configurations on the frontier that satisfy product constraints.
- **Caution:** Do not combine dimensions into one weighted score unless stakeholders approve and sensitivity analysis shows the conclusion is stable. Efficiency-aware retrieval evaluation demonstrates that the best system changes with latency and cost budgets ([Santhanam et al., 2023](https://aclanthology.org/2023.findings-acl.738/)).

### 15. Utility- and distraction-aware retrieval gain

- **Definition:** A retrieval ranking measure that gives positive utility to passages that improve the final answer and negative utility to passages that distract the generator.
- **Role:** Aligns retrieval evaluation with an LLM consumer rather than a human browsing ranked results.
- **Use:** Use after traditional IR baselines when related-but-irrelevant passages measurably harm answer quality; annotate downstream utility under a fixed generator.
- **Direction:** Higher is better.
- **Caution:** Utility is model- and prompt-dependent, so it is less portable than relevance labels. UDCG is a recent research metric motivated by the weak alignment between classical ranking measures and RAG answer accuracy ([UDCG](https://aclanthology.org/2026.eacl-long.391/)).

---

## Recommended scorecards by maturity

### Fundamental launch scorecard

Use this before comparing advanced RAG architectures:

1. HitRate@1/3/5 and Recall@5.
2. Precision@5 or nDCG@5.
3. EM plus token F1 for short-answer tasks, or a task-specific pass rate.
4. End-to-end p50/p95/p99 latency.
5. Error rate, timeout rate, input/output tokens, and cost per request.
6. Results split by answerable/unanswerable, language, domain, and question type.

### Intermediate quality scorecard

Add when retrieved evidence and citations are available:

1. Context Precision and Context/Claim Recall.
2. Faithfulness, answer correctness, answer completeness, and answer relevance.
3. Citation precision, citation recall, and provenance accuracy.
4. Negative rejection and false answer rate.
5. Context utilization.
6. Human expert score on a stratified audit sample.
7. Stage latency, fallback rate, and cost per quality-passing answer.

### Advanced assurance scorecard

Add for mature or high-risk systems:

1. Noise sensitivity, counterfactual robustness, and source-conflict accuracy.
2. Multi-source integration and a claim-level diagnostic matrix.
3. Judge–human agreement, judge bias tests, and confidence intervals.
4. ECE/Brier calibration and risk–coverage curves.
5. Freshness/staleness, drift, regression, and worst-group quality.
6. Quality–latency–cost Pareto frontier.

## How to interpret metric combinations

| Observed pattern | Likely diagnosis | First action |
|---|---|---|
| Low Recall@k / claim recall, high faithfulness | Generator uses evidence correctly, but required evidence is missing. | Improve chunking, query rewriting, filters, embeddings, or increase candidate `k`. |
| High recall, low context precision, high noise sensitivity | Evidence is present but buried among distractors. | Add or improve reranking, deduplication, filtering, or context compression. |
| High context precision/recall, low context utilization | Retriever works; generator ignores available evidence. | Change prompt/context ordering or generation model. |
| High faithfulness, low correctness | The retrieved source is wrong, stale, or misinterpreted as authoritative. | Improve corpus governance, source policy, conflict handling, and freshness. |
| High correctness, low faithfulness | The model likely used outside knowledge or unsupported reasoning. | Enforce evidence-bound generation and abstention; improve citations. |
| High citation recall, low citation precision | Most claims have citation markers, but many citations do not support them. | Validate claim-to-citation entailment and citation placement. |
| High citation precision, low citation recall | Existing citations are good, but many claims are uncited. | Require citations for every material externally verifiable claim. |
| High offline quality, poor p95 latency/fallback rate | Quality configuration is operationally fragile. | Profile stages, warm dependencies, reduce context, and test provider routing. |
| Strong average, weak Arabic/domain slice | Aggregate hides a coverage or evaluator bias problem. | Expand gold data and validate retriever, generator, and judge per slice. |

## Evaluation protocol and reporting rules

1. **Freeze the unit of comparison.** Use the same corpus snapshot, test cases, relevance labels, model settings, and evaluator version for an experiment.
2. **Evaluate modules separately and end to end.** Retrieval scores alone do not prove answer quality; answer scores alone cannot localize failure.
3. **Report distributions and slices.** Include sample counts, p50/p95/p99, per-language/domain results, worst-group scores, and failure examples.
4. **Pair precision with recall.** This applies to retrieval, answer claims, citations, and abstention.
5. **Use confidence intervals.** Bootstrap paired per-query results for deterministic metrics; calibrate judge-based estimates against human labels.
6. **Keep evaluator lineage.** Record judge model, prompt, temperature, embedding model, NLI model, library version, and retry policy.
7. **Audit judges.** LLM judges are measurements with error and bias, not ground truth. Validate agreement, order sensitivity, verbosity bias, and domain/language behavior.
8. **Do not optimize against the test set.** Maintain development, validation, and untouched test sets; refresh a separate production-representative set over time.
9. **Preserve examples.** Aggregate scores without failed queries, unsupported claims, and bad citations are weak diagnostic evidence.
10. **Define gates before release.** Example: minimum claim recall and faithfulness, maximum false-answer rate for unanswerable queries, maximum p95 latency, and zero broken provenance links.

## Source map

The catalog above synthesizes definitions and practices from primary papers and official project documentation:

1. Es et al., **RAGAs: Automated Evaluation of Retrieval Augmented Generation**, EACL 2024 — reference-free evaluation across retrieval focus, faithfulness, and answer quality: <https://aclanthology.org/2024.eacl-demo.16/>
2. Official RAGAS metric documentation — current definitions and formulas for context precision/recall, answer relevancy, faithfulness, and noise sensitivity: <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
3. Saad-Falcon et al., **ARES**, NAACL 2024 — context relevance, answer faithfulness, answer relevance, calibrated judges, prediction-powered inference, and confidence intervals: <https://aclanthology.org/2024.naacl-long.20/>
4. Ru et al., **RAGChecker**, NeurIPS 2024 — claim-level overall, retriever, and generator diagnostics: <https://arxiv.org/abs/2408.08067>
5. Amazon Science, official **RAGChecker tutorial and metric definitions**: <https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md>
6. Gao et al., **ALCE: Enabling Large Language Models to Generate Text with Citations**, EMNLP 2023 — correctness and citation precision/recall: <https://aclanthology.org/2023.emnlp-main.398/>
7. Petroni et al., **KILT**, NAACL 2021 — downstream task metrics combined with knowledge-source provenance: <https://aclanthology.org/2021.naacl-main.200/>
8. Chen et al., **RGB: Benchmarking Large Language Models in Retrieval-Augmented Generation** — noise robustness, negative rejection, information integration, and counterfactual robustness: <https://arxiv.org/abs/2309.01431>
9. Liu et al., **G-Eval**, EMNLP 2023 — LLM-based evaluation, human correlation, and evaluator bias caution: <https://aclanthology.org/2023.emnlp-main.153/>
10. Zhou et al., **Mitigating the Bias of Large Language Model Evaluation**, CCL 2024 — superficial-quality and verbosity bias in LLM judges: <https://aclanthology.org/2024.ccl-1.101/>
11. Santhanam et al., **Moving Beyond Downstream Task Accuracy for Information Retrieval Benchmarking**, ACL Findings 2023 — latency and cost-aware retrieval evaluation: <https://aclanthology.org/2023.findings-acl.738/>
12. Trappolini et al., **Redefining Retrieval Evaluation in the Era of LLMs**, EACL 2026 — limitations of classical IR metrics for LLM consumers and UDCG: <https://aclanthology.org/2026.eacl-long.391/>
13. NIST/TREC materials for classical ranked retrieval measures: <https://trec.nist.gov/presentations/TREC9/overview/tsld014.htm> and <https://trec.nist.gov/pubs/trec25/papers/DUTH-CX.pdf>
14. OpenTelemetry, **GenAI observability** — standardized operation duration and token-usage telemetry: <https://opentelemetry.io/blog/2026/genai-observability/>

## Final recommendation

Do not start with every metric. Establish a small, versioned gold set and implement the Fundamental scorecard. Add the Intermediate metrics to locate retrieval, generation, grounding, and citation failures. Use the Advanced layer only when its additional annotation and evaluator complexity answers a concrete risk question. The release decision should be based on a multi-dimensional quality gate, not a weighted “RAG score” that can hide a critical failure behind a good average.
