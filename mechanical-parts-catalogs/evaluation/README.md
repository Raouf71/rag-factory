# RAG Evaluation Roadmap for Production

A structured approach to reliably evaluating Retrieval-Augmented Generation pipelines in production environments.

---

## 1. The need for evaluation
* RAG evaluation is best treated as a **three-layer problem**. <br>
The three layers are:
    - ``Retrieval quality`` — Is the right information even entering the pipeline?
    - ``Generation quality`` — Is the model using that information faithfully and correctly?
    - ``Production health`` — Is the system behaving reliably on real, unseen traffic?

    > Failures at each layer look different, require different fixes &rarr; Each layer must be evaluated independently. Collapsing them into a single score causes a so-called ``silent failure``.

    <details>
    <summary><ins>Illustrative example</ins>:</summary>

    * RAG pipeline answering: ``"What is the refund policy for orders over €100?"``

        * Retriever pulls a wrong document — the general FAQ page instead of the refund policy page &rarr; **Retrieval score: 0/10**.
        * But FAQ page mentions refunds briefly, so the generator produces a partially correct answer that sounds confident &rarr; **Generation score: 6/10**.
        * Entire pipeline **scores 5/10** overall:
            * gets interpreted as **"needs improvement"** rather than **"fundamentally broken"**
            * &rarr; undetected failure
            * &rarr; retriever stays broken

    </details>

---

## 2. Evaluation concept

### Layer 1 — Retrieval Quality

> ⚠️ The retriever is where RAG most often fails silently. 

* <ins>Approach</ins>:

    - Create a ``golden evaluation`` set of **query → expected source document pairs** to use for testing.

* <ins>Metrics</ins>:

    The key metrics here are:
<div align="center">

| Metric | What it measures |
|---|---|
| **Recall@K** | Does the correct document appear in the top K results? |
| **MRR (Mean Reciprocal Rank)** | How highly ranked is the first relevant result? |
| **Context Precision** | How much of what was retrieved was actually useful? |
| **Context Recall** | How much of what was needed was actually retrieved? |

</div>

> Tools like **[RAGAS](https://docs.ragas.io/)** automate this process and integrate cleanly into CI pipelines.

---

### Layer 2 — Generation Quality

> ⚠️ Hallucination tends to live in this layer

* <ins>Approach</ins>:
    1. Use An **[LLM-as-judge](https://mistral.ai/news/llm-as-rag-judge)** pattern
    2. A separate model scores faithfulness and relevance on a sample of outputs
    3. A small human-reviewed sample to calibrate the judge's scoring

    <details>
    <summary><ins>Illustrative example</ins>:</summary>

    * ``1,000 RAG outputs to evaluate.``

        * Hand-review and score 50 of outputs manually
        * Run the LLM judge on the same 50
        * Compare scores — spot where they diverge (If you gave a 2 and the judge gave a 4 — it's miscalibrated)
        *  Adjust the judge's prompt (``"be stricter when the answer adds details not present in the source"``) until scores align with the 50 outputs
        * Now trust the LLM judge to score the remaining 950.
    > Outline: 50 manual reviewed outputs = the calibration set

    </details>
<br>

* <ins>Metrics</ins>:

    The key metrics here are:
<div align="center">

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the claims in the answer grounded in the retrieved context? |
| **Answer Relevance** | Does the answer actually address the question asked? |
| **Groundedness** | Can each sentence in the answer be attributed to a source? |

</div>


> Tools like **RAGAS**, **[TruLens](https://github.com/truera/trulens)**, **[DeepEval](https://github.com/confident-ai/deepeval)** automate this process.

---

### Layer 3 — Production Health

> Failures not anticipated by offline evals become apparent in production.

* <ins>Approach</ins>:

    The live system should be instrumented for:
    - **User feedback signals**:
        * thumbs up/down
        * follow-up queries
        * abandonment rates
    - **No-answer rate**:
        * too high &rarr; retrieval gaps
        * too low &rarr; overconfidence
    - **Retrieval latency**: slow retrieval degrades UX and often signals index issues. Track all three values ``p50, p95 and p99``.
        <details>
        <summary><ins>Illustrative example</ins>:</summary>
        
        * ``100 users search the RAG pipeline and each query gets recorded``:
            * p50 = 120ms — 50% finished in under 120ms (typical user)
            * p95 = 800ms — 95% finished under 800ms, but 5 users waited almost a second 
            * p99 = 3000ms — 1 user waited 3 full seconds
        * The p50 looks fine, 
            * &rarr; averages are misguiding
            * p99 tells us someone is hitting a slow code path — maybe a bloated index, an unoptimized vector search, or a timeout on a large chunk retrieval.

        </details>
        
    - **Context utilization**:
        * retrieved chunks but never cited are candidates for pruning or re-chunking
    - **Query distribution drift**:
        * incoming queries shift away from what the index was built for &rarr; quality degrades silently without this signal.

> Recommended tooling: **LangSmith**, **Braintrust**, **Arize**, **Evidently**.

---

## 3. Failure Taxonomy

> Metrics identify what's wrong. A failure taxonomy identifies *why* &rarr; fast and systematic interation.

* Each failure type points to a different fix:

<div align="center">

| Failure | Root Cause | Recommended Fix |
|---|---|---|
| Wrong answer, correct docs retrieved | Generation | Improve prompt, add citation-forcing instructions, upgrade model |
| Correct answer, wrong docs retrieved | Retrieval (lucky) | Improve retrieval regardless — luck doesn't scale |
| No relevant docs retrieved | Retrieval | Re-embed corpus, adjust chunking strategy, add a reranker |
| Hallucination beyond retrieved context | Generation | Strengthen grounding instructions, enforce source attribution |
| Query misunderstood upstream | Query processing | Add HyDE, query rewriting, or intent-based routing |

</div>

> Outline: Failure taxonomy → clear root cause → targeted fix

---

## Summary

```perl
Layer 1 — Online Signals (real users, real time)
└── CTR, thumbs up/down, session abandonment, follow-up question rate

Layer 2 — Automated Metrics (continuous, async)
└── RAGAS faithfulness + answer relevancy on sampled queries

Layer 3 — LLM-as-Judge (on flagged/low-confidence outputs)
└── GPT-4o scoring groundedness + relevance on outliers only

Layer 4 — Human Eval (periodic, targeted)
└── Spot-check on failure clusters found in Layer 1–2
```

The most important principle: **never evaluate RAG as a black box.**
1. Decompose it
2. instrument it
3. categorize its failures.