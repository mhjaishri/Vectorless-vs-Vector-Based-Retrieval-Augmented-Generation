---
title: "Vectorless vs Vector-Based Retrieval-Augmented Generation: A Comparative Study for Legal Question Answering"
author:
  - "[Author Name]^[1]^"
  - "[Co-Author 1]^[1]^"
  - "[Co-Author 2]^[1]^"
affiliation:
  - "1: [Department / Institution / Email]"
abstract: |
  Legal question answering is a high-stakes application of large language models (LLMs)
  in which grounded, auditable, and citation-backed answers are essential. Retrieval-Augmented
  Generation (RAG) is the prevailing architecture for such systems, but the design space spans
  two markedly different families: traditional vector-based pipelines that rely on dense embeddings
  and similarity search, and the recently proposed *vectorless* approach (PageIndex, 2025) that
  navigates a hierarchical document tree using an LLM agent without any embeddings or vector store.
  This paper presents a controlled head-to-head comparison of the two paradigms on the Constitution
  of India. We build two pipelines that share an identical answer-generation backbone and evaluate
  them on more than 200 question–answer pairs using six complementary metrics — context precision,
  context recall, hit rate, faithfulness, answer correctness, and semantic similarity. The results
  reveal a clean trade-off: the vectorless tree-based system dominates every retrieval-quality
  metric, while the vector-and-graph pipeline dominates every answer-quality metric. Our analysis
  attributes this split to the structural alignment between the Constitution's hierarchy and the
  PageIndex tree, and to the broader contextual envelope provided by top-k cosine retrieval. The
  findings argue that legal RAG systems should be evaluated on both axes jointly and motivate
  hybrid pipelines that combine vectorless precision with graph-augmented recall.
keywords:
  - Legal Question Answering
  - Retrieval-Augmented Generation
  - Vectorless RAG
  - PageIndex
  - Knowledge Graph
  - LLM Evaluation
---

# 1 Introduction {#introduction}

Legal information systems are among the most demanding applications of natural language processing.
Statutes are written in dense, cross-referencing prose; a single article often inherits meaning
from earlier definitions, depends on later schedules, and is amended by clauses scattered across
the document. Practitioners therefore expect any artificial-intelligence assistant in the legal
domain to do more than produce fluent answers — they require *grounded* answers that can be
verified against authoritative source material, with a clear audit trail back to the article,
clause, or sub-clause that supports each statement [@paul2023inlegalbert].

Retrieval-Augmented Generation (RAG) [@lewis2020rag] has become the dominant architecture for
this class of problem. By inserting a retrieval step between the user's question and the language
model, RAG grounds the generator in a small set of relevant passages, reducing parametric
hallucination and enabling source citation. The retrieval step is typically realised through dense
vector embeddings and approximate nearest-neighbour search [@karpukhin2020dpr], optionally
augmented with a knowledge graph that captures explicit cross-references between entities or
articles [@edge2024graphrag]. We refer to this family collectively as **vector-based RAG**.

In September 2025 a different approach gained attention under the name **PageIndex** — a
*vectorless* RAG technique that abandons embeddings and similarity search entirely. Instead,
the document is represented as a hierarchical tree (parts, chapters, articles, clauses) that
mirrors the table of contents. At query time, an LLM agent navigates this tree through
tool-calling steps, descending into branches whose titles or summaries appear relevant and
gathering the leaf passages it needs to compose an answer. Because no embeddings exist, retrieval
is fully deterministic in its addressing scheme and the document's logical structure is preserved
intact — both attractive properties for legal text, where hierarchy is meaningful.

These two families embody opposite design philosophies. Vector-based RAG treats the document as a
flat soup of chunks and relies on learned representations to surface relevant ones; vectorless
RAG treats the document as a structured artefact and relies on its existing organisation to guide
retrieval. Whether one or the other is preferable for legal question answering — and which
qualities of the answer they each preserve or sacrifice — is an open empirical question.

This paper provides a direct, controlled answer. We implement two pipelines on the same legal
corpus (the Constitution of India), with the same answer-generation model (`gpt-4o-mini` at
temperature zero) and the same prompts, varying only the retrieval philosophy. We evaluate the
two systems on more than 200 question–answer pairs using six complementary metrics drawn from
the recent RAG-evaluation literature [@es2024ragas; @saadfalcon2024ares; @adlakha2024evaluating].
Across the full evaluation, we observe a sharp three-versus-three split: the vectorless system
wins every retrieval-quality metric (precision, recall, hit rate), while the vector-plus-graph
system wins every answer-quality metric (faithfulness, correctness, semantic similarity).

Our contributions are threefold. First, we present a head-to-head comparison of vectorless and
vector-based RAG on a real legal corpus, with the generator held constant so that observed
differences are attributable to the retrieval architecture. Second, we introduce a six-metric
evaluation methodology that combines deterministic signals (token-level F1, embedding cosine,
ground-truth-source matching) with LLM-judged signals (claim-decomposition faithfulness,
structured-JSON correctness), reducing reliance on any single subjective measure. Third, we
articulate a clear architectural trade-off: vectorless retrieval offers superior source fidelity,
while vector-and-graph retrieval offers superior answer quality, suggesting that legal practitioners
should choose between the two based on whether citation accuracy or answer fluency is the dominant
requirement of their use case.

# 2 An Overview of RAG Approaches for Legal AI {#background}

Retrieval-Augmented Generation pipelines for legal text fall along a spectrum of how heavily they
rely on learned vector representations. This section surveys the spectrum and contrasts the two
endpoints that we compare empirically in later sections.

## 2.1 Traditional Vector-Based RAG

A traditional RAG pipeline begins with an offline ingestion stage in which the source corpus is
divided into chunks of bounded length, each chunk is embedded into a fixed-dimensional vector by a
neural encoder, and the resulting vectors are stored in an approximate-nearest-neighbour index.
At query time, the user question is embedded with the same encoder; the top-*k* most similar
chunks are retrieved by cosine distance and concatenated into the prompt that conditions answer
generation [@karpukhin2020dpr]. This design has two properties that make it convenient: the
retrieval step is independent of the LLM, and indexing scales linearly with corpus size.

For legal documents, however, the chunking step poses a structural problem. Legal text is rich in
cross-references — a clause in Article 32 of the Indian Constitution refers back to the
fundamental rights enumerated in Articles 12 to 30, while Article 368 invokes provisions about
amendment procedure that span several earlier parts. Naïve chunking by token length severs these
references; embedding-based retrieval may then surface a clause about *enforcement* of fundamental
rights without surfacing the rights themselves. Hyde-style query rewriting [@gao2023hyde] partially
mitigates the problem but does not eliminate it.

## 2.2 Knowledge-Graph Augmentation

A common remedy is to overlay a knowledge graph on top of the vector index. In our implementation,
each Constitution article becomes a node and each explicit reference (e.g. "subject to the
provisions of Article 19") becomes a directed `REFERS_TO` edge in a Neo4j graph. After the cosine
retrieval step returns its top-*k* chunks, the pipeline expands the result set with one or two
hops along these edges, restoring the cross-references that chunking would have destroyed. The
graph also provides a deterministic fallback when embedding similarity fails to surface a
known-related article. Variations of this idea appear in GraphRAG [@edge2024graphrag] and
related work that combines structured and unstructured retrieval.

## 2.3 Vectorless RAG via PageIndex

The vectorless approach (PageIndex, 2025 [@pageindex2025]) removes the embedding step entirely.
During ingestion the document is parsed into a hierarchical tree whose structure mirrors its
logical organisation: parts contain chapters, chapters contain articles, articles contain clauses
and sub-clauses. Each non-leaf node carries a short summary; each leaf node stores the raw text.
At query time, the system uses an LLM as a navigation agent: starting at the root, the agent
inspects child summaries, chooses a subtree to descend, and either returns the leaf content it
finds or recurses into another sub-tree. Retrieval terminates when the agent declares that it has
gathered sufficient evidence or a maximum hop budget is reached.

Because the retrieval surface is the document's own table of contents, vectorless RAG preserves
hierarchy by construction. There is no chunk-boundary loss: the leaf node of *Article 21* is
exactly the article and only the article. Cross-references can still be missed if the agent does
not navigate to a related article, but the addressing of any returned passage is canonical and
auditable. The trade-off is computational: each retrieval call invokes the LLM, and a single
multi-hop trace may consume two or three LLM calls before the answer model is even invoked.

# 3 Related Works {#related}

A growing body of literature studies the evaluation and improvement of RAG pipelines. Lewis et al.
[@lewis2020rag] established the basic encoder–decoder architecture; Karpukhin et al.
[@karpukhin2020dpr] introduced dense passage retrieval as a learned alternative to BM25.
Frameworks for automated RAG evaluation include RAGAS [@es2024ragas], which proposes
LLM-judged metrics for context precision, faithfulness, and answer relevance, and ARES
[@saadfalcon2024ares], which fine-tunes lightweight judges from synthetic question–answer pairs.
Liu et al. [@liu2023geval] showed that GPT-4 itself can serve as a reliable evaluator when prompted
with structured rubrics, motivating the LLM-as-judge methodology we adopt for several of our
metrics. Adlakha et al. [@adlakha2024evaluating] specifically demonstrated that
instruction-following LLMs frequently produce answers that are correct in isolation but
unsupported by retrieved context, while Mallen et al. [@mallen2023whennottrust] identified
parametric memory of popular entities as a contributor to that gap. Edge et al. [@edge2024graphrag]
proposed a graph-based retrieval variant that motivates the knowledge-graph layer of our
vector-based pipeline. In the legal domain, InLegalBERT [@paul2023inlegalbert] illustrates the
value of domain-adapted encoders for Indian legal text. Most of these works study retrieval
quality and answer quality independently, or evaluate a single architecture in isolation; we are
aware of no prior work that contrasts the *retrieval philosophy* — vectorless tree-navigation
versus vector-plus-graph similarity search — on a single legal corpus with a controlled
generator. Our study fills that gap.

# 4 A Case Study on the Indian Constitution {#case-study}

This section describes the corpus, the question–answer dataset, and the two RAG pipelines that
form the case study. The Indian Constitution is a particularly suitable test corpus: it is long
(over 450 articles in 25 parts), highly hierarchical, and densely cross-referenced, exposing each
retrieval philosophy to its characteristic strengths and weaknesses.

## 4.1 Dataset

The corpus consists of the full text of the Constitution of India in machine-readable form, with
its hierarchy of parts, chapters, articles, and clauses preserved. The evaluation dataset contains
more than 200 question–answer pairs, drawn from real interpretive questions about constitutional
provisions. Each pair records the question text, a reference answer written by a domain
annotator, and a set of *ground-truth source labels* identifying the article(s) on which the
correct answer must be based. Roughly three-quarters of the questions have a single ground-truth
article (e.g. "What does Article 21 guarantee?"), while the remainder require multiple articles
(e.g. "Who appoints the Chief Justice of India?" — Articles 124 and 126). This mix exercises
both the retrieval recall capacity of each system and its ability to combine evidence across
related provisions.

## 4.2 System A: Vector-and-Graph RAG (Traditional)

The traditional pipeline ingests the Constitution by chunking each article into self-contained
passages and embedding them with the open-source `nomic-embed-text` model (768 dimensions).
Vectors are stored in a Qdrant collection of 1,677 points using cosine distance. At query time,
the user question is embedded with the same model and the top-10 most similar chunks are
retrieved. The retrieved chunks are then augmented through one hop along `REFERS_TO` edges in a
Neo4j graph that encodes inter-article references mined during ingestion. Finally, the union of
cosine-retrieved and graph-expanded chunks is concatenated into a prompt that asks `gpt-4o-mini`
(temperature 0) to compose the answer. This architecture is summarised on the left side of
Figure 1.

## 4.3 System B: Vectorless RAG via PageIndex

The vectorless pipeline ingests the Constitution into a pre-built hierarchical tree. Each tree
node carries a brief summary; leaves carry article-level text. At query time, an LLM navigation
agent starts at the root, inspects summaries of children, and descends through up to three hops
(`MAX_HOPS = 3`), emitting tool calls to read leaf content when it identifies relevant
articles. The agent terminates by invoking a `done` tool, after which the same `gpt-4o-mini`
model — using the same answer prompt as System A — composes the final response from the
collected passages. This architecture is summarised on the right side of Figure 1. No
embeddings, vector indexes, or graph databases are involved in this pipeline; the only
infrastructure required is the pre-built JSON tree.

![Architecture overview of the two RAG pipelines under comparison: vector-and-graph
(left) versus vectorless tree navigation (right). Both share an identical
`gpt-4o-mini` answer-generation backbone.](figures/architecture_diagram.png){width=90%}

To make the comparison meaningful, every other variable is held constant. Both systems use the
same answer model, the same answer-generation prompt, the same temperature, the same evaluation
dataset, and the same metric definitions. Differences in evaluation scores can therefore be
attributed to the retrieval architecture rather than to confounding factors in the generator
or in the prompt design.

# 5 Evaluation Methodology and Results {#evaluation}

This section describes the metrics, the judge configuration, and the results obtained over the
full evaluation set.

## 5.1 Metrics

We evaluate each system along six complementary axes, three of which target retrieval quality and
three of which target answer quality.

**Context Precision.** For each question, every retrieved chunk is independently judged by an LLM
as relevant (YES) or not (NO) with respect to the question. Context precision is the fraction of
retrieved chunks judged relevant. This metric penalises retrieval pipelines that surface noisy or
off-topic passages.

**Context Recall.** This metric is computed deterministically. For each question, we check how
many of the ground-truth source labels appear, by exact match or by article-number boundary, in
the concatenated retrieved text. Context recall is the fraction of ground-truth sources that are
recovered. Because the metric is rule-based, it does not depend on judge subjectivity.

**Hit Rate.** A binary deterministic signal: 1 if any ground-truth source is found in the
retrieved chunks, 0 otherwise. This summarises whether the pipeline is *able* to find the
authoritative passage at all, regardless of how many ground-truth labels it recovers.

**Faithfulness.** The generated answer is decomposed into atomic factual claims by a first LLM
call; each claim is then independently verified against the retrieved context by a second LLM
call. Faithfulness is the fraction of claims judged supported [@es2024ragas;
@adlakha2024evaluating]. The metric captures *grounding* — whether the answer is derived from
the retrieved chunks rather than from the model's parametric memory.

**Answer Correctness.** A composite score `0.4 · token_F1 + 0.6 · L`, where token_F1 is the
unigram F1 between the generated answer and the reference answer, and `L` is an LLM-judged score
that returns a JSON object with two scalar fields: factual *correctness* and *completeness*
relative to the reference. This blend balances a deterministic surface-overlap signal with a
holistic semantic judgment.

**Semantic Similarity.** Cosine similarity between Ollama `nomic-embed-text` embeddings of the
generated and reference answers. This deterministic signal complements the previous LLM-judged
metrics with a fully reproducible distance measure.

## 5.2 Judge and Reproducibility

All LLM-judged metrics use OpenAI `gpt-4o-mini` at temperature 0. Each judge prompt is hashed by
SHA-256 and the verdict is cached on disk, so re-runs are deterministic and inexpensive. Judge
calls are rate-limited to 0.15 seconds between requests with exponential back-off on transient
errors. The full evaluation of more than 200 questions across both systems is resumable: the
runner persists per-question results to disk after every completed question, so an interrupted
run can be continued without recomputation.

## 5.3 Aggregate Results

Table 1 reports the mean of each metric over the full evaluation set for both pipelines, together
with the winning system on each axis.

**Table 1.** Aggregate evaluation metrics across the full evaluation set (more than 200
question–answer pairs). Bold indicates the higher score on each row.

| Metric                | PageIndex (vectorless) | Vector+Graph (traditional) | Winner       |
|-----------------------|------------------------|----------------------------|--------------|
| Context Precision     | **0.7971**             | 0.6813                     | PageIndex    |
| Context Recall        | **0.8430**             | 0.6083                     | PageIndex    |
| Hit Rate              | **0.8636**             | 0.6817                     | PageIndex    |
| Faithfulness          | 0.6527                 | **0.7331**                 | Vector+Graph |
| Answer Correctness    | 0.6849                 | **0.7676**                 | Vector+Graph |
| Semantic Similarity   | 0.8065                 | **0.8647**                 | Vector+Graph |

Figures 2 and 3 visualise the same data in bar-chart and radar-chart form respectively, while
Figure 4 reproduces the summary table as rendered by the evaluation pipeline.

![Comparison of all six evaluation metrics for the two systems. PageIndex (vectorless)
dominates the three retrieval-quality metrics on the left, while Vector+Graph
(traditional) dominates the three answer-quality metrics on the right.](figures/comparison_bar.png){width=85%}

![Radar view of the same six-metric evaluation, showing the complementary
strengths of the two architectures.](figures/comparison_radar.png){width=70%}

![Tabular summary of aggregate metrics, generated by the evaluation
pipeline.](figures/summary_table.png){width=80%}

The pattern is unambiguous. The vectorless system wins every retrieval-quality metric, in some
cases by a considerable margin: context recall is higher by 23.5 percentage points
(0.8430 versus 0.6083), and hit rate by 18.2 percentage points (0.8636 versus 0.6817).
Conversely, the vector-and-graph system wins every answer-quality metric: faithfulness is higher
by 8.0 percentage points (0.7331 versus 0.6527), answer correctness by 8.3 percentage points
(0.7676 versus 0.6849), and semantic similarity by 5.8 percentage points (0.8647 versus 0.8065).
Neither system wins on both axes; both systems display a clearly dominant strength.

## 5.4 Discussion

Two structural factors plausibly explain the observed split. On the retrieval side, the
Constitution is hierarchical by design — its parts, chapters, articles, and clauses form a tree
whose internal nodes carry meaningful semantic labels (e.g. "Fundamental Rights",
"Directive Principles"). The PageIndex tree mirrors this hierarchy directly; navigating from the
root to a leaf article is therefore a sequence of choices over genuinely informative summaries.
By contrast, the vector pipeline must learn to map a question to article-relevant chunks through
embedding similarity alone, with no awareness of the document's logical structure. When the
question is phrased in terms close to the article's title, both systems perform well; when the
question is phrased in functional terms (e.g. "Who has the power to make laws on subjects in the
Concurrent List?"), the tree-navigation agent benefits more from explicit structural cues than the
embedding model does from learned similarity. This advantage is reflected in PageIndex's higher
context precision, recall, and hit rate.

On the answer side, the trade-off reverses. The vector-and-graph pipeline retrieves a fixed top-10
set of chunks and expands it by one graph hop, providing a wide and somewhat redundant context
window of typically 12 to 15 passages. PageIndex returns only the leaf passages explicitly visited
by the agent, which is a smaller and more focused set (an average of 3 to 4 chunks per question
in our run). Although focused retrieval improves precision, it also gives the answer model less
material to triangulate from. With the broader vector-and-graph context, `gpt-4o-mini` can stitch
together facts across multiple loosely-related passages and produce answers that score higher on
faithfulness, correctness, and semantic similarity. In essence, the vectorless pipeline trades
breadth of context for purity, while the vector-based pipeline trades purity for breadth — and
the answer model performs slightly better with breadth on this dataset. We note that this
breadth–purity trade-off is itself a function of hyperparameter choices (top-*k* in the vector
pipeline, `MAX_HOPS` in the tree pipeline); future work should sweep these parameters to map the
Pareto frontier between the two regimes.

# 6 Conclusion {#conclusion}

This paper has presented a controlled comparison of vectorless and vector-based
Retrieval-Augmented Generation in the legal-question-answering domain. Using the Constitution of
India as the corpus and an evaluation set of more than 200 question–answer pairs, we measured
both systems on six complementary metrics that span retrieval quality and answer quality. The
results yield a clean architectural trade-off: the vectorless tree-navigation pipeline (PageIndex,
2025) wins every retrieval-quality metric, while the traditional vector-and-graph pipeline wins
every answer-quality metric.

For practitioners building legal AI systems, the practical implication is that the choice of
retrieval philosophy should be driven by which form of correctness matters more for the
deployment in question. When the system must produce verifiable citations to authoritative
provisions — for example in audit, compliance, or research workflows — the vectorless approach
offers materially higher source-fidelity. When the priority is fluent, complete answers to
multi-article questions and citation accuracy is secondary, the vector-and-graph pipeline is
better suited. For evaluation, we argue that both axes should always be reported together: a
single answer-quality score hides the grounding deficit of a vector pipeline, while a single
retrieval-quality score hides the breadth limitation of a vectorless pipeline. Future legal AI
systems may benefit from hybrid architectures that combine the structural precision of vectorless
tree navigation with the broad recall of vector-and-graph retrieval, jointly optimising for both
forms of correctness rather than trading one against the other.

**Acknowledgments.** [Acknowledgments to be added by authors.]

**Disclosure of Interests.** The authors have no competing interests to declare that are relevant
to the content of this article.

# References {.unnumbered}

::: {#refs}
:::
