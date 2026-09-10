
⚖️ **Legal AI — Intelligent Legal Question Answering**

Legal AI is a research-oriented AI system developed for legal question answering based on Indian law, with a primary focus on the Constitution of India. The project explores and compares two Retrieval-Augmented Generation (RAG) approaches for improving the retrieval and answering of legal information.

The first approach, **PageIndex RAG**, uses hierarchical, tree-based document retrieval combined with multi-hop agent reasoning. The second approach, **Vector + Graph RAG**, combines semantic vector search using **Qdrant** with knowledge-graph traversal through **Neo4j**.

A comprehensive evaluation framework was developed to compare both approaches using **110 legal questions** across six metrics: **context precision, context recall, hit rate, faithfulness, answer correctness, and semantic similarity**.

The project also includes reproducible evaluation scripts, automated result visualizations, retrieval adapters, and Docker-based infrastructure for Qdrant and Neo4j. A research paper was also prepared to document the methodology, experiments, and findings.

🔬 **Research Focus:**
The main objective is to study how hierarchical, vectorless retrieval compares with traditional vector-based and graph-enhanced RAG methods for legal question answering. The research evaluates both the quality of retrieved legal context and the quality of the generated answers.

The final evaluation revealed an interesting **3–3 split**: **PageIndex RAG performed better on retrieval-quality metrics, while Vector + Graph RAG achieved better results on answer-quality metrics.**
