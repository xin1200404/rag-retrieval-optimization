# RAG Retrieval Optimization & Evaluation

A project that evaluated and optimized retrieval strategies for scientific document search using the SciFact dataset.

## Overview

This project investigates how different retrieval methods affect retrieval quality and computational efficiency.

Four retrieval approaches are implemented and compared:

- BM25 lexical retrieval
- Dense retrieval using Sentence Transformers
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Hybrid retrieval with Cross-Encoder reranking

The project also includes candidate pool optimization, RRF parameter experiments, and query-level error analysis.

## Dataset

Experiments use the SciFact dataset in BEIR format.

The official test set is reserved for final evaluation, while a development set is created from the original training relevance judgments.

## Final Test Results

| Method | Recall@5 | nDCG@10 | Latency (ms) |
|---|---:|---:|---:|
| BM25 | 0.7284 | 0.6519 | 31.10 |
| Dense | 0.7379 | 0.6451 | 17.05 |
| Hybrid | 0.7395 | 0.6816 | 50.55 |
| Optimized Hybrid + Reranker | **0.7482** | **0.6976** | 3227.60 |

## Key Findings

1. Hybrid retrieval improves overall retrieval effectiveness compared with individual lexical and dense methods.
2. Cross-Encoder reranking achieves the highest retrieval quality but introduces substantial latency.
3. Candidate pool size creates a tradeoff between retrieval effectiveness and computation time.
4. RRF parameter changes do not necessarily improve end-to-end performance when candidate sets remain unchanged.
5. Query-level analysis reveals both successful reranking and retrieval regressions.

## Technologies

Python, NumPy, Pandas, scikit-learn, Sentence Transformers, BM25, PyTorch, and Matplotlib.

## Project Structure

- `notebooks/`: Retrieval implementation, evaluation, and experiments
- `results/`: Experimental metrics and visualizations
- `reports/`: Research report
- `data/`: Local SciFact dataset and embedding cache (not tracked by Git)

## Reproducing the Experiments

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Download the SciFact dataset in BEIR format into `data/scifact/`.
5. Open `notebooks/01_retrieval_experiments.ipynb` and run the notebook using the project-root working directory.

The notebook generates document embeddings, evaluates retrieval methods, performs parameter optimization, and saves experimental results.

## Limitations

This project evaluates document retrieval and reranking rather than a complete RAG answer-generation system. All reported latency measurements depend on the experimental hardware and software configuration.
