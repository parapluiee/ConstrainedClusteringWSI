# *Semi-supervised Word Sense Disambiguation*

This project implements both supervised Word Sense Disambiguation (WSD) and semi-supervised Word Sense Induction (WSI) approaches, exploring the trade-off between annotated data requirements and clustering-based methods with constraints.

Overview

The project addresses the fundamental challenge in WSD: the lack of sufficient sense-annotated data. We compare three approaches:

  Supervised WSD: Traditional classification with labeled data

  Unsupervised WSI: Clustering-based sense discovery
  
  Semi-supervised WSI: Constrained clustering with minimal supervision

# Dataset

FrenchSemEval (FSE) - French verb disambiguation dataset with Wiktionary sense annotations

XML format with contextual sentences

Gold standard annotations in separate key file

Each instance contains: target word, lemma, context sentence, and sense label

# Word Representations
Three different context representation methods are implemented:

1. BERT Embeddings (bert) :
  Uses DistilBERT-base-uncased|
  Contextual embeddings for target words|
  Captures semantic relationships in context
2. FastText Embeddings (fasttext) :
  French FastText pre-trained vectors|
  Context window averaging (±5 words around target)|
  Static embeddings with local context

# not sure ( u can delete this guys )
"3. Frequency Features (frequency)"
  TF-IDF or count-based features
  Context word frequencies around target word
  Sparse vector representation of vocabulary

Classification Methods

Supervised Regression (regression):
  Logistic Regression with SAGA solver|
  Supports training data reduction experiments|
  Baseline: Most frequent sense prediction
  
Base Clustering (base-clustering):
  Standard K-means clustering|
  No supervision, purely unsupervised|
  Maps clusters to senses using confusion matrix

Constrained Clustering (constr-clustering):
  Semi-supervised K-means with seed constraints|
  Uses n_seeds examples per cluster for initialization|
  Balances supervision and clustering approaches
  
# Project Structure: 

├── data.py                 # XML parsing and data preparation

├── ws_embeddings.py        # Word representation methods

├── classifiers.py          # WSD and WSI implementations

├── constr_KMeans.py       # Custom constrained K-means algorithm

├── utils.py               # Utility functions (train/test split, distance metrics)

├── metrics.py             # Evaluation metrics for classification and clustering

├── main.py                # Single experiment runner

├── collect_data.py        # Comprehensive evaluation script

└── data/

    ├── FSE-1.1.data.xml   # FrenchSemEval dataset
    
    └── FSE-1.1.gold.key.txt # Gold standard annotations

Usage
Single Experiment
  python main.py <data_path> <gold_path> <embedding> <calculated> <classifier> <clus_metric>

Parameters:
data_path: Path to FSE XML file|
gold_path: Path to gold key file|
embedding: bert, fasttext, or frequency|
calculated: T (load cached) or F (compute new embeddings)|
classifier: regression, base-clustering, or constr-clustering|
clus_metric: cossim (cosine similarity) or distance (Euclidean)

Examples:
Supervised WSD with BERT: 

python main.py data/FSE-1.1.data.xml data/FSE-1.1.gold.key.txt bert F regression distance


Semi-supervised clustering with FastText:

python main.py data/FSE-1.1.data.xml data/FSE-1.1.gold.key.txt fasttext F constr-clustering cossim

# Frequency-based unsupervised clustering ( u know )
python main.py data/FSE-1.1.data.xml data/FSE-1.1.gold.key.txt frequency F base-clustering distance

Comprehensive Evaluation
  python collect_data.py
    Runs all combinations of:
      Embeddings: BERT, FastText, Frequency|
      Training data reductions: 100%, 75%, 50%, 40%, 30%|
      Constraint levels: 1-30 seeds per cluster|
      Distance metrics: Cosine similarity, Euclidean distance

Evaluation Metrics
  Classification Metrics
    Accuracy: Overall and per-lemma classification accuracy
    F1-Score: Macro-averaged F1 across all senses
    Baseline Comparison: Performance vs. most frequent sense
  Clustering Metrics
    Rand Score: Agreement between predicted and gold clusters
    Normalized Mutual Information (NMI): Information-theoretic clustering quality
    F-B-Cubed: Specialized WSI evaluation metric
    
Dependencies:

torch

transformers

scikit-learn

fasttext

huggingface_hub

pandas

numpy
