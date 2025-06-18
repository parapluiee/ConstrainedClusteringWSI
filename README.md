# ConstrainedClusteringWSI

This project provides a modular framework for supervised, unsupervised, and semi-supervised word sense disambiguation (WSD) and word sense induction (WSI) on french verbs using FSE dataset. It supports multiple embedding strategies (BERT, CamemBERT, FastText, frequency-based) and clustering/classification approaches, with a focus on constrained clustering for leveraging limited supervision. 

## Features

- **Flexible Embedding Support:** BERT, CamemBERT, FastText, and frequency-based representations.
- **Multiple Evaluation Modes:** Supervised (classification), unsupervised (clustering), and semi-supervised (constrained clustering).
- **Custom K-Means:** Implementation of constrained K-means for semi-supervised clustering.
- **Comprehensive Metrics:** Accuracy, F1, Rand Index, NMI, F-Bcubed, and baseline comparisons.
- **Reproducible Experiments:** Modular scripts for single runs and batch evaluations.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parapluiee/ConstrainedClusteringWSI.git
   cd ConstrainedClusteringWSI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data:**
   - Place your XML data file and gold key file in the data directory.
   - Update paths in your scripts if necessary.

## Usage

### Single Experiment

Run a single experiment with specified embedding and classifier:

```bash
python main.py data/FSE-1.1.data.xml data/FSE-1.1.gold.key.txt 'camembert' 'regression'
```

**Arguments:**
- `<data_path>`: Path to XML data file.
- `<gold_path>`: Path to gold key file.
- `<embedding>`: One of `bert`, `camembert`, `fasttext`, `freq`.
- `<classifier>`: One of `regression`, `base-clustering`, `constr-clustering`.
- `--calculated`: Use precomputed embeddings (`true` or `false`).
- `--clus_metric`: Clustering metric (`cossim` for cosine, `dist` for Euclidean).
- `--eval_option`: `supervised` or `unsupervised`.

### Batch Experiments

To run comprehensive evaluations across all settings:

```bash
python collect_data.py
```

Edit `collect_data.py` to customize experiment sweeps.



## Project Structure

- `data.py`: Data loading and preparation.
- `ws_embeddings.py`: Embedding generation.
- `classifiers.py`: Classification and clustering logic.
- `constr_KMeans.py`: Constrained K-means implementation.
- `utils.py`: Utility functions.
- `metrics.py`: Evaluation metrics.
- `main.py`: Single experiment runner.
- `collect_data.py`: Batch experiment runner.
- `data` folder containing the french dataset

## License

This project is for research and educational purposes.

