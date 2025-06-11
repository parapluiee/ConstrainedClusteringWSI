import numpy as np
import fasttext
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from utils import custom_train_test_split

#downloading french embedding model
model_path = hf_hub_download(repo_id="facebook/fasttext-fr-vectors", filename="model.bin")
model_ft = fasttext.load_model(model_path)


def sentence_to_vector(tokens, idx, model, window=5, vector_size=300):
    start = max(idx - window, 0)
    end = min(idx + window + 1, len(tokens))
    context = tokens[start:idx] + tokens[idx+1:end]

    vectors = []
    for word in context:
      vectors.append(model.get_word_vector(word))  # for fastText

    return np.mean(vectors, axis=0)


def train_classifier(df):
    """
    Train a classifier on the given DataFrame.
    The DataFrame should contain 'lemma', 'sem_label', and 'emb_ft' columns.
    """
    y_true_all = []
    y_pred_all = []
    results = []

    for name, group in df.groupby('lemma'):
        if group['sem_label'].nunique() < 2:
            continue
        train, test = custom_train_test_split(group)
        # train, test = train_test_split(group,test_size=0.2, random_state=42)
        y_test = test['sem_label'].tolist()
        most_common = train['sem_label'].mode()[0]
        acc_baseline = accuracy_score(y_test, [most_common] * len(y_test))
        if group['sem_label'].nunique() >= 2:
            clf = LogisticRegression(max_iter=1000, solver='saga')
            X_train = np.array([x for x in train['emb_ft']])
            X_test = np.array([x for x in test['emb_ft']])
            clf.fit(X_train, train['sem_label'])
            y_pred = clf.predict(X_test)

            acc_model = accuracy_score(y_test, y_pred)
        # if the group has only one sense that sense no classification is needed
        else:
            print(f"verb {name} has only one sense. no classification is need.")
            y_pred = [most_common] * len(test)
            acc_model = acc_baseline


    # Track all predictions for global report
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

    results.append({
        'lemma': name,
        'baseline_acc': acc_baseline,
        'model_acc': acc_model,
        'diff': acc_model - acc_baseline
    })

def eval(results):
    # results
    diffs = [r['diff'] for r in results]
    best = max(results, key=lambda x: x['diff'])
    worst = min(results, key=lambda x: x['diff'])
    avg_diff = np.mean(diffs)

    print("\n=== Model vs Baseline Comparison ===")
    print(f"models accuracy: {accuracy_score(y_true_all, y_pred_all):.4f} ")
    print(f"Best improvement: {best['lemma']} (+{best['diff']:.2f})")
    print(f"Worst (drop): {worst['lemma']} ({worst['diff']:.2f})")
    print(f"Average difference from baseline: {avg_diff:.2f}")
    # print(classification_report(y_true_all, y_pred_all, zero_division=0))

if __name__ == "__main__":
    # adding fasttext embeddings to dataframe
    df['emb_ft'] = df.apply(lambda x: sentence_to_vector(x['sent'], x['idx'], model_ft), axis=1)
    # Train the classifier
    results = train_classifier(df)
    # Evaluate the results
    eval(results)
