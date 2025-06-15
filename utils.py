from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sklearn.metrics as sklm
import pandas as pd
import numpy as np
def custom_train_test_split(df):
  # Get the value counts of 'sem_label'
  label_counts = df['sem_label'].value_counts()

  # Identify labels with only one example
  single_example_labels = label_counts[label_counts == 1].index.tolist()

  # Filter the DataFrame to get rows with single examples
  rows_to_duplicate = df[df['sem_label'].isin(single_example_labels)]

  # Duplicate these rows
  duplicated_rows = pd.concat([rows_to_duplicate] * 1, ignore_index=True) # Duplicate once

  # Concatenate the original DataFrame with the duplicated rows
  df = pd.concat([df, duplicated_rows], ignore_index=True)

  train, test = train_test_split(df,test_size=0.2, stratify=df['sem_label'])#, random_state=42)

  return train, test


def agirre_matr(cluster, ws, senses):
    
    s2idx = {senses[i]:i for i in range(len(senses))}
    M = np.zeros((len(senses), len(senses)))
    for i in range(len(ws)):
        M[cluster[i]][s2idx[ws[i]]]+=1
    M = M / M.sum(axis=1)[:, None]
    return M
    #cluster assignment, real word sense\


def cl_distance(X, centroids):
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def cl_cossim(X, centroids):
    return sklm.pairwise.cosine_similarity(X, centroids)
