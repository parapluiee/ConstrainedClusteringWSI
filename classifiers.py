from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from constr_KMeans import ConKMeans
import utils
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, v_measure_score, normalized_mutual_info_score

def regression(data_dict, emb_name):
     #multiclass solver, the one guillaume recommended
    solver = 'saga'
    #80/20 train test
    split = .8
    #group by lemma
    preds = pd.DataFrame({"pred":list(), "gold":list(), "lemma":list()})
    lemma_most_com = dict()
    data_dict = data_dict[emb_name] 
    for name in data_dict:
        #this randomly sorts the list
        #point where we split

        classifier = LR(solver=solver)
        #unsplit data
        labels = data_dict[name]['labels']
        X_train = data_dict[name]['X_train']
        Y_train = data_dict[name]['Y_train']
        X_test = data_dict[name]['X_test']
        Y_test = data_dict[name]['Y_test']

        if (len(labels) == 1):
            app_df = pd.DataFrame({
                "pred":[list(labels)[0]]*len(X_test),
                "cluster":[list(labels)[0]]*len(X_test),
                "gold":Y_test,
                "lemma":[name]*len(X_test)}
                    )
            #cross_validation, need to make sure equal senses in training data
        else: 
                #need to ensure equal distribution of labels
            classifier.fit(X_train, Y_train)
            pred = classifier.predict(X_test)
            app_df = pd.DataFrame({
            "pred":pred,
            "cluster":pred,
            "gold":Y_test,
            "lemma":[name]*len(X_test)}
            ) 
        preds = pd.concat([preds, app_df], axis=0)
    return preds

def sk_clustering(data_dict, emb_name):
    preds = pd.DataFrame({"pred":list(), "gold":list(), "lemma":list()})
    data_dict = data_dict[emb_name]
    # Iterate over each lemma
    for name in data_dict:
        sem_labels = data_dict[name]['labels']
        gold_labels = data_dict[name]['Y']
        embeddings = torch.tensor(data_dict[name]['X'])
        # Dimensionality reduction 
        pca = PCA(n_components=50, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
        #or UMAP
        umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        # Clustering with known number of senses
        n_senses = len(sem_labels)
        kmeans = KMeans(n_clusters=n_senses, random_state=42)
        predicted_labels = kmeans.fit_predict(reduced_embeddings)

        # Evaluation metrics
        app_df = pd.DataFrame({
            "pred": [sem_labels[int(x)] for x in predicted_labels],
            "gold": gold_labels,
            "lemma": [name] * len(gold_labels)
        })
        preds = pd.concat([preds, app_df], axis=0)

        # Calculate clustering metrics
        ari = adjusted_rand_score(gold_labels, predicted_labels)
        v_measure = v_measure_score(gold_labels, predicted_labels)
        nmi = normalized_mutual_info_score(gold_labels, predicted_labels)

        print(f"Adjusted Rand Index (ARI): {ari:.3f}")
        print(f"V-Measure: {v_measure:.3f}")
        print(f"NMI: {nmi:.3f}")
    
    return preds

def base_clustering(data_dict, emb_name, sim_metric, m_m, iterations=100):
    preds = pd.DataFrame({"pred":list(), "gold":list(), "lemma":list()})

    data_dict = data_dict[emb_name] 
    for name in data_dict:
        sem_labels = data_dict[name]['labels']
        X_train = torch.tensor(data_dict[name]['X_train'])
        Y_train = data_dict[name]['Y_train']
        X_test = torch.tensor(data_dict[name]['X_test'])
        Y_test = data_dict[name]['Y_test']
        #fine to find k based on all labels, should be the same if split works correctly
        k = len(sem_labels)
        # Define the number of iterations
        clusterer = ConKMeans(k, sim_metric, m_m)
        M = clusterer.fit(X_train, Y_train, sem_labels)

        distances_pred, clusters_pred = clusterer._assign_labels(X_test)

        app_df = pd.DataFrame({
            "pred":[sem_labels[int(x)] for x in np.argmax(np.dot(distances_pred, M), axis=1)],
                    "cluster":clusters_pred,
                    "gold":Y_test,
                    "lemma":[name]*len(X_test)}
                    ) 
        preds = pd.concat([preds, app_df], axis=0)
    return preds

def constr_clustering(data_dict, sim_metric, m_m, emb_name, n_seeds=1):

    data_dict = data_dict[emb_name] 
    preds = pd.DataFrame({"pred":list(), "gold":list(), "lemma":list()})
    for name in data_dict:
        senses = data_dict[name]['labels']
        train = data_dict[name]['train']
        X_train = torch.tensor(data_dict[name]['X_train'])
        Y_train = data_dict[name]['Y_train']
        X_test = torch.tensor(data_dict[name]['X_test'])
        Y_test = data_dict[name]['Y_test']

        
        train=train.reset_index(drop=True)
        num2label = [name for name, _ in train.groupby('sem_label')]
        seeds = [list(group.head(n_seeds).index.values) for name, group in train.groupby('sem_label')]
        
        
        #fine to find k based on all labels, should be the same if split works correctly
        k = len(senses)
        clusterer = ConKMeans(k, sim_metric, m_m)
        # Define the number of iterations
        
        #add logic to average a number of seeds for centroid initialization
        M = clusterer.fit(X_train, Y_train, senses, seeds=seeds, n_seeds=n_seeds)
        distances_pred, labels_pred = clusterer._assign_labels(X_test)
        app_df = pd.DataFrame({
            "pred":[senses[int(x)] for x in np.argmax(np.dot(distances_pred, M), axis=1)],
            "cluster":labels_pred,
            "gold":Y_test,
            "lemma":[name]*len(X_test)}
        ) 
        preds = pd.concat([preds, app_df], axis=0)
    return preds 
