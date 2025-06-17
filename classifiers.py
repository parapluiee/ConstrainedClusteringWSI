from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from constr_KMeans import ConKMeans
import utils
import warnings
from sklearn.exceptions import ConvergenceWarning
def regression(data_dict, emb_name, per_train=1):
     #multiclass solver, the one guillaume recommended
    solver = 'saga'
    #80/20 train test
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
        #This is horribly inefficient, but we cannot randomly split
        if per_train < 1:
            new_df = pd.DataFrame({"index":range(len(X_train)), "sem_label":Y_train})
            new_train, _ = utils.custom_train_test_split(new_df, train_split=per_train) 
            new_train_indices = list(new_train['index'])
            X_train = X_train[new_train_indices]
            Y_train = Y_train[new_train_indices]
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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
