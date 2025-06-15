from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from constr_KMeans import ConKMeans
import utils
def regression(df, split, emb_name):
     #multiclass solver, the one guillaume recommended
    solver = 'saga'
    #80/20 train test
    split = .8
    #group by lemma
    preds = pd.DataFrame({"pred":list(), "gold":list(), "lemma":list()})
    lemma_most_com = dict()
    
    for name, group in df.groupby('lemma'):
        #this randomly sorts the list
        #point where we split

        classifier = LR(solver=solver)
        #unsplit data
        labels = set(group['sem_label'].values)
        #sometimes there is only one sem_label type, which is trivial
        if (len(labels) == 1 or len(group) == 1):
            app_df = pd.DataFrame({
                "pred":[list(labels)[0]]*len(X_test),
                "gold":Y_test,
                "lemma":[name]*len(X_test)}
                    )
        else:
            #cross_validation, need to make sure equal senses in training data
            train, test = utils.custom_train_test_split(group) 
            X_train = np.array([np.array(x) for x in train[emb_name]])
            Y_train = train['sem_label'].values

            X_test = np.array([np.array(x) for x in test[emb_name]])
            Y_test = test['sem_label'].values


                #need to ensure equal distribution of labels
            classifier.fit(X_train, Y_train)
            pred = classifier.predict(X_test)
            app_df = pd.DataFrame({
            "pred":pred,
            "gold":Y_test,
            "lemma":[name]*len(X_test)}
            ) 
        preds = pd.concat([preds, app_df], axis=0)
    return preds

def base_clustering(df, emb_name, split=.8, iterations=100):
    preds = pd.DataFrame({"pred":list(), "gold":list(), "lemma":list()})

    for name, lemma in df.groupby('lemma'):
        train, test = utils.custom_train_test_split(lemma)
        X_train = torch.tensor(np.array([np.array(x) for x in train[emb_name]]))
        Y_train = np.array(train['sem_label'])

        X_test = torch.tensor(np.array([np.array(x) for x in test[emb_name]]))
        Y_test = np.array(test['sem_label'])
        
        #fine to find k based on all labels, should be the same if split works correctly
        k = lemma['sem_label'].nunique()
        centroids = X_train[torch.randperm(X_train.size(0))[:k]]
        # Define the number of iterations
        sem_labels = list(set(lemma['sem_label']))
        sem_labels.sort()
        for _ in range(iterations):
          # Calculate distances from data points to centroids
            distances = torch.cdist(X_train, centroids)

            # Assign each data point to the closest centroid
            _, labels = torch.min(distances, dim=1)

            # Update centroids by taking the mean of data points assigned to each centroid
            for i in range(k):
                if torch.sum(labels == i) > 0:
                    centroids[i] = torch.mean(X_train[labels == i], dim=0)

        #use entire data as training and testa
        distances_pred = torch.cdist(X_test, centroids)
        M = utils.agirre_matr(labels, Y_train, sem_labels)
            # Assign each data point to the closest centroid
        _, clusters_pred = torch.min(distances_pred, dim=1)
        app_df = pd.DataFrame({
            "pred":[sem_labels[int(x)] for x in np.argmax(np.dot(distances_pred, M), axis=1)],
                    "cluster":clusters_pred,
                    "gold":Y_test,
                    "lemma":[name]*len(X_test)}
                    ) 
        preds = pd.concat([preds, app_df], axis=0)
    return preds

def constr_clustering(df, split, emb_name, n_seeds=1):

    preds = pd.DataFrame({"pred":list(), "gold":list(), "lemma":list()})
    for name, lemma in df.groupby('lemma'):
        train, test = utils.custom_train_test_split(lemma)
        train=train.reset_index(drop=True)
        num2label = [name for name, _ in train.groupby('sem_label')]
        senses = list(set(lemma['sem_label'].tolist()))
        senses.sort()
        seeds = [list(group.head(n_seeds).index.values) for name, group in train.groupby('sem_label')]
        
        X_train = torch.tensor(np.array([x for x in train[emb_name]]))

        Y_train = np.array(train['sem_label'])

        X_test = torch.tensor(np.array([x for x in test[emb_name]]))
        Y_test = np.array(test['sem_label'])
        
        #fine to find k based on all labels, should be the same if split works correctly
        k = lemma['sem_label'].nunique()
        """
        #workaround until even split is achieved
        while (len(seeds) < k):
            seeds.append([])
            new_sense = num2label[0]
            new_sense = new_sense.replace('0', str(len(num2label)))
            num2label.append(new_sense)
        """
        clusterer = ConKMeans(k)
        # Define the number of iterations
        
        #add logic to average a number of seeds for centroid initialization
        centr_ids = [x[0:n_seeds] for x in seeds]
        #mean n_seeds for each cluster
        centroids = torch.stack([torch.mean(torch.stack(list(train.iloc[seed_ids][emb_name])), axis=1) for seed_ids in centr_ids])
        M = clusterer.fit(seeds, centroids, X_train, Y_train, senses)
        distances_pred, labels_pred = clusterer._assign_labels(X_test)
        app_df = pd.DataFrame({
            "pred":[senses[int(x)] for x in np.argmax(np.dot(distances_pred, M), axis=1)],
            "cluster":labels_pred,
            "gold":Y_test,
            "lemma":[name]*len(X_test)}
        ) 
        preds = pd.concat([preds, app_df], axis=0)
    return preds 
