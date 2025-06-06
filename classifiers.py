from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
        print(labels)
        #sometimes there is only one sem_label type, which is trivial
        if (len(labels) == 1 or len(group) == 1):
            app_df = pd.DataFrame({
                "pred":[labels[0]]*max(1, int(split*len(group))),
                "gold":Y_test,
                "lemma":[name]*len(X_test)}
                    )
        else:
            #cross_validation, need to make sure equal senses in training data
            train, test = train_test_split(group, train_size=split)
            X_train = np.array([x.numpy() for x in train[emb_name]])
            Y_train = train['sem_label'].values

            X_test = np.array([x.numpy() for x in test[emb_name]])
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
