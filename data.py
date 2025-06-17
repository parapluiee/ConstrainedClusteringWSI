import xml.etree.ElementTree as ET
import utils
import numpy as np
import pandas as pd
def create_df(data_path):
    """
    Create a DataFrame from the XML data.
    Args:
        data_path (str): Path to the XML file.
    Returns:
        pd.DataFrame: DataFrame containing the data with columns 'ist_id', 'word', 'lemma', 'idx', and 'sent', 
        corresponding to instance ID, target word, target word's lemma, index in sentence, and the sentence itself.
    """
    #Parsing xml
    tree = ET.parse(data_path)
    root = tree.getroot()
    #Each sentence has one instance (data point), as far as I could tell
    sents = [sent for text in root for sent in text]
    ist_id = list()
    word = list()
    lemma = list()
    idx = list()
    sent_list = list()
    for sent in sents:

        #Find instance, i is the index of that instance
        for i in range(len(sent)):
            if sent[i].tag == 'instance':
                inst = sent[i]
                break
        #ids point to gold classes
        ist_id.append(inst.attrib['id'])

        #not needed but it looks neater
        word.append(inst.text)

        #which classifier we will use
        lemma.append(inst.attrib['lemma'])

        #used when creating bert embeddings
        idx.append(i)
        sent_list.append([w.text for w in sent])
    return pd.DataFrame({
        "ist_id":ist_id,
        "word":word,
        "lemma":lemma,
        "idx":idx,
        "sent":sent_list
    })


def get_df(gold_path, xml_path):
    df = create_df(xml_path)
    with open(gold_path) as f:
        s = [x.replace('\n', '').split(' ') for x in f]
        gold_labels = {x[0]:x[1] for x in s}
    df['sem_label'] = df['ist_id'].map(lambda x: gold_labels[x])
    return df


def prepare_data(df, emb_name, split=.8):
    """
    Prepare data for training and testing.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        emb_name (list): List of available embedding methods to be used (['bert', 'fasttext']).
        split (float): Proportion of data to be used for training.
    Returns:
        dict: Dictionary containing training and testing data for each embedding method and lemma.
        (train, X_train, X_test, Y_train, Y_test, labels)
    """
    by_emb = {emb:dict() for emb in emb_name} 
    for name, group in df.groupby('lemma'):
        labels = list(set(group['sem_label'].values))
        labels.sort()
        train, test = utils.custom_train_test_split(group)
        for emb in emb_name:
            X_train = np.array([np.array(x) for x in train[emb]])
            Y_train = train['sem_label'].values

            X_test = np.array([np.array(x) for x in test[emb]])
            Y_test = test['sem_label'].values
            by_emb[emb][name] = {
                "train":train,
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
                "labels": labels
                }
    return by_emb


