import xml.etree.ElementTree as ET


import pandas as pd
def create_df(data_path):
    #Parsing xml
    print(data_path)
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

