import data
import ws_embeddings
import classifiers
import metrics
import pandas as pd
import argparse
from numpy import load
import numpy as np
import utils 
parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument('data_path')
parser.add_argument('gold_path')
parser.add_argument('embedding', choices=['bert', 'fasttext'])
parser.add_argument('calculated', choices=['T', 'F'])
parser.add_argument('classifier', choices=['regression', 'base-clustering', 'constr-clustering'])
parser.add_argument('clus_metric')
args=parser.parse_args()


XML_PATH = args.data_path
GOLD_PATH = args.gold_path
EMBED = args.embedding
CLASSIFIER = args.classifier
SPLIT = .8
CALC = args.calculated
CLUS_METRIC = args.clus_metric
def main(xml_path, gold_path, split, embed, calc, classifier, clus_metric):
    df = data.get_df(gold_path, xml_path)
    lem_most_com = df.groupby('lemma')['sem_label'].apply(lambda x:x.mode().iloc[0]).to_dict()
    print("Creating embeddings from: ", xml_path)
    match clus_metric:
        case "cossim": 
            cl_metric = utils.cl_cossim
            m_m = np.argmax
        case _:
            cl_metric = utils.cl_distance
            m_m = np.argmin
    match embed:
        case "bert":
            if calc == 'T':
                np_data = load(xml_path + 'bert.npy', allow_pickle=True)
                #print(np_data)
                df['bert'] = np_data
                #print(df)
                #return
            else:
                df['bert'] = ws_embeddings.embed_bert(df)
                np.save(open(xml_path + 'bert.npy', 'wb'), np.array(df['bert']))
        case "fasttext":
                if calc == 'T':
                    np_data = load(xml_path + 'ft.npy', allow_pickle=True)
                #print(np_data)
                    df['fasttext'] = np_data
                #print(df)
                #return
                else:
                    df['fasttext'] = ws_embeddings.fasttext_emb(df)
                    np.save(open(xml_path + 'ft.npy', 'wb'), np.array(df['fasttext']))
            
    print("Beginning training: ", classifier)
    match classifier:
        case "regression":
            preds = classifiers.regression(df[['lemma', 'sem_label', embed]], split, embed)
        case "base-clustering":
            preds = classifiers.base_clustering(df[['lemma', 'sem_label', embed]], emb_name=embed, m_m=m_m, sim_metric=cl_metric, split=split) 
        case "constr-clustering":
            preds = classifiers.constr_clustering(df[['lemma', 'sem_label', embed]], sim_metric=cl_metric, m_m=m_m, split=split, emb_name=embed, n_seeds=1)
    
    match classifier:
        case "regression":
            preds['cluster'] = preds['pred'] 
            clustering_metric = metrics.clustering_metrics(preds, lem_most_com)
            metric = metrics.base_metrics(preds, lem_most_com)
        case _:
            clustering_metric = metrics.clustering_metrics(preds, lem_most_com)
            metric = metrics.base_metrics(preds, lem_most_com)
    
    print(clustering_metric)
    print(metric)
main(XML_PATH, GOLD_PATH, SPLIT, EMBED, CALC, CLASSIFIER, CLUS_METRIC)
