import data
import ws_embeddings
import classifiers
import metrics
import pandas as pd
import argparse
from numpy import load
import numpy as np
import utils
import time
import frequency_embeddings
parser=argparse.ArgumentParser(description="Run the Word Sense evaluation with different embeddings and classifiers")
parser.add_argument('data_path')
parser.add_argument('gold_path')
parser.add_argument('embedding', choices=['bert', 'camembert', 'fasttext', 'context'])
parser.add_argument('classifier', choices=['regression', 'base-clustering', 'constr-clustering'])
parser.add_argument('--calculated', choices=['true', 'false'], default='false')
parser.add_argument('--clus_metric', choices=['cossim', 'dist'], help="cossim for cosine similarity, distance for euclidean distance", default='cossim')

parser.add_argument('--eval_option', default='prediction', choices=['prediction', 'clustering']) 
args=parser.parse_args()

XML_PATH = args.data_path
GOLD_PATH = args.gold_path
EMBED = args.embedding
CLASSIFIER = args.classifier
SPLIT = .8
CALC = args.calculated
CLUS_METRIC = args.clus_metric
EVAL_OP = args.eval_option
def main(xml_path, gold_path, split, embed, calc, classifier, clus_metric, eval_option):
    df = data.get_df(gold_path, xml_path)
    lem_most_com = df.groupby('lemma')['sem_label'].apply(lambda x:x.mode().iloc[0]).to_dict()
    print("Creating embeddings from: ", xml_path)
    match clus_metric:
        case "cossim": 
            cl_metric = utils.cl_cossim
            m_m = np.argmax
        case "dist":
            cl_metric = utils.cl_distance
            m_m = np.argmin
    if calc == 'true':
        df[embed] = load(xml_path + embed + '.npy', allow_pickle=True)
    else:
        match embed:
            case "bert":
                df['bert'] = ws_embeddings.embed_bert(df)
                np.save(open(xml_path + 'bert.npy', 'wb'), np.array(df['bert']))
            case "camembert":
                df['camembert'] = ws_embeddings.embed_camembert(df)
                np.save(open(xml_path + 'camembert.npy', 'wb'), np.array(df['camembert']))
            case "fasttext":
                df['fasttext'] = ws_embeddings.fasttext_emb(df)
                np.save(open(xml_path + 'ft.npy', 'wb'), np.array(df['fasttext']))
    data_dict = data.prepare_data(df, [embed])
    print("Beginning training: ", classifier)
    prediction = eval_option == 'prediction' 
    match classifier:
        case "regression":
            preds = classifiers.regression(data_dict, embed)
        case "base-clustering":
            preds = classifiers.base_clustering(data_dict, emb_name=embed, m_m=m_m, sim_metric=cl_metric, prediction=prediction) 
        case "constr-clustering":
            preds = classifiers.constr_clustering(data_dict, sim_metric=cl_metric, m_m=m_m, emb_name=embed, n_seeds=1, prediction=prediction)

    with open('results.txt', 'a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write(f"Embedding: {embed}, Classifier: {classifier}, Clus Metric: {clus_metric}\n")
        if prediction:
            metric = metrics.base_metrics(preds, lem_most_com)
            print("total accuracy", metric['tot_acc'])
            print("total f1 macro", metric['tot_f1_macro'])

            f.write(f"total accuracy: {metric['tot_acc']}\n")
            f.write(f"total f1 macro: {metric['tot_f1_macro']}\n")
            f.write(f"all metrics: {metric}\n")
            f.write(f"percentage of lemmas which beat baseline: {metric['beat_base_score']}")
        else:
            clustering_metric = metrics.clustering_metrics(preds, lem_most_com)
            print("clustering total rand score", clustering_metric['tot_rand_score'])
            print("clustering total nmi(vscore)", clustering_metric['tot_nmi(vscore)'])
            f.write(f"clustering metrics: {clustering_metric}\n")
        f.write("\n\n")
main(XML_PATH, GOLD_PATH, SPLIT, EMBED, CALC, CLASSIFIER, CLUS_METRIC, EVAL_OP)
