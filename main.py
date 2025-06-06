import data
import ws_embeddings
import classifiers
import metrics
import pandas as pd
import argparse

parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument('data_path')
parser.add_argument('gold_path')
parser.add_argument('embedding', choices=['bert_emb'])
parser.add_argument('classifier', choices=['regression'])
args=parser.parse_args()


XML_PATH = args.data_path
GOLD_PATH = args.gold_path
EMBED = args.embedding
CLASSIFIER = args.classifier
SPLIT = .8

def main(xml_path, gold_path, split, embed, classifier):
    df = data.get_df(gold_path, xml_path)
    lem_most_com = df.groupby('lemma')['sem_label'].apply(lambda x:x.mode().iloc[0]).to_dict()
    print(lem_most_com)
    match embed:
        case "bert_emb":
            df['bert_emb'] = ws_embeddings.embed_bert(df)
    match classifier:
        case "regression":
            preds = classifiers.classifier(df[['lemma', 'sem_label', embed]], split, embed)
    
    
    print(lem_most_com)
    #preds have columns:
        #pred, gold, lemma
    metric = metrics.base_metrics(preds, lem_most_com)
    print(metric)
main(XML_PATH, GOLD_PATH, SPLIT, EMBED, CLASSIFIER)
