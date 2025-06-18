import utils, data, ws_embeddings, classifiers, metrics
from numpy import load
import numpy as np
import json
EMBEDDINGS = ['bert', 'fasttext', 'camembert']#, 'frequency']
CLASSIFIERS = ['regression', 'base-clustering', 'constr-clustering']
DATA_PATH =  'data/FSE-1.1.data.xml'
GOLD_PATH = 'data/FSE-1.1.gold.key.txt'
CL_METRICS = ['cossim', 'distance']

def main(xml_path, gold_path, embeddings):
    print("Load Data")
    df = data.get_df(gold_path, xml_path)
    lem_most_common = df.groupby('lemma')['sem_label'].apply(lambda x:x.mode().iloc[0]).to_dict()

    print("BERT embeddings")
    #df['bert'] = ws_embeddings.embed_bert(df)

    df['bert'] = load(xml_path + 'bert.npy', allow_pickle=True)
    print("Fasttext embeddings")
    #df['fasttext'] = ws_embeddings.fasttext_emb(df)

    df['fasttext'] = load(xml_path + 'ft.npy', allow_pickle=True)
    #df['frequency'] = ws_embeddings.sentence_to_vector(df, )

    print("Camembert embeddings")
    df['camembert'] = load(xml_path + 'camembert.npy', allow_pickle=True)
    #df['camenbert'] = ws_embeddings.embed_camembert(df)
    data_dict = data.prepare_data(df, embeddings)
    regression_dict = dict()
    clustering_dict = dict()

    for embed in embeddings:
        
        print("\n" + embed + "\n----")
        print("--regression")
        
        for reduction in [1.0, .75, .50, .40, .30]:
            print("----Percentage of training data used: ", reduction)
            regression_dict[embed + '_regression_' + str(int(reduction*100))] = classifiers.regression(data_dict, embed, per_train=reduction)
        
        print("--base-cl-cos")
        
        clustering_dict[embed + '_base-clustering-cos'] = classifiers.base_clustering(data_dict, emb_name=embed, m_m=np.argmax, sim_metric=utils.cl_cossim, supervised=False) 
        regression_dict[embed + '_base-clustering-cos'] = classifiers.base_clustering(data_dict, emb_name=embed, m_m=np.argmax, sim_metric=utils.cl_cossim, supervised=True) 
        
        print("--base-cl-dist")
        clustering_dict[embed + '_base-clustering-dist'] = classifiers.base_clustering(data_dict, emb_name=embed, m_m=np.argmin, sim_metric=utils.cl_distance, supervised=False) 
        regression_dict[embed + '_base-clustering-dist'] = classifiers.base_clustering(data_dict, emb_name=embed, m_m=np.argmin, sim_metric=utils.cl_distance, supervised=True) 
        #30/31 is the maximum possible senses, but the vast majority of senses have < 5 examples, with a majority around 1/2/3
        for seeds in range(1, 6):
            print("--Number of seeds (constraints): ", seeds)
            print("----const-cl-cos")
            clustering_dict[embed + '_constr-clustering-cos_' + str(seeds)] = classifiers.constr_clustering(data_dict, sim_metric=utils.cl_cossim, m_m=np.argmax, emb_name=embed, n_seeds=seeds, supervised=False)
            regression_dict[embed + '_constr-clustering-cos_' + str(seeds)] = classifiers.constr_clustering(data_dict, sim_metric=utils.cl_cossim, m_m=np.argmax, emb_name=embed, n_seeds=seeds, supervised=True)
            print("----constr-cl-dist")
            clustering_dict[embed + '_constr-clustering-dist_' + str(seeds)] = classifiers.constr_clustering(data_dict, sim_metric=utils.cl_distance, m_m=np.argmin, emb_name=embed, n_seeds=seeds, supervised=False)
            regression_dict[embed + '_constr-clustering-dist_' + str(seeds)] = classifiers.constr_clustering(data_dict, sim_metric=utils.cl_distance, m_m=np.argmin, emb_name=embed, n_seeds=seeds, supervised=True)
    
    reg_output = {name:metrics.base_metrics(regression_dict[name], lem_most_common) for name in regression_dict}

    cl_output = {name:metrics.clustering_metrics(clustering_dict[name], lem_most_common) for name in clustering_dict}
    print(reg_output)
    print(cl_output)
    with open('supervised_result.json', 'w') as f:
        json.dump(reg_output, f)
    with open('unsupervised_result.json', 'w') as f:
        json.dump(cl_output, f)
    

main(DATA_PATH, GOLD_PATH, EMBEDDINGS)
