from sklearn import metrics
import numpy as np
import bcubed
def base_metrics(df, lem_most_com):
    output_dict = dict()
    output_dict['tot_acc'] = metrics.accuracy_score(df['gold'], df['pred'])
    output_dict['tot_f1_macro'] = metrics.f1_score(df['gold'], df['pred'], average='macro')

    per_acc = df.groupby('lemma').apply(lambda x: metrics.accuracy_score(x['gold'], x['pred']))
    per_f1_macro = df.groupby('lemma').apply(lambda x: metrics.f1_score(x['gold'], x['pred'], average='macro'))

    baseline = df.groupby('lemma').apply(lambda x: metrics.accuracy_score(x['gold'], [lem_most_com[x['lemma'].iloc[0]]] * len(x)))
    beat_baseline = baseline.le(per_acc)
    output_dict['beat_base_score'] = sum(beat_baseline) / len(beat_baseline)
    return output_dict

def clustering_metrics(df, lem_most_com):
    output_dict = dict()
    per_rand_score = df.groupby('lemma').apply(lambda x: metrics.rand_score(np.array(x['cluster']), np.array(x['gold'])))
    per_nmi = df.groupby('lemma').apply(lambda x: metrics.normalized_mutual_info_score(np.array(x['cluster']), np.array(x['gold'])))

    output_dict['tot_rand_score'] = per_rand_score.mean()
    output_dict['tot_nmi(vscore)'] = per_nmi.mean()
   

    baseline_nmi = df.groupby('lemma').apply(lambda x: metrics.normalized_mutual_info_score(x['gold'], [lem_most_com[x['lemma'].iloc[0]]] * len(x)))
    #bcubed requires rewriting of pandas to dictionary 
    per_fbcubed = df.groupby('lemma').apply(lambda x: bcubed_helper(x))
    output_dict['tot_fbcubed'] = per_fbcubed.mean()
    baseline_rand = df.groupby('lemma').apply(lambda x: metrics.rand_score(x['gold'], [lem_most_com[x['lemma'].iloc[0]]] * len(x)))

    beat_baseline_nmi = baseline_nmi.le(per_nmi)

    baseline_fbcubed = df.groupby('lemma').apply(lambda x: bcubed_helper(x))
    beat_baseline_rand = baseline_rand.le(per_rand_score)
    beat_baseline_fbcubed = baseline_fbcubed.le(per_fbcubed)
    output_dict['beat_base_score_rand'] = sum(beat_baseline_rand) / len(beat_baseline_rand)
    output_dict['beat_base_score_nmi(vscore)'] = sum(beat_baseline_nmi) / len(beat_baseline_nmi)
    
    output_dict['beat_base_score_fbcubed'] = sum(beat_baseline_fbcubed) / len(beat_baseline_fbcubed)
    return output_dict


def bcubed_helper(x, baseline_class=None):
    pred_dict = x['cluster'].to_dict()
    if baseline_class:
        gold_dict = {v:baseline_class for v in pred_dict.keys()}
    else:
        gold_dict = x['gold'].to_dict()
    pred_dict = {v:{pred_dict[v]} for v in pred_dict}
    gold_dict = {v:{gold_dict[v]} for v in gold_dict}
    precision = bcubed.precision(pred_dict, gold_dict)
    recall = bcubed.recall(pred_dict, gold_dict)
    return bcubed.fscore(precision, recall)
