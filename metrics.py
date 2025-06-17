from sklearn import metrics
import numpy as np
def base_metrics(df, lem_most_com):
    output_dict = dict()
    output_dict['tot_acc'] = metrics.accuracy_score(df['gold'], df['pred'])
    output_dict['tot_f1_macro'] = metrics.f1_score(df['gold'], df['pred'], average='macro')

    output_dict['per_acc'] = df.groupby('lemma').apply(lambda x: metrics.accuracy_score(x['gold'], x['pred']))
    output_dict['per_f1_macro'] = df.groupby('lemma').apply(lambda x: metrics.f1_score(x['gold'], x['pred'], average='macro'))

    output_dict['baseline'] = df.groupby('lemma').apply(lambda x: metrics.accuracy_score(x['gold'], [lem_most_com[x['lemma'].iloc[0]]] * len(x)))
    output_dict['beat_baseline'] = output_dict['baseline'].le(output_dict['per_acc'])
    output_dict['beat_base_score'] = sum(output_dict['beat_baseline']) / len(output_dict['beat_baseline'])
    return output_dict

def clustering_metrics(df, lem_most_com):
    output_dict = dict()
    output_dict['per_rand_score'] = df.groupby('lemma').apply(lambda x: metrics.rand_score(np.array(x['cluster']), np.array(x['gold'])))
    output_dict['per_nmi(vscore)'] = df.groupby('lemma').apply(lambda x: metrics.normalized_mutual_info_score(np.array(x['cluster']), np.array(x['gold'])))

    output_dict['tot_rand_score'] = output_dict['per_rand_score'].mean()
    output_dict['tot_nmi(vscore)'] = output_dict['per_nmi(vscore)'].mean()
   

    output_dict['baseline_nmi(vscore)'] = df.groupby('lemma').apply(lambda x: metrics.normalized_mutual_info_score(x['gold'], [lem_most_com[x['lemma'].iloc[0]]] * len(x)))
    


    output_dict['baseline_rand'] = df.groupby('lemma').apply(lambda x: metrics.rand_score(x['gold'], [lem_most_com[x['lemma'].iloc[0]]] * len(x)))

    output_dict['beat_baseline_nmi(vscore)'] = output_dict['baseline_nmi(vscore)'].le(output_dict['per_nmi(vscore)'])

    output_dict['beat_baseline_rand'] = output_dict['baseline_rand'].le(output_dict['per_rand_score'])

    output_dict['beat_base_score_rand'] = sum(output_dict['beat_baseline_rand']) / len(output_dict['beat_baseline_rand'])

    output_dict['beat_base_score_nmi(vscore)'] = sum(output_dict['beat_baseline_nmi(vscore)']) / len(output_dict['beat_baseline_nmi(vscore)'])
    return output_dict


