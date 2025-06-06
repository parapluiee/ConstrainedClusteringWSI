from sklearn import metrics
def base_metrics(df, lem_most_com):
    output_dict = dict()
    output_dict['tot_acc'] = metrics.accuracy_score(df['gold'], df['pred'])
    output_dict['tot_f1'] = metrics.accuracy_score(df['gold'], df['pred'])
    output_dict['per_acc'] = df.groupby('lemma').apply(lambda x: metrics.accuracy_score(x['gold'], x['pred']))
    output_dict['baseline'] = df.groupby('lemma').apply(lambda x: metrics.accuracy_score(x['gold'], [lem_most_com[x['lemma'].iloc[0]]] * len(x)))
    output_dict['beat_baseline'] = output_dict['baseline'].le(output_dict['per_acc'])
    return output_dict




