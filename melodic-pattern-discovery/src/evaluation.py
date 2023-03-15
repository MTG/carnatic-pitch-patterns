from collections import Counter

import numpy as np
import pandas as pd 

def f1_score(p,r):
    return 2*p*r/(p+r) if (p+r != 0) else 0

def load_annotations(annotations_path):

    annotations_orig = pd.read_csv(annotations_path, sep='\t')
    annotations_orig.columns = ['s1', 's2', 'duration', 'short_motif', 'motif', 'phrase']
    for c in ['short_motif', 'motif', 'phrase']:
        l1 = len(annotations_orig)

        # ensure_dir that whitespace is
        annotations_orig[c] = annotations_orig[c].apply(lambda y: y.strip() if pd.notnull(y) else np.nan)   

        # remove phrases that occur once
        one_occ = [x for x,y in Counter(annotations_orig[c].values).items() if y == 1]
        annotations_orig = annotations_orig[~annotations_orig[c].isin(one_occ)]
        l2 = len(annotations_orig)
        print(f'    - {l1-l2} {c}s removed from annotations for only occurring once')

    annotations_orig['s1'] = pd.to_datetime(annotations_orig['s1']).apply(lambda y: y.time())
    annotations_orig['s2'] = pd.to_datetime(annotations_orig['s2']).apply(lambda y: y.time())

    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)

    annotations_orig['tier'] = annotations_orig[['short_motif', 'motif', 'phrase']].apply(pd.Series.first_valid_index, axis=1)
    annotations_orig['text'] = [annotations_orig.loc[k, v] if v is not None else None for k, v in annotations_orig['tier'].iteritems()]

    annotations_orig = annotations_orig[['tier', 's1', 's2', 'text']]

    return annotations_orig


def load_annotations_brindha(annotations_path, min_m=None, max_m=None):

    annotations_orig = pd.read_csv(annotations_path, sep='\t')
    annotations_orig.columns = ['tier', 'not_used', 's1', 's2', 'duration', 'text']
    
    annotations_orig['s1'] = pd.to_datetime(annotations_orig['s1']).apply(lambda y: y.time())
    annotations_orig['s2'] = pd.to_datetime(annotations_orig['s2']).apply(lambda y: y.time())
    annotations_orig['duration'] = pd.to_datetime(annotations_orig['duration']).apply(lambda y: y.time())

    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['duration'] = annotations_orig['duration'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)

    if min_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)>min_m]
    if max_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)<max_m]

    annotations_orig = annotations_orig[annotations_orig['tier'].isin(['underlying_full_phrase','underlying_sancara'])]
    good_text = [k for k,v in Counter(annotations_orig['text']).items() if v>1]
    annotations_orig = annotations_orig[annotations_orig['text'].isin(good_text)]


    annotations_orig = annotations_orig[annotations_orig['s2']- annotations_orig['s1']>=1]
    
    return annotations_orig[['tier', 's1', 's2', 'text']]


def is_match_v2(sp, lp, sa, ea, partial_perc=0.3):

    ep = sp + lp
    
    # partial if identified pattern captures a
    # least <partial_perc> of annotation
    la = (ea-sa) # length of annotation

    overlap = 0

    # pattern starts in annotation
    if (sa <= sp <= ea):
        if ep < ea:
            overlap = (ep-sp)
        else:
            overlap = (ea-sp)

    # pattern ends in annotation
    if (sa <= ep <= ea):
        if sa < sp:
            overlap = (ep-sp)
        else:
            overlap = (ep-sa)

    # if intersection between annotation and returned pattern is 
    # >= <partial_perc> of each its a match!
    if overlap/la >= partial_perc and overlap/lp >= partial_perc:
        return 'match'
    else:
        return None


def evaluate_annotations(annotations_raw, starts, lengths, partial_perc):

    annotations = annotations_raw.copy()
    results_dict = {}
    group_num_dict = {}
    occ_num_dict = {}
    is_matched_arr = []
    for i, seq_group in enumerate(starts):
        length_group  = lengths[i]
        ima = []
        for j, seq in enumerate(seq_group):
            im = 0
            length = length_group[j]
            for ai, (tier, s1, s2, text) in zip(annotations.index, annotations.values):
                matched = is_match_v2(seq, length, s1, s2, partial_perc=partial_perc)
                if matched:
                    im = 1
                    if ai not in results_dict:
                        results_dict[ai] = matched
                        group_num_dict[ai] = i
                        occ_num_dict[ai] = j
            ima = ima + [im]
        is_matched_arr.append(ima)


    annotations['match']     = [results_dict[i] if i in results_dict else 'no match' for i in annotations.index]
    annotations['group_num'] = [group_num_dict[i] if i in group_num_dict else None for i in annotations.index]
    annotations['occ_num']   = [occ_num_dict[i] if i in occ_num_dict else None for i in annotations.index]

    return annotations, is_matched_arr


def evaluate(annotations_raw, starts, lengths, partial_perc):
    annotations, is_matched = evaluate_annotations(annotations_raw, starts, lengths, partial_perc)
    ime = [x for y in is_matched for x in y]
    precision = sum(ime)/len(ime) if ime else 1
    recall = sum(annotations['match']!='no match')/len(annotations)
    f1 = f1_score(precision, recall)
    return recall, precision, f1, annotations



def evaluate_quick(annotations_filt, starts_sec_exc, lengths_sec_exc, eval_tol, partial_perc):
    annotations_tagged = evaluate_annotations(annotations_filt, starts_sec_exc, lengths_sec_exc, eval_tol, partial_perc=partial_perc)
    annotations_motif = annotations_tagged[annotations_tagged['tier']=='motif']
    annotations_phrase = annotations_tagged[annotations_tagged['tier']=='phrase']

    n_patterns = sum([len(g) for g in starts_sec_exc])
    recall_all = sum((annotations_tagged['match']!='no match').values)/len(annotations_tagged)
    recall_motif = sum((annotations_motif['match']!='no match').values)/len(annotations_motif)
    recall_phrase = sum((annotations_phrase['match']!='no match').values)/len(annotations_phrase)

    precision_all = len(annotations_tagged[annotations_tagged['group_num'].notnull()][['group_num', 'occ_num']].drop_duplicates())/n_patterns

    print(f'Recall (All): {round(recall_all,2)}')
    print(f'Precision (All): {round(precision_all,2)}')

    print(f'Recall (Motif): {round(recall_motif,2)}')
    print(f'Recall (Phrase): {round(recall_phrase,2)}')
    
    return annotations_tagged, recall_all, precision_all, recall_motif, recall_phrase