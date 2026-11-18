"""Small, explicitly labeled negation-contrast feasibility experiment."""
import json
import time
import pandas as pd
from common import ROOT,read_data,dump,manifest
from pairs import mine_pairs,disjoint_pairs,build_selections

OPERATORS={'not','never','hardly','without','lacks','lacking','devoid','fails','failed','cannot',"n't",'unable'}
# Transparent exclusions from assistant review of the 43 retrieved candidates.
# No val/test outcomes were consulted. This is not an independent human annotation.
AMBIGUOUS={(35913,32211):'absence of vulgarity is not necessarily positive sentiment',
           (37987,48432):'popularity comparison does not directly establish sentiment',
           (21897,12068):'pulse racing can be positive or negative without context'}


def main():
    start=time.time();ds,meta=read_data()
    texts,labels=list(ds['pool']['sentence']),list(ds['pool']['label'])
    edges,stats=mine_pairs(texts,labels,max_span=1)
    dump(ROOT/'data/pair_candidates.json',edges)
    reviewed=[];candidates=[]
    for e in edges:
        is_operator=(not e['removed'] and e['added'] in OPERATORS) or (not e['added'] and e['removed'] in OPERATORS)
        if not is_operator: continue
        reason=AMBIGUOUS.get((e['a'],e['b']))
        reviewed.append(dict(e,reviewer='assistant, not independent human',
            verdict='ambiguous_excluded' if reason else 'plausible_contrast',
            reason=reason or 'Explicit negation/absence changes the interpretation of shared context; fragment caveat remains'))
        if not reason: candidates.append(e)
    pd.DataFrame(reviewed).to_csv(ROOT/'reports/pair_operator_audit.csv',index=False)
    pd.DataFrame(edges).sample(min(100,len(edges)),random_state=42).to_csv(ROOT/'reports/broad_pair_audit.csv',index=False)
    stats.update(operator_candidates=len(reviewed),reviewed_candidates=len(candidates),
                 disjoint_reviewed=len(disjoint_pairs(candidates,17)),target_pairs=32,
                 pair_members=64,actual_fraction=64/6534,
                 finding='20% pair-member target is infeasible: 653 disjoint pairs needed, only 444 broad pairs and fewer credible contrasts.',
                 status='small feasibility experiment; no claim of adequate power for overall accuracy')
    dump(ROOT/'reports/mining.json',stats)
    summaries=[]
    for seed in [17,42,101]:
        # Tiny epsilon only avoids float-floor rounding 32 -> 31.
        selections,pairs=build_selections(texts,labels,candidates,6534,(64+1e-6)/6534,seed)
        assert len(pairs)==32
        pd.DataFrame(pairs).to_csv(ROOT/'reports'/f'selected_pairs_s{seed}.csv',index=False)
        for name,ids in selections.items():
            manifest('pair_'+name,ids,seed,meta,dict(pair_count=32,pair_members=64,
                actual_fraction=64/6534,selector='one-token negation/absence contrasts, assistant-reviewed',
                CPU_seconds=time.time()-start,
                limitations='Only 64/6534 rows form contrasts; exploratory. Original labels unchanged.'))
            summaries.append(dict(method='pair_'+name,seed=seed,n=len(ids),positive_share=sum(labels[i] for i in ids)/len(ids),
                                 mean_words=sum(len(texts[i].split()) for i in ids)/len(ids)))
    pd.DataFrame(summaries).to_csv(ROOT/'reports/pair_selection_summary.csv',index=False)
    print(stats)

if __name__=='__main__':main()
