"""Broader CPU candidate mining; NO automatic acceptance or training selection."""
import argparse,sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/02_hypotheses/code'))
from pairs import mine_pairs,disjoint_pairs
from datasets import load_from_disk
import pandas as pd

OPERATORS={'not','never','hardly','without','lacks','lacking','devoid','fails','failed','cannot',"n't",'unable','no','barely','neither','nothing','less'}
POS={'good','great','excellent','wonderful','engaging','entertaining','funny','moving','effective','compelling','enjoyable','smart','beautiful','best','better','interesting','memorable','original'}
NEG={'bad','terrible','awful','boring','dull','unfunny','tedious','pointless','ineffective','unconvincing','forgettable','stupid','ugly','worst','worse','uninteresting','unoriginal','predictable'}

def category(e):
    removed=e['removed'].split();added=e['added'].split()
    if not removed or not added:
        span=removed or added
        if set(span)&OPERATORS:return 'operator_insertion_deletion'
    if (set(removed)&POS and set(added)&NEG) or (set(removed)&NEG and set(added)&POS):return 'polarity_word_substitution'
    if (set(removed)|set(added))&OPERATORS:return 'operator_phrase_rewrite'
    return 'other_opposite_label_edit'

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-span',type=int,default=2);a=p.parse_args()
    ds=load_from_disk(str(a.dataset));pool=ds['pool'];texts=list(pool['sentence']);labels=list(pool['label'])
    edges,stats=mine_pairs(texts,labels,max_span=a.max_span,min_context=4)
    for e in edges:
        e['category']=category(e);e['review_status']='unreviewed';e['accepted']=False
        e['orig_idx_a']=int(pool[e['a']]['orig_idx']);e['orig_idx_b']=int(pool[e['b']]['orig_idx'])
    a.output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(edges);frame.to_csv(a.output/'candidates.csv',index=False)
    stats.update(disjoint_all=len(disjoint_pairs(edges,17)),by_category={c:{'candidates':len(g),'disjoint':len(disjoint_pairs(g.to_dict('records'),17))} for c,g in frame.groupby('category')},
      note='All unreviewed. Opposite gold labels do not certify causal/semantic validity. No new training subsets created.')
    (a.output/'stats.json').write_text(json.dumps(stats,indent=2))
    # Review queue covers each category and contains original gold labels, not generated text.
    frame.groupby('category',group_keys=False).apply(lambda g:g.sample(min(30,len(g)),random_state=17)).to_csv(a.output/'review_sample.csv',index=False)
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
