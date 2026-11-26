from pathlib import Path
import json
import os
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[2]/'.hf-cache'))
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from common import ROOT,read_data,dump

if __name__=='__main__':
    ds,meta=read_data()
    config=json.loads((ROOT/'config.json').read_text())
    model=config['model']
    revision=config['model_revision']
    tok=AutoTokenizer.from_pretrained(model,revision=revision)
    tok.save_pretrained(str(ROOT/'data/tokenizer'))
    encoded=ds.map(lambda batch:tok(batch['sentence'],truncation=True,max_length=config['training']['max_length']),batched=True,
                   remove_columns=['sentence','orig_idx'])
    if 'pool_idx' in encoded['pool'].column_names: encoded['pool']=encoded['pool'].remove_columns('pool_idx')
    encoded.save_to_disk(str(ROOT/'data/tokenized'))
    dump(ROOT/'data/tokenization.json',dict(model=model,model_revision=revision,max_length=config['training']['max_length'],snapshot=meta))
    print('Tokenization complete; model weights will download in Colab:',revision)
