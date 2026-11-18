from pathlib import Path
import os
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[2]/'.hf-cache'))
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from common import ROOT,read_data,dump

if __name__=='__main__':
    ds,meta=read_data()
    model='answerdotai/ModernBERT-base'
    revision=HfApi().model_info(model).sha
    tok=AutoTokenizer.from_pretrained(model,revision=revision)
    tok.save_pretrained(str(ROOT/'data/tokenizer'))
    encoded=ds.map(lambda batch:tok(batch['sentence'],truncation=True,max_length=64),batched=True,
                   remove_columns=['sentence','orig_idx'])
    if 'pool_idx' in encoded['pool'].column_names: encoded['pool']=encoded['pool'].remove_columns('pool_idx')
    encoded.save_to_disk(str(ROOT/'data/tokenized'))
    dump(ROOT/'data/tokenization.json',dict(model=model,model_revision=revision,max_length=64,snapshot=meta))
    dump(ROOT/'config.json',dict(model=model,model_revision=revision,training=dict(epochs=3,batch_size=32,
         learning_rate=5e-5,weight_decay=.01,warmup_ratio=.06,max_length=64,checkpoint_steps=100)))
    print('Tokenization complete; model weights will download in Colab:',revision)
