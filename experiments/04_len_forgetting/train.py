"""Run one fixed-protocol ModernBERT experiment; supports CUDA and Apple MPS."""
import argparse
import csv
import fcntl
import json
import os
import platform
from pathlib import Path
import sys
import time
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0, str(REPO/'experiments/01_cpu_selection/code'))
from common import digest, read_data


def metrics(pred):
    y=pred.label_ids; p=np.argmax(pred.predictions,axis=-1)
    return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,average='macro')),
            'positive_f1':float(f1_score(y,p))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--training-seed',type=int,required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--output',type=Path,default=REPO/'results/len_forgetting/runs')
    p.add_argument('--checkpoint-root',type=Path,default=REPO/'outputs/len_forgetting/checkpoints')
    p.add_argument('--cpu',action='store_true')
    a=p.parse_args()
    if not a.cpu and not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        raise RuntimeError('No CUDA or MPS GPU visible. Explicit --cpu required.')
    selection=json.loads(a.selection.read_text()); expected=selection.pop('sha256')
    if digest(selection)!=expected: raise ValueError('Selection checksum mismatch')
    ds,meta=read_data()
    if selection['pool_sha256']!=meta['splits']['pool']['sha256']: raise ValueError('Pool hash mismatch')
    ids=selection['indices']
    if len(ids)!=6534 or len(set(ids))!=6534: raise ValueError('Invalid selection size')
    cfg=json.loads((REPO/'experiments/01_cpu_selection/config.json').read_text())
    tok_meta=json.loads((REPO/'experiments/01_cpu_selection/data/tokenization.json').read_text())
    if tok_meta['snapshot']!=meta or tok_meta['model_revision']!=cfg['model_revision']:
        raise ValueError('Tokenized data/model version mismatch')
    data=load_from_disk(str(REPO/'experiments/01_cpu_selection/data/tokenized'))
    runid=f"{selection['name']}__ss{selection['selection_seed']}__ts{a.training_seed}"
    if a.pilot: runid+='__pilot'
    result_dir=a.output/runid; checkpoint_dir=a.checkpoint_root/runid
    lock_dir=REPO/'outputs/len_forgetting/locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file=(lock_dir/f'{runid}.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    result_dir.mkdir(parents=True,exist_ok=True)
    if (result_dir/'metrics.json').exists():
        saved=json.loads((result_dir/'metrics.json').read_text())
        if saved['selection_sha256'] != expected: raise ValueError('Existing run uses different selection')
        print('Already complete',runid,flush=True);return
    set_seed(a.training_seed)
    tokenizer=AutoTokenizer.from_pretrained(str(REPO/'experiments/01_cpu_selection/data/tokenizer'),local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(cfg['model'], revision=cfg['model_revision'],
                  num_labels=2, attn_implementation='sdpa')
    gpu='cuda' if torch.cuda.is_available() and not a.cpu else ('mps' if torch.backends.mps.is_available() and not a.cpu else 'cpu')
    cfgtrain=cfg['training']
    kwargs=dict(output_dir=str(checkpoint_dir), num_train_epochs=cfgtrain['epochs'],
        per_device_train_batch_size=cfgtrain['batch_size'], per_device_eval_batch_size=32,
        learning_rate=cfgtrain['learning_rate'],weight_decay=cfgtrain['weight_decay'],
        warmup_ratio=cfgtrain['warmup_ratio'], seed=a.training_seed,data_seed=a.training_seed,
        eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='accuracy',greater_is_better=True,save_total_limit=2,
        logging_steps=20,report_to=[],bf16=gpu=='cuda' and torch.cuda.is_bf16_supported(),
        fp16=gpu=='cuda' and not torch.cuda.is_bf16_supported(),use_cpu=a.cpu,
        dataloader_num_workers=0,dataloader_pin_memory=False,optim='adamw_torch',save_safetensors=True)
    if a.pilot: kwargs.update(max_steps=20,eval_strategy='steps',eval_steps=10,save_strategy='steps',save_steps=10)
    trainer=Trainer(model=model,args=TrainingArguments(**kwargs),train_dataset=data['pool'].select(ids),
        eval_dataset=data['val'], processing_class=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
    last=get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    started=time.time();print('RUN',runid,'device',gpu,'resume',last,flush=True)
    trainer.train(resume_from_checkpoint=last)
    evaluated='val' if a.pilot else 'test'; pred=trainer.predict(data[evaluated])
    probs=torch.tensor(pred.predictions).softmax(-1).numpy(); labels=pred.label_ids
    with (result_dir/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['orig_idx','label','prediction','p_positive'])
        w.writerows(zip(ds[evaluated]['orig_idx'],labels,probs.argmax(axis=1),probs[:,1]))
    result={'name':selection['name'],'selection_seed':selection['selection_seed'],'training_seed':a.training_seed,
        'selection_sha256':expected,'metrics':metrics(pred),'split':evaluated,'pilot':a.pilot,
        'seconds':time.time()-started,'global_step':trainer.state.global_step,
        'best_checkpoint':trainer.state.best_model_checkpoint,'best_metric':trainer.state.best_metric,
        'device':gpu,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
        'transformers':__import__('transformers').__version__,'model_revision':cfg['model_revision'],
        'log_history':trainer.state.log_history}
    (result_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
