"""Train one prepared subset. Local fast checkpoints + verified Drive replicas."""
import argparse
import json
import os
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score,f1_score
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,Trainer,TrainingArguments,TrainerCallback,set_seed
from common import ROOT,dump,digest,read_data,environment
from checkpoints import publish,restore,prune


class DurableCheckpoint(TrainerCallback):
    def __init__(self,persistent,interval): self.persistent=Path(persistent);self.interval=interval
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step % self.interval == 0: control.should_save=True
        return control
    def on_save(self,args,state,control,**kwargs):
        checkpoint=Path(args.output_dir)/f'checkpoint-{state.global_step}'
        publish(checkpoint,self.persistent)
        prune(self.persistent,state.best_model_checkpoint)
        dump(self.persistent.parent/'progress.json',dict(step=state.global_step,epoch=state.epoch,log_history=state.log_history))
        return control


def metrics(pred):
    labels=pred.label_ids; predicted=np.argmax(pred.predictions,axis=-1)
    return dict(accuracy=accuracy_score(labels,predicted),macro_f1=f1_score(labels,predicted,average='macro'),
                positive_f1=f1_score(labels,predicted))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',required=True)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--persistent',required=True)
    p.add_argument('--local',required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--cpu',action='store_true')
    a=p.parse_args()
    if not torch.cuda.is_available() and not a.cpu: raise RuntimeError('GPU unavailable. Use --cpu only for intentional CPU tests.')
    config=json.loads((ROOT/'config.json').read_text())
    selection=json.loads(Path(a.selection).read_text())
    expected=selection.pop('sha256');assert digest(selection)==expected;selection['sha256']=expected
    ds,meta=read_data()
    if selection['pool_sha256']!=meta['splits']['pool']['sha256']: raise ValueError('Selection/dataset mismatch')
    ids=selection['indices'];assert len(ids)==len(set(ids))==6534 and min(ids)>=0 and max(ids)<len(ds['pool'])
    tmeta=json.loads((ROOT/'data/tokenization.json').read_text())
    assert tmeta['model_revision']==config['model_revision'] and tmeta['snapshot']==meta
    training=dict(config['training'])
    if a.pilot: training.update(max_steps=20,eval_steps=10,save_steps=10)
    code_hash=digest({f.name:f.read_text() for f in (ROOT/'code').glob('*.py')})
    identity=dict(selection_sha256=expected,config=config,training=training,seed=a.seed,
                  code_hash=code_hash,pilot=a.pilot)
    runid=f"{selection['name']}__ss{selection['selection_seed']}__ts{a.seed}__{digest(identity)[:12]}"
    persistent=Path(a.persistent)/('pilots' if a.pilot else 'runs')/runid
    local=Path(a.local)/runid
    persistent.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    if (persistent/'metrics.json').exists(): print('Already completed:',runid);return
    dump(persistent/'identity.json',identity)
    dump(persistent/'environment.json',environment())
    set_seed(a.seed)
    tokenized=load_from_disk(str(ROOT/'data/tokenized'))
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'data/tokenizer'),local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(config['model'],revision=config['model_revision'],num_labels=2,attn_implementation='sdpa')
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    kwargs=dict(output_dir=str(local),num_train_epochs=training['epochs'],
        per_device_train_batch_size=training['batch_size'],per_device_eval_batch_size=32,
        learning_rate=training['learning_rate'],weight_decay=training['weight_decay'],
        warmup_ratio=training['warmup_ratio'],seed=a.seed,data_seed=a.seed,
        eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='accuracy',greater_is_better=True,save_total_limit=3,
        logging_steps=20,report_to=[],bf16=bf16,fp16=torch.cuda.is_available() and not bf16,
        use_cpu=a.cpu,dataloader_num_workers=0,optim='adamw_torch',save_safetensors=True)
    if a.pilot: kwargs.update(max_steps=20,eval_strategy='steps',eval_steps=10,save_strategy='steps',save_steps=10)
    trainer=Trainer(model=model,args=TrainingArguments(**kwargs),
        train_dataset=tokenized['pool'].select(ids),eval_dataset=tokenized['val'],
        processing_class=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics,
        callbacks=[DurableCheckpoint(persistent/'checkpoints',10 if a.pilot else training['checkpoint_steps'])])
    checkpoint=restore(persistent/'checkpoints',local)
    started=time.time()
    print('Run:',runid,'resume:',checkpoint,flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    # Held-out test evaluated only after training, with val-selected weights.
    evaluated=tokenized['val'] if a.pilot else tokenized['test']
    pred=trainer.predict(evaluated)
    probabilities=torch.tensor(pred.predictions).softmax(-1).numpy()
    import pandas as pd
    pd.DataFrame(dict(label=pred.label_ids,prediction=probabilities.argmax(-1),p_positive=probabilities[:,1])).to_csv(persistent/'predictions.csv',index=False)
    result=dict(name=selection['name'],selection_seed=selection['selection_seed'],training_seed=a.seed,
        metrics=metrics(pred),last_session_seconds=time.time()-started,global_step=trainer.state.global_step,
        best_checkpoint=trainer.state.best_model_checkpoint,selection_sha256=expected,
        split='val_pilot' if a.pilot else 'test',pilot=a.pilot,
        log_history=trainer.state.log_history)
    dump(persistent/'metrics.json',result)
    # Completed runs resume by skipping; optimizer checkpoints are no longer needed.
    if not config.get('keep_completed_checkpoints', False):
        if (persistent/'checkpoints').exists(): shutil.rmtree(persistent/'checkpoints')
        if local.exists(): shutil.rmtree(local)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
