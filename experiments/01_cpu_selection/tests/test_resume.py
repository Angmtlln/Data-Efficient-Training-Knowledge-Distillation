"""Actual tiny-model training test, offline CPU. Not project accuracy results."""
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import torch
from transformers import BertConfig,BertForSequenceClassification,Trainer,TrainingArguments,TrainerCallback,set_seed
from checkpoints import restore,verify
from train import DurableCheckpoint

class SimulatedInterruption(Exception):pass

class Interrupt(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        if state.global_step==4:raise SimulatedInterruption()

class ResumeTest(unittest.TestCase):
    def test_actual_trainer_resume_matches_uninterrupted(self):
        torch.set_num_threads(2)
        data=[dict(input_ids=[1,3+i%5,2],attention_mask=[1,1,1],labels=i%2) for i in range(24)]
        def make(path,callbacks):
            set_seed(71)
            model=BertForSequenceClassification(BertConfig(vocab_size=16,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,num_labels=2))
            args=TrainingArguments(output_dir=str(path),use_cpu=True,max_steps=8,
                per_device_train_batch_size=2,save_strategy='steps',save_steps=2,
                report_to=[],seed=71,data_seed=71,disable_tqdm=True,logging_steps=2)
            return Trainer(model=model,args=args,train_dataset=data,callbacks=callbacks)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            full=make(root/'full',[]);full.train()
            interrupted=make(root/'session1',[DurableCheckpoint(root/'drive',2),Interrupt()])
            with self.assertRaises(SimulatedInterruption):interrupted.train()
            cp=restore(root/'drive',root/'session2')
            self.assertTrue(cp.endswith('checkpoint-4'))
            resumed=make(root/'session2',[DurableCheckpoint(root/'drive',2)])
            resumed.train(resume_from_checkpoint=cp)
            for k,v in full.model.state_dict().items():
                self.assertTrue(torch.allclose(v,resumed.model.state_dict()[k],atol=1e-8,rtol=0),k)
            # Corrupt latest durable checkpoint: the preceding complete copy survives.
            latest=root/'drive/checkpoint-8'
            (latest/'optimizer.pt').write_bytes(b'incomplete')
            self.assertFalse(verify(latest))
            fallback=restore(root/'drive',root/'session3')
            self.assertTrue(fallback.endswith('checkpoint-6'))

    def test_epoch_validation_and_best_checkpoint_after_relocation(self):
        torch.set_num_threads(2)
        data=[dict(input_ids=[1,3+i%5,2],attention_mask=[1,1,1],labels=i%2) for i in range(24)]
        class StopAt15(TrainerCallback):
            def on_save(self,args,state,control,**kwargs):
                if state.global_step==15: raise SimulatedInterruption()
        def make(path,callbacks):
            set_seed(71)
            model=BertForSequenceClassification(BertConfig(vocab_size=16,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,num_labels=2))
            args=TrainingArguments(output_dir=str(path),use_cpu=True,num_train_epochs=2,
                per_device_train_batch_size=2,save_strategy='epoch',eval_strategy='epoch',
                load_best_model_at_end=True,metric_for_best_model='accuracy',save_total_limit=3,
                report_to=[],seed=71,data_seed=71,disable_tqdm=True)
            return Trainer(model=model,args=args,train_dataset=data,eval_dataset=data,
                compute_metrics=lambda out:dict(accuracy=float((out.predictions.argmax(-1)==out.label_ids).mean())),callbacks=callbacks)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            full=make(root/'full',[]);full.train()
            interrupted=make(root/'old',[DurableCheckpoint(root/'drive',5),StopAt15()])
            with self.assertRaises(SimulatedInterruption): interrupted.train()
            checkpoint=restore(root/'drive',root/'new')
            self.assertTrue(checkpoint.endswith('checkpoint-15'))
            self.assertTrue((root/'new/checkpoint-12').exists())
            resumed=make(root/'new',[DurableCheckpoint(root/'drive',5)])
            resumed.train(resume_from_checkpoint=checkpoint)
            for k,v in full.model.state_dict().items():
                self.assertTrue(torch.allclose(v,resumed.model.state_dict()[k],atol=1e-8,rtol=0),k)

if __name__=='__main__':unittest.main()
