import sys, unittest, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'code'))
import numpy as np
import torch
from transformers import ModernBertConfig,ModernBertForSequenceClassification,TrainingArguments
from pair_training import make_packs,pair_penalty,PairTrainer,PairCollator,PackedDataset
from prepare import parse

class Tests(unittest.TestCase):
    def test_tree(self):
        root,nodes=parse('(0 (3 good) (0 (2 but) (0 boring)))')
        self.assertEqual(root,(0,['good','but','boring']));self.assertEqual(len(nodes),5)
    def test_packs(self):
        ids=list(range(6534));pairs=[[i,i+1] for i in range(0,262,2)]
        for seed in [1103,1201,1301]:
            packs=make_packs(ids,pairs,seed,True)
            self.assertEqual(len(packs),205);self.assertTrue(all(len(p)==32 for p in packs[:-1]));self.assertEqual(len(packs[-1]),6)
            where={x:i for i,p in enumerate(packs) for x in p}
            self.assertTrue(all(where[a]==where[b] for a,b in pairs))
            self.assertEqual(packs,make_packs(ids,pairs,seed,True))
    def test_gradient(self):
        logits=torch.zeros(2,2,requires_grad=True);pairs=torch.tensor([[0,1]])
        loss=pair_penalty(logits,pairs,.2);loss.backward()
        self.assertLess(logits.grad[0,1].item(),0);self.assertGreater(logits.grad[1,1].item(),0)
        self.assertEqual(pair_penalty(torch.tensor([[0.,8.],[8.,0.]]),pairs,.2).item(),0)
    def test_real_trainer_cpu(self):
        # A tiny randomly initialized ModernBERT checks the Trainer integration only.
        # These fixtures never enter experimental results.
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(Path(__file__).parent/'data/tokenizer'),local_files_only=True)
        rows=[dict(tokenizer('good movie' if i%2 else 'bad movie'),label=i%2) for i in range(64)]
        config=ModernBertConfig(vocab_size=len(tokenizer),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=128,local_attention=32,num_labels=2,pad_token_id=tokenizer.pad_token_id)
        model=ModernBertForSequenceClassification(config);model.config._attn_implementation='eager'
        with tempfile.TemporaryDirectory() as tmp:
            args=TrainingArguments(output_dir=tmp,use_cpu=True,per_device_train_batch_size=1,per_device_eval_batch_size=8,max_steps=2,save_steps=1,logging_steps=1,report_to=[],remove_unused_columns=False,disable_tqdm=True)
            trainer=PairTrainer(model=model,args=args,train_dataset=PackedDataset(rows,list(range(64)),[[0,1],[2,3]],1103,True),eval_dataset=rows[:8],data_collator=PairCollator(tokenizer),rank_weight=.2)
            trainer.train();self.assertEqual(trainer.state.global_step,2)
            pred=trainer.predict(rows[:8]);self.assertEqual(pred.predictions.shape,(8,2));self.assertTrue(np.isfinite(pred.predictions).all())
            self.assertTrue((Path(tmp)/'checkpoint-2/trainer_state.json').exists())
if __name__=='__main__':unittest.main()
