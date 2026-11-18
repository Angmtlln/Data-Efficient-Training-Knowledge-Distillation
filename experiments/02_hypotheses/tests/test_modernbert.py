"""Offline architecture smoke test, random tiny weights; not an accuracy experiment."""
import unittest
import torch
from transformers import ModernBertConfig,ModernBertForSequenceClassification

class ModernBertSmoke(unittest.TestCase):
    def test_sdpa_forward_backward(self):
        torch.set_num_threads(2)
        config=ModernBertConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,
            num_attention_heads=2,intermediate_size=32,max_position_embeddings=64,num_labels=2,pad_token_id=0,bos_token_id=1,eos_token_id=2,cls_token_id=1,sep_token_id=2,
            attn_implementation='sdpa')
        model=ModernBertForSequenceClassification(config)
        out=model(input_ids=torch.tensor([[1,5,2],[1,8,2]]),attention_mask=torch.ones(2,3,dtype=torch.long),labels=torch.tensor([0,1]))
        out.loss.backward()
        self.assertEqual(tuple(out.logits.shape),(2,2))
        self.assertTrue(torch.isfinite(out.loss))

if __name__=='__main__':unittest.main()
