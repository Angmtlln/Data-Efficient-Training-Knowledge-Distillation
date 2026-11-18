import unittest
import numpy as np
import torch
from prepare_skills import gradient_scores

class GradientTest(unittest.TestCase):
    def test_scores_equal_autograd_inner_product(self):
        torch.manual_seed(7)
        h=torch.randn(8,5,dtype=torch.float64)
        y=torch.tensor([0.,1.,1.,0.,1.,0.,0.,1.],dtype=torch.float64)
        w=torch.randn(6,dtype=torch.float64,requires_grad=True)
        x=torch.cat([h,torch.ones(8,1,dtype=torch.float64)],dim=1)
        p=(x@w).sigmoid()
        grads=[]
        for i in range(8):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(x[i]@w,y[i])
            grads.append(torch.autograd.grad(loss,w,retain_graph=True)[0])
        grads=torch.stack(grads);refs=np.array([1,3])
        expected=grads@grads[refs].mean(0)
        actual=gradient_scores(h.numpy(),y.numpy(),p.detach().numpy(),refs)
        np.testing.assert_allclose(actual,expected.numpy(),rtol=1e-10,atol=1e-10)

if __name__=='__main__':unittest.main()
