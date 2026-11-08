import math,unittest
import torch
from par_repro.pruning import noisy_actions,balanced_indices,balanced_loss,action_mask
from par_repro._paper_router import PaperRouter,PaperRouterConfig

class PaperV3Tests(unittest.TestCase):
    def test_noise_is_variance10_before_softmax_and_joint_actions(self):
        heads=torch.randn((4,590),generator=torch.Generator().manual_seed(8));start=7
        actions=noisy_actions(heads,start,11);generator=torch.Generator().manual_seed(11)
        for action in actions:
            scores=(heads+math.sqrt(10)*torch.randn(heads.shape,generator=generator)).softmax(-1).mean(0)[start:start+576]
            self.assertEqual(action['tokens'],torch.topk(scores,400,largest=False).indices.sort().values.tolist())
            self.assertEqual(int(action_mask(action,400).sum()),404)
        self.assertGreater(len({tuple(a['layers']) for a in actions}),1)
        self.assertGreater(len({tuple(a['tokens']) for a in actions}),1)
        self.assertEqual(actions,noisy_actions(heads,start,11))

    def test_balanced_sets_use_two_extremes_and_reject_cross_ties(self):
        self.assertEqual(balanced_indices([4.,0.,3.,1.,2.]),([1,3],[2,0]))
        self.assertIsNone(balanced_indices([0.,1.,1.,1.,2.]))
        with self.assertRaises(ValueError):balanced_indices([0.,1.,2.,float('nan'),4.])

    def test_all_four_DPO_pairs_match_independent_bernoulli(self):
        logits=torch.tensor([[.3,-1.,2.],[-.7,.6,.1]],dtype=torch.float64,requires_grad=True)
        positive=torch.tensor([[[1,0,0],[0,1,0]],[[0,0,1],[1,1,0]]],dtype=torch.float64)
        negative=1-positive
        def logp(actions):return (actions*torch.nn.functional.logsigmoid(logits[:,None,:])+(1-actions)*torch.nn.functional.logsigmoid(-logits[:,None,:])).sum(-1)
        margin=logp(positive)[:,:,None]-logp(negative)[:,None,:]
        expected=-torch.nn.functional.logsigmoid(margin).mean()
        actual=balanced_loss(logits,positive,negative)
        torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0],rtol=1e-14,atol=1e-14)

    def test_layer_only_actions_keep_all_visual_tokens(self):
        action={'layers':[16,19,25,31],'tokens':[]}
        mask=action_mask(action,0);self.assertEqual(int(mask.sum()),4);self.assertEqual(int(mask[:576].sum()),0)
        with self.assertRaises(ValueError):action_mask(action,400)

    def test_nonlinear_head_and_bidirectional_question_interaction(self):
        torch.manual_seed(0)
        cfg=PaperRouterConfig(input_size=16,hidden_size=16,intermediate_size=32,num_heads=4,head_hidden_size=8)
        model=PaperRouter(cfg);linear=[m for m in model.score.modules() if isinstance(m,torch.nn.Linear)]
        self.assertEqual(len(linear),2);self.assertTrue(any(isinstance(m,torch.nn.GELU) for m in model.score.modules()))
        visual=torch.randn(1,576,16);question=torch.randn(1,3,16)
        first=model(visual,question);second=model(visual,question+3.)
        self.assertEqual(first.shape,(1,592));self.assertGreater(float((first[:,:576]-second[:,:576]).abs().max()),0)
        torch.testing.assert_close(model.probabilities(visual,question),first.sigmoid())

if __name__=='__main__':unittest.main()
