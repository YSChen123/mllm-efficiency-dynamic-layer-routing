from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from par_repro.math_core import sample_layer_actions,kl_full_to_pruned,preference_extremes,action_log_prob,dpo_loss

class PARCoreTests(unittest.TestCase):
    def test_actions_have_exact_budget_and_last16_only(self):
        actions=sample_layer_actions(seed=3)
        self.assertEqual(len(actions),5)
        self.assertEqual(len({tuple(x) for x in actions}),5)
        self.assertTrue(all(len(x)==4 and all(16<=i<32 for i in x) for x in actions))
        self.assertEqual(actions,sample_layer_actions(seed=3))

    def test_kl_is_directed_and_zero_for_identity(self):
        p=torch.tensor([[2.,-1.]])
        q=torch.tensor([[0.,0.]])
        self.assertEqual(kl_full_to_pruned(p,p).item(),0.)
        expected=(p.softmax(-1)*(p.log_softmax(-1)-q.log_softmax(-1))).sum()
        torch.testing.assert_close(kl_full_to_pruned(p,q),expected)
        self.assertNotAlmostEqual(kl_full_to_pruned(q,p).item(),expected.item())

    def test_best_worst_and_tie_are_explicit(self):
        self.assertEqual(preference_extremes([.2,.1,.5,.3,.4]),(1,2))
        self.assertIsNone(preference_extremes([.1]*5))
        with self.assertRaises(ValueError):preference_extremes([float('nan')]*5)

    def test_reference_free_loss_and_gradient_direction(self):
        logits=torch.zeros(1,4,requires_grad=True)
        chosen=torch.tensor([[1,0,0,0]])
        rejected=torch.tensor([[0,1,0,0]])
        loss=dpo_loss(action_log_prob(logits,chosen),action_log_prob(logits,rejected))
        self.assertAlmostEqual(loss.item(),0.69314718,6)
        loss.backward()
        self.assertLess(logits.grad[0,0],0)
        self.assertGreater(logits.grad[0,1],0)

    def test_extreme_logits_are_finite(self):
        logits=torch.tensor([[1000.,-1000.]])
        lp=action_log_prob(logits,torch.tensor([[0,1]]))
        self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue(torch.isfinite(dpo_loss(lp,-lp)).all())

if __name__=='__main__':unittest.main()
