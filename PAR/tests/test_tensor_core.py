import copy
import unittest
import torch
from par_repro._paper_router import PaperRouter, PaperRouterConfig
from par_repro.router import ControlledRouter
from par_repro.batching import epoch_batches, accumulated_update


class ControlledSettingsTests(unittest.TestCase):
    def config(self):
        return PaperRouterConfig(input_size=16, hidden_size=16, intermediate_size=32,
                                 num_heads=4, head_hidden_size=8)

    def test_bidirectional_control_bitwise_equal(self):
        torch.manual_seed(7)
        old = PaperRouter(self.config()).eval()
        new = ControlledRouter(self.config(), 'bidirectional').eval()
        new.load_state_dict(old.state_dict(), strict=True)
        v, q = torch.randn(2, 6, 16), torch.randn(2, 5, 16)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        self.assertTrue(torch.equal(old(v, q, mask), new(v, q, mask)))

    def test_causal_visual_cannot_read_later_question(self):
        torch.manual_seed(7)
        causal = ControlledRouter(self.config(), 'causal').eval()
        bidirectional = ControlledRouter(self.config(), 'bidirectional').eval()
        bidirectional.load_state_dict(causal.state_dict())
        v, q = torch.randn(1, 6, 16), torch.randn(1, 5, 16)
        changed = q + torch.randn_like(q) * 10
        self.assertTrue(torch.equal(causal(v, q)[:, :6], causal(v, changed)[:, :6]))
        self.assertFalse(torch.equal(bidirectional(v, q)[:, :6], bidirectional(v, changed)[:, :6]))
        self.assertFalse(torch.equal(causal(v, q)[:, 6:], causal(v, changed)[:, 6:]))

    def test_global16_last_batch_keeps_all5000(self):
        batches = epoch_batches(0, 16)
        self.assertEqual(len(batches), 313)
        self.assertEqual(len(batches[-1]), 8)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(5000)))
        self.assertEqual([i for b in epoch_batches(0, 4) for i in b], [i for b in batches for i in b])
        self.assertNotEqual(epoch_batches(0, 16), epoch_batches(1, 16))

    def test_accumulation_weights_short_microbatch(self):
        torch.manual_seed(3)
        a = torch.nn.Linear(3, 1).double()
        b = copy.deepcopy(a)
        x, y = torch.randn(10, 3).double(), torch.randn(10, 1).double()
        oa, ob = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
        oa.zero_grad()
        loss = ((a(x) - y) ** 2).mean()
        loss.backward()
        oa.step()
        actual = accumulated_update(b, ob, list(range(10)), lambda ids: ((b(x[ids]) - y[ids]) ** 2).mean(), 4)
        self.assertAlmostEqual(float(loss), actual, places=12)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)

    def test_invalid_modes_and_batches_fail(self):
        with self.assertRaises(ValueError):
            ControlledRouter(self.config(), 'typo')
        with self.assertRaises(ValueError):
            epoch_batches(0, 0)


if __name__ == '__main__':
    unittest.main()
