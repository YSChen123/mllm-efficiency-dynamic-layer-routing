import random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from rng_compat import restore_rng_state

class RNGCompatibilityTest(unittest.TestCase):
    def test_own_trainer_numpy_python_and_torch_rng_restore_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rng_state_0.pth'
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            state={'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.random.get_rng_state()}
            torch.save(state,path)
            expected=(random.random(),np.random.random(),torch.rand(5))
            random.seed(99);np.random.seed(99);torch.manual_seed(99)
            restore_rng_state(path)
            self.assertEqual(random.random(),expected[0]);self.assertEqual(np.random.random(),expected[1]);self.assertTrue(torch.equal(torch.rand(5),expected[2]))

if __name__=='__main__':unittest.main()
