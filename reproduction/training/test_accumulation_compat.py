import contextlib,unittest
from accumulation_compat import compatible_no_sync

class PartitionedEngine:
    def zero_optimization_partition_gradients(self):return True
    def no_sync(self):raise AssertionError('no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 3')

class AccumulationCompatibilityTest(unittest.TestCase):
    def test_partitioned_zero_uses_own_accumulation_without_no_sync(self):
        def original(accelerator,model):return model.no_sync()
        with compatible_no_sync(original,object(),PartitionedEngine()):pass
    def test_dense_ddp_retains_original_no_sync(self):
        calls=[]
        @contextlib.contextmanager
        def original(accelerator,model):
            calls.append((accelerator,model));yield
        accelerator=object();model=object()
        with compatible_no_sync(original,accelerator,model):pass
        self.assertEqual(calls,[(accelerator,model)])

if __name__=='__main__':unittest.main()
