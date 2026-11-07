import unittest
import torch
from roe_repro.precision_retry import guarded_generate


class PrecisionRetryTests(unittest.TestCase):
    def model(self):
        m=torch.nn.Linear(2,1,bias=False).half().eval()
        with torch.no_grad():m.weight.fill_(500)
        m.register_buffer('cache',torch.tensor([0.1234567],dtype=torch.float32))
        return m

    def test_actual_half_overflow_recovers_and_restores_weights_and_buffers(self):
        model=self.model();weight=model.weight.detach().clone();cache=model.cache.clone()
        calls=[];retries=[]
        def call():
            calls.append(model.weight.dtype)
            value=model(torch.full((1,2),500.,dtype=model.weight.dtype))
            if not torch.isfinite(value).all():raise RuntimeError('Nonfinite FP16 router logits at layer 31; no fallback or sample skipping')
            model.cache.add_(1)
            return value
        value,retried=guarded_generate(model,call,lambda error:retries.append(str(error)))
        self.assertTrue(retried);self.assertEqual(value.item(),500000.)
        self.assertEqual(calls,[torch.float16,torch.float32]);self.assertEqual(len(retries),1)
        self.assertEqual(model.weight.dtype,torch.float16)
        torch.testing.assert_close(model.weight,weight,rtol=0,atol=0)
        torch.testing.assert_close(model.cache,cache,rtol=0,atol=0)

    def test_non_numeric_exception_is_not_retried(self):
        calls=[]
        def call():calls.append(1);raise RuntimeError('KV cache shape mismatch')
        with self.assertRaisesRegex(RuntimeError,'cache shape'):
            guarded_generate(self.model(),call,lambda error:self.fail('must not retry'))
        self.assertEqual(calls,[1])

    def test_failed_float_retry_also_restores_dtype(self):
        model=self.model();calls=[]
        def call():
            calls.append(model.weight.dtype)
            raise RuntimeError('Nonfinite final-position FP16 logits; no fallback or sample skipping')
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            guarded_generate(model,call,lambda error:None)
        self.assertEqual(calls,[torch.float16,torch.float32])
        self.assertEqual(model.weight.dtype,torch.float16)


if __name__=='__main__':unittest.main()
