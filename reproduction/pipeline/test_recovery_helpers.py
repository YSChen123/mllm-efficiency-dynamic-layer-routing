import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('helpers',Path(__file__).with_name('recovery_helpers.py'))
        self.h=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.h)

    def error(self,code,headers):
        e=RuntimeError('HTTP failure')
        e.response=SimpleNamespace(status_code=code,headers=headers)
        return e

    def test_429_waits_until_server_window_resets(self):
        attempts=[]; waits=[]
        def operation():
            attempts.append(1)
            if len(attempts)==1: raise self.error(429,{'RateLimit':'"api";r=0;t=42'})
            return 'prepared'
        self.assertEqual(self.h.retry_hub(operation,sleep=waits.append,emit=lambda message:None),'prepared')
        self.assertEqual(len(attempts),2)
        self.assertGreaterEqual(waits[0],42)

    def test_403_is_not_retried(self):
        waits=[]
        def operation(): raise self.error(403,{})
        with self.assertRaises(RuntimeError):self.h.retry_hub(operation,sleep=waits.append)
        self.assertEqual(waits,[])

    def test_429_retries_are_bounded(self):
        attempts=[]
        def operation():
            attempts.append(1)
            raise self.error(429,{'Retry-After':'10'})
        with self.assertRaises(RuntimeError):
            self.h.retry_hub(operation,attempts=3,sleep=lambda seconds:None,emit=lambda message:None)
        self.assertEqual(len(attempts),3)

    def test_selects_most_free_memory_even_when_gpu_busy(self):
        rows = '0, 97887, 27678, 40\n1, 97887, 24306, 56\n2, 97887, 26866, 0\n'
        self.assertEqual(self.h.select_gpu(rows, minimum_mib=26624)['index'], '0')

    def test_rejects_insufficient_memory(self):
        with self.assertRaises(RuntimeError):
            self.h.select_gpu('0, 97887, 24000, 0\n', minimum_mib=26624)


if __name__=='__main__':unittest.main()
