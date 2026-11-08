from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from par_repro.layer_skip import skip_layers

class LayerSkipTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(5)
        cfg=LlamaConfig(vocab_size=23,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,pad_token_id=0,
            bos_token_id=1,eos_token_id=None)
        cfg._attn_implementation='eager'
        return LlamaForCausalLM(cfg).eval()

    def test_only_retained_blocks_execute_and_restoration_is_exact(self):
        model=self.model();original=model.model.layers;events=[]
        handles=[block.register_forward_pre_hook(lambda m,a,i=i:events.append(i)) for i,block in enumerate(original)]
        with skip_layers(model,[2],candidate_start=2) as retained:
            self.assertEqual(retained,[0,1,3])
            model(torch.tensor([[1,4,8]]),use_cache=False)
            self.assertEqual(events,[0,1,3])
            self.assertEqual([block.self_attn.layer_idx for block in model.model.layers],[0,1,2])
        self.assertIs(model.model.layers,original)
        self.assertEqual([block.self_attn.layer_idx for block in original],[0,1,2,3])
        for handle in handles:handle.remove()

    def test_exception_restores_layers(self):
        model=self.model();original=model.model.layers
        with self.assertRaises(RuntimeError):
            with skip_layers(model,[3],candidate_start=2):raise RuntimeError('test')
        self.assertIs(model.model.layers,original)

    def test_cache_and_uncached_generation_agree(self):
        model=self.model();tokens=torch.tensor([[1,4,8]])
        with torch.inference_mode(),skip_layers(model,[2],candidate_start=2):
            cached=model.generate(tokens,max_new_tokens=4,do_sample=False,use_cache=True)
            uncached=model.generate(tokens,max_new_tokens=4,do_sample=False,use_cache=False)
        torch.testing.assert_close(cached,uncached)

    def test_early_or_duplicate_layer_is_rejected(self):
        model=self.model()
        for ids in [[0],[2,2],[4]]:
            with self.assertRaises(ValueError):
                with skip_layers(model,ids,candidate_start=2):pass

if __name__=='__main__':unittest.main()
