"""Exercise actual release decoder/cache behavior with deterministic small weights."""
import unittest
import types
import torch
from transformers import LlamaConfig
from llava.model.language_model.modeling_llama import LlamaForCausalLM
from llava.train.roe_adapter import Adapter, Skip_Router
from roe_repro import routing_backend as backend


def model_and_input(full):
    torch.manual_seed(17)
    config=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128)
    model=LlamaForCausalLM(config).eval()
    for layer,enabled in zip(model.model.layers,full):
        layer.adapter_MHA=Adapter(16,8)
        layer.adapter_MHA_router=Skip_Router(16)
        with torch.no_grad():
            layer.adapter_MHA.conv_A.weight.zero_()
            layer.adapter_MHA.conv_B.weight.zero_()
            layer.adapter_MHA_router.expert[0].weight.zero_()
            if enabled:layer.adapter_MHA_router.expert[0].weight[0].fill_(0.1)
        layer.weight=None
    inputs=torch.randn(1,6,16)*0.1
    inputs[:,:2,:]=1
    mask=torch.zeros(1,2,6,dtype=torch.long)
    mask[:,0,:3]=1;mask[:,1,3:]=1
    model.eval()
    return model,inputs,mask


class RoutingTests(unittest.TestCase):
    @torch.no_grad()
    def test_first_layer_skip_keeps_length_and_freezes_path_during_decode(self):
        model,inputs,mask=model_and_input([False,True,False])
        counts={'router':[0]*3,'attention':[0]*3,'adapter':[0]*3}
        def counter(kind,i):
            def hook(*args):counts[kind][i]+=1
            return hook
        for i,layer in enumerate(model.model.layers):
            layer.adapter_MHA_router.register_forward_hook(counter('router',i))
            layer.self_attn.register_forward_hook(counter('attention',i))
            layer.adapter_MHA.register_forward_hook(counter('adapter',i))
        backend.install(model)
        out=model(inputs_embeds=inputs,question_mask=mask,use_cache=True)
        for length in [6,7,8]:
            self.assertEqual(out.past_key_values[0][0].shape[2],length)
            self.assertTrue(torch.isfinite(out.logits).all())
            if length<8:
                out=model(input_ids=torch.tensor([[3]]),past_key_values=out.past_key_values,
                    question_mask=mask,use_cache=True)
        self.assertEqual(counts['router'],[1,1,1])
        self.assertEqual(counts['attention'],[0,3,0])
        self.assertEqual(counts['adapter'],[3,0,3])
        model(inputs_embeds=inputs,question_mask=mask,use_cache=True)
        self.assertEqual(counts['router'],[2,2,2])

    @torch.no_grad()
    def test_full_routes_match_untouched_official_forward(self):
        model,inputs,mask=model_and_input([True,True,True])
        for layer in model.model.layers:
            layer.forward=types.MethodType(backend.release.forward_llama,layer)
            layer.adapter_MHA_router.forward=types.MethodType(backend.release.Skip_Router.forward,layer.adapter_MHA_router)
        expected=model(inputs_embeds=inputs,question_mask=mask,use_cache=True)
        backend.install(model)
        actual=model(inputs_embeds=inputs,question_mask=mask,use_cache=True)
        torch.testing.assert_close(actual.logits,expected.logits,rtol=0,atol=0)
        for a,b in zip(actual.past_key_values,expected.past_key_values):
            torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
