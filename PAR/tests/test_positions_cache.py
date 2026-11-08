import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from par_repro.pruning import forward_preserved, generate_preserved


class PhysicalPruningTests(unittest.TestCase):
    def model(self, backend='eager'):
        torch.manual_seed(3)
        cfg = LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                          num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, eos_token_id=None)
        cfg._attn_implementation = backend
        model = LlamaForCausalLM(cfg).eval()
        return model, model.get_input_embeddings()(torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 10]]))

    def test_full_logits_match_hf_on_both_backends(self):
        for backend in ['eager', 'sdpa']:
            model, embeds = self.model(backend)
            with torch.inference_mode():
                expected = model(inputs_embeds=embeds, use_cache=False).logits
                actual = forward_preserved(model, embeds, 2, [], [], visual_tokens=5, candidate_start=2)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_compression_and_original_cache_timeline(self):
        model, embeds = self.model()
        _, cache, trace = forward_preserved(model, embeds, 2, [3], [0, 2, 4], use_cache=True, visual_tokens=5, candidate_start=2)
        self.assertEqual(trace, [(0, 9), (1, 9), (2, 6)])
        self.assertEqual([cache.get_seq_length(i) for i in range(3)], [9, 9, 6])
        self.assertEqual([b.self_attn.layer_idx for b in model.model.layers], list(range(4)))

    def test_cached_and_uncached_generation_equal(self):
        for backend in ['eager', 'sdpa']:
            model, embeds = self.model(backend)
            cached = generate_preserved(model, embeds, 2, [3], [0, 2, 4], 4, True, visual_tokens=5, candidate_start=2)
            uncached = generate_preserved(model, embeds, 2, [3], [0, 2, 4], 4, False, visual_tokens=5, candidate_start=2)
            torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
