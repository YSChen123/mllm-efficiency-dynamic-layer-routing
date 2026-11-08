import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from par_repro._paper_router import PaperRouter


class ControlledRouter(PaperRouter):
    def __init__(self, settings, attention_mode='bidirectional'):
        if attention_mode not in ['bidirectional', 'causal']:
            raise ValueError('Unknown router attention mode')
        super().__init__(settings)
        self.attention_mode = attention_mode

    def forward(self, visual, text, text_mask=None):
        if self.attention_mode == 'bidirectional':
            return super().forward(visual, text, text_mask)
        if visual.ndim != 3 or text.ndim != 3 or visual.shape[0] != text.shape[0]:
            raise ValueError('Invalid router inputs')
        if visual.shape[-1] != self.settings.input_size or text.shape[-1] != self.settings.input_size:
            raise ValueError('Invalid input width')
        batch = visual.shape[0]
        if text_mask is None:
            text_mask = torch.ones(text.shape[:2], device=visual.device, dtype=torch.bool)
        if text_mask.shape != text.shape[:2]:
            raise ValueError('Invalid text mask')
        layers = self.layer_embeddings.unsqueeze(0).expand(batch, -1, -1)
        hidden = self.project(torch.cat([visual.to(self.project.weight.dtype), text.to(self.project.weight.dtype), layers], dim=1))
        valid = torch.cat([torch.ones(visual.shape[:2], device=visual.device, dtype=torch.bool), text_mask.bool(),
                           torch.ones(layers.shape[:2], device=visual.device, dtype=torch.bool)], dim=1)
        attention = _prepare_4d_causal_attention_mask(valid, hidden.shape[:2], hidden, 0)
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.block(hidden, attention_mask=attention, position_ids=positions, use_cache=False)[0]
        scores = self.score(self.norm(hidden)).squeeze(-1)
        return torch.cat([scores[:, :visual.shape[1]], scores[:, -self.settings.candidate_count:]], dim=1)
