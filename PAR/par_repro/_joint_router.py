"""One shared40M controller; bidirectional attention lets visual tokens read the question."""
import torch
from par_repro._base_router import LayerRouter

class JointRouter(LayerRouter):
    def forward(self,visual,text,text_mask=None):
        if visual.ndim!=3 or text.ndim!=3 or visual.shape[0]!=text.shape[0] or visual.shape[-1]!=self.settings.input_size or text.shape[-1]!=self.settings.input_size:
            raise ValueError('Invalid Joint inputs')
        batch=visual.shape[0];device=visual.device
        if text_mask is None:text_mask=torch.ones(text.shape[:2],device=device,dtype=torch.bool)
        if text_mask.shape!=text.shape[:2]:raise ValueError('Invalid text mask')
        layers=self.layer_embeddings.unsqueeze(0).expand(batch,-1,-1)
        hidden=self.project(torch.cat([visual.to(self.project.weight.dtype),text.to(self.project.weight.dtype),layers],dim=1))
        valid=torch.cat([torch.ones(visual.shape[:2],dtype=torch.bool,device=device),text_mask.bool(),
            torch.ones(layers.shape[:2],dtype=torch.bool,device=device)],dim=1)
        attention=torch.zeros((batch,1,hidden.shape[1],hidden.shape[1]),device=device,dtype=hidden.dtype)
        attention.masked_fill_(~valid[:,None,None,:],torch.finfo(hidden.dtype).min)
        positions=(valid.long().cumsum(-1)-1).clamp_min(0)
        hidden=self.block(hidden,attention_mask=attention,position_ids=positions,use_cache=False)[0]
        scores=self.score(self.norm(hidden)).squeeze(-1)
        return torch.cat([scores[:,:visual.shape[1]],scores[:,-self.settings.candidate_count:]],dim=1)
