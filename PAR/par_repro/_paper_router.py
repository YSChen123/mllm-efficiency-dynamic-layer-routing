"""Paper MLP+sigmoid, explicit unverified widths, no conditional branch objective."""
from dataclasses import dataclass
import torch
from torch import nn
from par_repro._base_router import RouterConfig
from par_repro._joint_router import JointRouter

@dataclass
class PaperRouterConfig(RouterConfig):
    head_hidden_size:int=512

class PaperRouter(JointRouter):
    def __init__(self,settings):
        super().__init__(settings)
        self.score=nn.Sequential(nn.Linear(settings.hidden_size,settings.head_hidden_size,bias=False),nn.GELU(),
            nn.Linear(settings.head_hidden_size,1,bias=False))
        for module in self.score.modules():
            if isinstance(module,nn.Linear):nn.init.normal_(module.weight,mean=0.,std=settings.init_std)
        self.prune_m=400

    def probabilities(self,visual,text,text_mask=None):
        return self(visual,text,text_mask).sigmoid()
