from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------------------------
#%%%
#i made this second 
# this is inside the decoder block
#it normalizes the input and then feeds to the cassual self attention funtion
#then again normalizes it and adds
#then it feeds it to mlp


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 =nn.LayerNorm(config.n_emd)
        self.mlp = MLP(config)
    
    def forward(self , x ):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



#%%




@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
 #%%
 # this block defines the basic architecture
 # it stores every layer weughts in the dictionary
 # wte is embedding layer
 # wpe is positional layer
 # h defines the nth attention head
 # lm_head is the layer norm layer
 # ln_f is the new layer  \
 # i made this first  
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config =config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_emb),
            wpe = nn.Embedding(config.block_size,config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range (config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.nembd,config.vocab_size,bias =False)