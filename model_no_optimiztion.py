#%%


from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# -------------------------------------------------------------------------------


#%%
#this is the multi headed attention layer
# the naming is same as target
#  i made this fourth
# 1024 tokens of dim 728 get in it 

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd )
        # output projection
        self.c_proj = nn.Linear(config.n_embd , config.n_embd)
        # regularization  
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    
    def forward(self,x):
        B,T,C = x.size()# batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        qkv =self.c_attn(x) #qkv multiplication
        q,k,v =qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v  = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        att =(q @ k.transpose (-2,-1)) * (1.0/ math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0 ,float('-inf')) #autoregressive mask
        att =F.softmax(att,dim =-1) #normalize
        y= att @ v #matt mul (B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs) #weighted sum

        y = y.transpose(1,2).contiguous().view(B,T,C)# re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


#%%
#this is the mlp block 
#it feed forwards the information
#it is a basic mlp model of input size n_embd
#then the output size is 4 times that
#then it returns x of same size
#using gelu because of dEAD RElu problem
#i made this third

class MLP (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate= 'tanh')
        self.c_proj = nn.Linear( 4 * config.n_embd , config.n_embd)


    def forward(self,x):
       x=  self.c_fc(x)
       x= self.gelu (x)
       x=  self.c_proj(x) 
       return x


#%%%
#i made this second 
# this is inside the decoder block
#it normalizes the input and then feeds to the cassual self attention funtion (it is
# a change from the traditional arch in gpt 2)configure_optimizers
#then again normalizes it and adds
#then it feeds it to mlp (aka feed forward network)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 =nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self , x ):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



#%%




@dataclass
class GPTConfig:
    block_size: int = 1024 #max seq length
    vocab_size: int = 50257 # number of tokens: 50k BPE merges + 256 Byte tokes + 1 end of text
    n_layer: int = 12 #no. of layers
    n_head: int = 12 #no. of heads
    n_embd: int = 768 #embdding dim
 #%%
 # this block defines the basic architecture
 # it stores every layer weughts in the dictionar y
 # wte is embdding layer
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
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range (config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias =False)
#this was made to send data to my model
    def forward (self,idx,targets = None):
        B,T = idx.size()
        # idx is of shape (B, T) block size with t tokens in a sequence b staked up
        assert T <= self.config.block_size, f"cant forward seq of length {T}, block size is only {self.config.block_size}"
         # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd) 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb  # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x) 
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #cant take multi dim loss, it just flattens them out
        
        return logits ,loss

#this is trash code

  
#%%
#i made it after attention


#%%
#test
device = "cuda"
import tiktoken
enc = tiktoken.get_encoding("gpt2")
with open('input_toy.txt','r') as f:
    text = f.read()

text = text[:1000]
tokens = enc.encode(text)
B , T = 4 , 32   
buf = torch.tensor(tokens[:B*T +1],device=device)
x= buf[:-1].view(B,T)
y= buf[1:].view(B,T)

#get logits
model =GPT(GPTConfig()) #random modell intialization
# model = GPT.from_pretrained('gpt2')
model.to('cuda')

logits ,loss =model(x,y)

print(loss)
import sys; sys.exit(0)


# #%%
# model.eval()
# num_return_sequences = 5
# max_length = 32



# text = "Hello, I'm a language model,"
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
# x= tokens.to('cuda')

# #%%


# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1)<max_length:
#     with torch.no_grad():
#         logits = model(x)
#         logits = logits[:,-1,:]
#         probs = F.softmax(logits,dim=-1)
#         topk_probs,topk_indices = torch.topk(probs,50,dim=-1)
#         ix = torch.multinomial(topk_probs,1)
#         xcol =torch.gather(topk_indices,-1,ix)
#         x=torch.cat((x,xcol),dim=1)

# #%%

# for i in range (num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(" ",decoded)


# %%
