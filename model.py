#%%


from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

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
        self.c_proj.OP = 1 
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


        #replaced by flash attn
        # att =(q @ k.transpose (-2,-1)) * (1.0/ math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0 ,float('-inf')) #autoregressive mask
        # att =F.softmax(att,dim =-1) #normalize
        # y= att @ v #matt mul (B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs) #weighted sum

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

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
        self.c_proj.OP = 1
        


    def forward(self,x):
       x=  self.c_fc(x)
       x= self.gelu (x)
       x=  self.c_proj(x) 
       return x


#%%%
#i made this second 
# this is inside the decoder block
#it normalizes the input and then feeds to the cassual self attention funtion (it is
# a change from the traditional arch in gpt 2)
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
         # weight sharing scheme
         #model initization (from gpt code from open ai)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights) #iterates all the sub module

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'OP'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #std = 1/sqrt(dmintions)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


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

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
#%%
#optimizer object and regularizatrion
#splits parameters that should be weight and that shouldnt be 
#decayed wale are jinko regularize krna h only 2d
    def configure_optimizers(self, weight_decay, learning_rate, device):
            # start with all of the candidate parameters (that require grad)
            param_dict = {pn: p for pn, p in self.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum (p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and 'cuda' in device
            
            print(f"using fused AdamW: {use_fused}")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
            return optimizer

#%%
#-------------------------------------#
import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input_toy.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded{len(self.tokens)} tokens")
        print(f"1 epoch ={len(self.tokens) // (B * T) }Batches")
        #state
        self.current_position =0



    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
           
            self.current_position = 0
        return x, y
#%%


#%%
#test
import time

torch.cuda.empty_cache()
device = "cuda"
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


#%%
#now that i only have 6gigs of gpu and the paper says a 0.5M of Batch size
#its not really 0.5 milion its .5M/1024 = 488k
#and my gpu explodes at bize size of 2,
#i need to simulate the orignal batch size
#this techibiqe is called gradient accumiliation

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B =  2 # micro batch size
T = 1024 # sequence length
#2048 tokens in 1 time
#256 times i need to loop this 
assert total_batch_size % (B * T ) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T )

print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")










#%%



train_loader = DataLoaderLite(B=2,T=1024)

torch.set_float32_matmul_precision('high') #takes data presion down,increases speed by 8 times

#get logits
model =GPT(GPTConfig(vocab_size=50304)) #random modell intialization
#i increased and added a few fake tokens to make an odd number a nice number(near 2 number)
#they are never used and are driven to 0 anyway
# model = GPT.from_pretrained('gpt2')
model.to('cuda')

model = torch.compile(model)  #gcc of python
#unable to run on this pc please check
#kaafi important
#%%
#learning late optimization (cosine decay learing schedule)
max_lr = 4e-4  #from table 2.1
min_lr = max_lr * 0.1 
warmup_steps = 10
max_steps = 50 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)






#%%



#%%


#optimization and regularixation acc to paper
# optimizer = torch.optim.AdamW(model.parameters(),lr= 3e-4,betas=(0.9, 0.95), eps=1e-8) 
#from gpt 3 paper
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=4e-4, device=device)
#replacement to stochastic gragient decent optimizer
#%%


for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() #always start with 0
    loss_accum = 0.0
    x,y =train_loader.next_batch()
    x,y = x.to(device), y.to(device)
    for micro_step in range (grad_accum_steps):

        with torch.autocast(device_type=device, dtype=torch.bfloat16): #little to no documentation
            logits ,loss =model(x,y)
        loss = loss / grad_accum_steps #normalizes everything
        loss_accum += loss.detach() #detachs the tensor

        loss.backward() #it accumulates the gradients
    #from gpt3 paper
    #calculates the norm of paramenter vector(length<1.0)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#%%


    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




    optimizer.step()
    torch.cuda.synchronize()
    t1= time.time()
    dt = (t1-t0)*1000
    tokens_processed = (train_loader.B * train_loader.T *grad_accum_steps)
    tokens_per_sec = tokens_processed / (t1-t0)
    print(f"step{step:4d},   loss:{loss_accum.item():.6f}, lr {lr:.4e} ,  dt: {dt:.2f}ms,  norm: {norm:.4f} ,  tok/sec: {tokens_per_sec:.2f}") #loss tensor with single elemnt

import sys; sys.exit(0)


#%%


num_return_sequences = 5
max_length = 32



text = "Hello, I'm a language model,"
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x= tokens.to('cuda')

#%%


torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1)<max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits,dim=-1)
        topk_probs,topk_indices = torch.topk(probs,50,dim=-1)
        ix = torch.multinomial(topk_probs,1)
        xcol =torch.gather(topk_indices,-1,ix)
        x=torch.cat((x,xcol),dim=1)

#%%

for i in range (num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(" ",decoded)


# %%
