#%%
from transformers import GPT2LMHeadModel
model_hf  = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()
for k ,v in sd_hf.items():
    print(k,v.shape)

#wte is wieght token embedding -Size([50257, 768])   
#there are 50257 token in gpt 2 vocab, for each token we have 768 embeddings i.e 
#768 numbers are used to represent a token

# wpe is wieght pos embedding -Size([1024, 768])
# as context lenght is 1024 tokens we use 1024 positonal embedding
#.h is layers

#%%

sd_hf["transformer.wte.weight"].view(-1)[:20]
#%%


import matplotlib.pyplot as plt
plt.plot(sd_hf["transformer.wpe.weight"][:,150])
plt.plot(sd_hf["transformer.wpe.weight"][:,200])
plt.plot(sd_hf["transformer.wpe.weight"][:,250])

plt.show()
#finding-model not properly trained as the graph is not smooth and all noisy
#as the positional embeddings are insitialized randomly can try sins and coseins



#%%
 
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

#finding- setting seed to some constant still gives different outputs 
#(no documentaion found )

#%%
 
