
from transformers import GPT2LMHeadModel
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2',device='cuda')
set_seed(42)
print (generator("Hello, I'm a monkey ,", max_length=50, num_return_sequences=5))

