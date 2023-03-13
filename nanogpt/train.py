# this file trains on the tiny shakespeare
# Bonus Goals: 
# Add the tests for gradients and graphs at as many steps as possible.


with open('./data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
    
# print(f"length of the dataset in characters: {len(text)}")
# print(f"{text[:1000]}")


# Create the character tokenizers, stoi, itos, encoder and decoder functions.

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab Size: {vocab_size} and vocabulary: {''.join(chars)}")


## Develop the strategy to create the sequence

stoi = { ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string and output is list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder; takes the list of ints and outputs the string
print(encode("hii there"), decode(encode("hii there")))

## Google uses sentence piece sub-word tokenizer 
## OpenAI uses BytePair tokenizer - called tiktoken - example below

import tiktoken
enc = tiktoken.get_encoding('gpt2')
# print(f"GPT-2 vocab size: {enc.n_vocab}")
# print(enc.encode("hii there"), enc.decode(enc.encode("hii there")))

# Trade offs are 
# (a) very long sequences of integers and small vocabulary or 
# (b) very short sequences of integers and large vocabulary


# Let's now encode the entire text into text dataset and store it in torch.tensor - this class is very well built.
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000])

# Lets separate out the data into train and validation sets.
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:] # true shakespeare Text


# We never feed the entire text into the transformers instead we sample random chunks and train on chunks
# max length of chunks is block_size
block_size = 8
print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"When input is {context}, the target is: {target}")
    # print(f"The input characters are {decode([c.item() for c in context])} The target character is --> {decode([target.item()])}")
    # print("-----")
    
# We call this dimension - Time Dimension for sequences that we will be feeding into the Transformers

# Next, for utilizing GPU we will pack this into multiple mini batches - batch dimension.


# batch Dimension
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length of predictions?

def get_batch(split):
    # generate small batches of data of input x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)

print('targets:')
print(yb.shape)
print(yb)


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        # print(f"---------\n When input is {context.tolist()}, then the target: {target}")
        # print(f"The input characters are {decode(context.tolist())} The target character is --> {decode([target.item()])} \n-------")


# input to the transformers is 4 X 8 tensors and targets are 4 X 8 tensor that will come at the end to calculate the loss function.
# Next, lets feed it into the simplest possible neural network - bigram language model

import torch 
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        
        # idx and targets are both B, T tensor of integers
        logits = self.token_embedding_table(idx) # B, T, C
        
        if targets == None:
            loss = None
        else:
            # reshape the matrix with what Pytorch needs - read cross entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets  = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            
            # focus only on last time step because embedding is vocab_size X vocab_size - 2 dimension bigram model
            logits = logits[:, -1, :] # becomes B, C
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # B, C
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


# create a Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


batch_size = 32
for steps in range(10000):
    xb, yb = get_batch('train')
    
    # eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item()) 


print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
