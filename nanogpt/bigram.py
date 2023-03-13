import torch 
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
split_size = 0.9
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length of predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ----------------------

torch.manual_seed(1337)

with open('./data/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Create the character tokenizers, stoi, itos, encoder and decoder functions.

chars = sorted(list(set(text)))
vocab_size = len(chars)
## Develop the strategy to create the sequences and mapping from characters to integers.
stoi = { ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string and output is list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder; takes the list of ints and outputs the string

# Lets separate out the data into train and validation sets.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(split_size*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:] # true shakespeare Text


# data loading
def get_batch(split):
    # generate small batches of data of input x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y= y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # losses = losses.to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()        
    return out


# super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size, device=device)
        
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


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer much more advance than SGD
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    
    # eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    
# generate mode the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
