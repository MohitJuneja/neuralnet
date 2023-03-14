# The math trick in self attention, i.e., how to allow the tokens to learn about the previous tokens - the idea is implemented in
# the paper with Q and K vector. Each token will aska 

# consider the following example:

import torch
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape) # torch.size([4,8,2])

# how do we extract x[b, t] = mean_{i<=t} x[b, i]. In plain English, how do we get the average of all the preceding tokens but no future tokens

xbow = torch.zeros((B, T, C)) # bag of words means average from all the tokens
for b in range(B):  # for all the batches
    for t in range(T): # for every time step (token) in each batch
        xprev = x[b,:t+1]  # shape (t, C) # for that batch, take all the tokens upto and including the t timestep.
        xbow[b,t] = torch.mean(xprev, 0)  # average of the characters up until and including the timestep on the zero'th dimension - i.e., for each batch B

print(x[0], xbow[0])        
# Above is CPU specific inefficient code with for loops
# How can we convert the code into parallizable GPU code with Matrices
# Answer is with torch.tril which creates a matrix with top half of the off-diagonals as 0.0 or any other value such as -inf.

# Efficient Method
torch.manual_seed(42)
a = torch.ones(3, 3)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b
d = torch.tril(torch.ones(3,3))
e = d / torch.sum(d, 1, keepdim=True)
for k,v in {'a': a, 'b': b, 'c': c, 'd': d, 'e':e}.items():
    print(f"{k}==")
    print(f"{v}")
    print("---")
    
    
# so, we can use this trick to remove the for loop above on line 17,18

wei = torch.tril(torch.ones(T,T)) # triangular matrix
wei = wei / torch.sum(wei, 1, keepdim=True) # weighted aggregation & information gets taken from prev tokens
xbow2 = wei @ x  # (T,T) @ (B, T, C) ---> (B,T,T) @ (B,T,C) --> (B, T, C)
print(torch.allclose(xbow, xbow2))


# next, same values can be calculated by including a softmax function such that we can utilize this matrix for self-attention later (measuring affinities)
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
print(wei)


