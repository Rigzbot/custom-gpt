import torch
import torch.nn as nn
from torch.nn import functional as F
from config import n_embd, block_size, dropout


class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # triangular matrix for masking forward tokens
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)   # (B, T, head_size)
    q = self.query(x) # (B, T, head_size)
    # compute attention scores
    wei = q @ k.transpose(-2, -1) * C**-0.5 # # (B, T, head_size) @ (B, 16, head_size) ---> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # make upper triangle values -inf in wei
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    # perform weight aggregation of values
    v = self.value(x)
    out = wei @ v # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
    return out
  
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd) # for skip connection
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating over the channel dimension
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.GELU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)
  
class TransformerBlock(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number of heads in multi-head architecture
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd) # Normalizing layer applied before self-attention and MLP layer
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # residual/skip connection 
    x = x + self.ffwd(self.ln2(x))
    return x