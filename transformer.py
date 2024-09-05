import torch
from torch.nn import functional as F
import torch.nn as nn

# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = max_iters / 10
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 180
n_layer = 6
n_head = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
	text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)	#put all integers in a big tensor
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
	# generate a small batch of data of inputs x and targets y
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y

@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)	# pass target values Y to forward pass
			losses[k] = loss.item()	# save calculated loss to that eval iteration
		out[split] = losses.mean()	# get mean loss over split
	model.train()
	return out

# def plot_attention_weights(attention_weights, tokens):
#     """
#     Plots the attention weights as a heatmap.

#     :param attention_weights: The attention weights tensor of shape (T, T)
#     :param tokens: The list of tokens in the context
#     """
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
#     plt.xlabel('Query Position')
#     plt.ylabel('Key Position')
#     plt.title('Attention Weights')
#     plt.show()

class Head(nn.Module):
	""" one head of self - attention"""

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)	#basically just matrix multiplication with weights
		self.query = nn.Linear(n_embd, head_size, bias=False)	#basically just matrix multiplication with weights
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)
		q = self.query(x)
		# compute attention scores ("affinities")
		wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		wei = F.softmax(wei, dim=-1)
		wei = self.dropout(wei)
		# perform weighted aggregation of values
		v = self.value(x)
		out = wei @ v
		return out

class MultiHeadAttention(nn.Module):
	""" multiple heads of self-attention in parallel"""

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(head_size * num_heads, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out

class FeedForward(nn.Module):
	""" a simple layer followed by non-linearity """

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		return self.net(x)


class Block(nn.Module):
	""" Transformer block: communication followed by computation """

	def __init__(self, n_embd, n_head):
		super().__init__()
		head_size = n_embd // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embd)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))		# apply layer norm before feeding into head
		x = x + self.ffwd(self.ln2(x))		# apply layer norm before forward computation
		return x


# model with transformer
class GPT(nn.Module):

	def __init__(self):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)	# identity of token
		self.position_embedding_table = nn.Embedding(block_size, n_embd)	# positions of each token in the context
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # creates transformer blocks of communication & computation
		self.ln_f = nn.LayerNorm(n_embd) # final layer norm
		self.lm_head = nn.Linear(n_embd, vocab_size) # converts from n_embd vector space to vocab size, so that next char can be predicted

	def forward(self, idx, targets=None):
		B, T = idx.shape

		# idx and targets are both (B,T) tensor of integers
		tok_emb = self.token_embedding_table(idx) # (B,T,C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
		x = tok_emb + pos_emb # (B, T, C)
		x = self.blocks(x)
		logits = self.lm_head(x) # (B, T, vocab_size)

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# crop context to last block_size tokens
			idx_cond = idx[:, -block_size:]
			# get the predictions
			logits, loss = self(idx_cond)
			# focus only on the last time step
			logits = logits[:, -1, :] # becomes (B, C)
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # (B, C)
			# sample from the distribution
			idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
			# append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx

def train(model):
	# create a PyTorch optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

	for iter in range(max_iters):

		# every once in a while evaluate the loss on train and val sets
		if iter % eval_interval == 0:
			losses = estimate_loss()
			print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

		# sample a batch of data
		xb, yb = get_batch('train')

		# evaluate the loss
		logits, loss = model(xb, yb)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

