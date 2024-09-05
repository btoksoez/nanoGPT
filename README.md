# nanoGPT


![transformer](https://imgs.search.brave.com/QlCAPeD3oC57obw7qO5AQ3OsuUk1ysfGl8j4HiEPCgM/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy8z/LzM0L1RyYW5zZm9y/bWVyLF9mdWxsX2Fy/Y2hpdGVjdHVyZS5w/bmc)

## Introduction

- simple bigram: loss: 2.5
- simple bigram with one self-attention head: 2.4
- multiple attention heads (4): 2.27
- with feed forward layer (simple linear layer with non-linear activation): 2.24
- with multiple blocks of attention heads and feedforward layers + residual connections: 1.97
- adding layer norms before feedforward and attention heads and after transformer block: 2.00


## Usage

## Learnings
- BPE tokenizer: The BPE tokenizer used in GPT-2 is a subword tokenization method that breaks text into smaller units to handle rare and complex words. It starts by splitting words into individual characters. The most frequent pairs of characters or subwords are merged iteratively to create a vocabulary of subwords, until a predefined vocabulary size is reached. This process helps the model handle out-of-vocabulary words by breaking them into familiar subwords, allowing efficient processing with a compact vocabulary. GPT-2 uses these subwords as tokens, converting them into integer indices that the model can understand and generate text from.
	Example:

	Start with characters: ['l', 'o', 'w', 'e', 'r']
	Merge 'l' and 'o': ['lo', 'w', 'e', 'r']
	Merge 'lo' and 'w': ['low', 'e', 'r']
	Merge 'low' and 'e': ['lowe', 'r']
	The word "lower" might end up as two tokens: ['lowe', 'r']

- **`torch.randint` Arguments**:
   - **Question**: What are the arguments for `torch.randint`?
   - **Answer**: `torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)`. It generates random integers between `low` (inclusive) and `high` (exclusive) with the specified `size`.

- **Understanding `torch.stack`**:
   - **Question**: What does `x = torch.stack([data[i:i+block_size] for i in ix])` do?
   - **Answer**: It creates a batch of sequences from the `data` tensor, where each sequence is of length `block_size`. The resulting tensor `x` has shape `[batch_size, block_size]`.

- **`torch.nn.Embedding`**:
   - **Question**: What does `torch.nn.Embedding` do?
   - **Answer**: `torch.nn.Embedding` creates a lookup table that maps indices to dense vectors, commonly used to convert word indices into word embeddings.

- **`super().__init__()`**:
   - **Question**: What is `super().__init__()`?
   - **Answer**: It calls the constructor of the parent class to ensure it is properly initialized. In the context of PyTorch, it initializes the `nn.Module` base class.

- **Automatic Parent Class Construction**:
   - **Question**: Does Python automatically construct the parent class?
   - **Answer**: No, you need to explicitly call the parent class's constructor using `super().__init__()`.

- **Contents of `nn.Module`**:
   - **Question**: What does the base class `nn.Module` contain?
   - **Answer**: `nn.Module` provides functionality for parameter management, submodule management, defining the forward pass, switching between training and evaluation modes, and saving/loading the model state.

- Automatic forward Call: In PyTorch, calling an instance of a class that inherits from nn.Module automatically invokes the forward method due to the overridden __call__ method in nn.Module.
	```
	m = BigramLanguageModel(vocab_size)
	out = m(xb, yb)
	```
	calls m.forward(xb, yb)

- #### Summary of `optimizer.step()` and Embedding Initialization

	##### What Happens with `optimizer.step()`

	1. **Gradient Computation**:
	- Before calling `optimizer.step()`, the gradients of the loss with respect to each parameter are computed using `loss.backward()`. These gradients are stored in the `.grad` attribute of each parameter.

	2. **Parameter Update**:
	- The `optimizer.step()` method iterates over all the parameters of the model and updates each parameter based on its gradient and the specific optimization algorithm being used (e.g., AdamW).

	3. **AdamW Optimization Algorithm**:
	- For the AdamW optimizer, the parameter update involves:
		- Computing biased first and second moment estimates.
		- Correcting these estimates for bias.
		- Updating the parameter using these corrected estimates.

	##### Embedding Table Initialization

	1. **Random Initialization**:
	- The embedding table in `nn.Embedding(vocab_size, vocab_size)` is initialized with weights drawn from a normal distribution \( N(0, 1) \).
	- This means that initially, the probabilities (or logits) for any character being followed by another character are essentially random.

	2. **Training Process**:
	- **Forward Pass**: The model uses the current embedding table to predict the next character based on the current character.
	- **Loss Calculation**: The model calculates the loss (e.g., cross-entropy loss) between the predicted logits and the actual next character in the training data.
	- **Backward Pass**: The gradients of the loss with respect to the embedding table weights are computed using backpropagation.
	- **Parameter Update**: The optimizer updates the weights of the embedding table in the direction that reduces the loss.

	3. **Probability Adjustment**:
	- Over many iterations, the weights in the embedding table are adjusted such that the logits (and thus the probabilities) for character sequences that occur frequently in the training data are increased.

	##### Example of Weight Update

	- **Initialization**: The embedding table is initialized with weights drawn from a normal distribution \( N(0, 1) \).
	- **Training**: The weights are updated based on the gradients computed from the loss, moving in the direction that reduces the loss.
	- **Probability Adjustment**: Over time, the weights for frequently occurring character sequences are increased, making the model more likely to predict these sequences.

	This process allows the model to learn the probabilities of character sequences from the training data, starting from a state where the weights are randomly initialized around 0 with a variance of 1.
- The .to(device) method is used to move tensors to a specified device (CPU or GPU).
- self-attention:
	- we want tokens to only communicate with the tokens before them (not future ones), so we use tril to set the upper triangle of weights to -inf
	- we want to somehow make the weights adjustable, so that each token can learn what to pay attention to, so we use Linear layers
	- we use softmax to normalize weights across rows
	- if two elements in the context have high probabilities, they will interact stronger with each other / pay attention to each other
	```Python
	# a single head performing self-attention
	# basically gets some initial random weights in the right shape
	head_size = 16
	key = nn.Linear(C, head_size, bias=False)	#basically just matrix multiplication with weights
	query = nn.Linear(C, head_size, bias=False)	#basically just matrix multiplication with weights
	k = key(x)
	q = query(x)
	wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

	# mask all the upper triangle (so that items only look back, not forward)
	tril = torch.tril(torch.ones(T, T))
	wei = wei.masked_fill(tril == 0, float('-inf'))

	# normalize over rows, so that they sum to 1
	wei = F.softmax(wei, dim=-1)

	# multiply these final weights with inputs
	out = wei @ x
	```
	basically doing this:
	![attention](<./imgs/attention.png>)

	The weights look like this (for each batch):
	```
	[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
	[0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
	[0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
	[0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
	[0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
	[0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
	[0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
	[0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]]
	```
	- attention is basically a communication mechanism, between nodes (each character in the context), and the previous ones are always connected to the next one
	- batches don't talk to each other (there are no weights learned in between batches)
	- an encoder attention block would let all nodes talk to each other (no tril), a decoder block never wants future nodes to talk to previous ones (autoregresssive)
	- self-attention: all information comes from x (keys, values, queries)
	- cross-attention: other nodes can give information
	- scaled attention: normalizes weights over sqrt(head size) to get variance one
		- -> you want values to have variance of one, because otherwise softmax will "sharpen" values, e.g. converge to the one with the highest value
- residual connections: have a direct pathway from inputs to next layer, without computation; then add up the computed layer and the direct inputs to the next layer's inputs. Because it's an addition, when backpropagating the gradients will just be passed through.
	![residual](<./imgs/residual.png>)

	- simply add the attention and feedforward to the geiven inputs:
	```Python
	def forward(self, x):
		x = x + self.sa(x)
		x = x + self.ffwd(x)
		return x
	```
	- add a projection layer, that scales down from num_heads * head_size to n_embd
	``` self.proj = nn.Linear(head_size * num_heads, n_embd) ```
- Layer norm: instead of normalizing over columns (=batch norm), we normalize over rows
	- in this case: normalize over features/embeddings of each token, not over batches
	```Python
	def _call_(self, x):
		# calculate the forward pass
		xmean = x.mean(1, keepdim=True) # layer mean
		xvar = x.var(1, keepdim=True) # layer variance
		xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
		self.out = self.gamma * xhat + self.beta
		return self.out
		# gamma and beta are parameters that can be learned
	```
- adding a dropout layer, to prevent overfitting
	![dropout](<./imgs/dropout.png>)



## Definitions to remember
- layer norm
- dropout layer
