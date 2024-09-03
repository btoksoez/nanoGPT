# nanoGPT

## Introduction

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
	- we want tokens to only communicate with the tokens before them (not future ones)
	- 

## Definitions to remember

