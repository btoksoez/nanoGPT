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

## Definitions to remember

