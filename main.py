import torch
from bigram import BigramLanguageModel
from transformer import GPT

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

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# generate from the model
def generate(model, max_new_tokens):
	context = torch.zeros((1, 1), dtype=torch.long, device=device)
	print(f"\nGenerate {max_new_tokens} characters:")
	print(decode(model.generate(context, max_new_tokens)[0].tolist()), "\n")

# complete given prompt
def complete(prompt, model, max_new_tokens):
	context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
	print(f"Generate {max_new_tokens} characters:\n")
	print(decode(model.generate(context, max_new_tokens)[0].tolist()) + "\n")

def main():
	# prompt user
	while True:
		model_choice = input("Choose model:\n Bigram(1) or GPT (2) or exit? ")
		if model_choice == "exit":
			break
		if model_choice == "1":
			model = BigramLanguageModel(vocab_size)
			params = './parameters/bigram_params.pth'
		elif model_choice == "2":
			model = GPT()
			params = './parameters/gpt_params.pth'
		else:
			print("Invalid choice. Please choose 1, 2, or exit.")
			continue

		# load the model parameters
		model.load_state_dict(torch.load(params, map_location=device, weights_only=True))
		model.to(device)
		model.eval()

		while True:
			choice = input("Do you want to:\nGenerate (1), complete (2) or back? ")
			if choice == "back":
				break
			max_new_tokens = int(input("How many characters do you want to generate? "))
			if choice == "1":
				generate(model, max_new_tokens)
			elif choice == "2":
				prompt = input("Your prompt to complete: ")
				complete(prompt, model, max_new_tokens)
			else:
				print("Invalid choice. Please choose 1, 2, or exit.")

if __name__ == "__main__":
	main()
