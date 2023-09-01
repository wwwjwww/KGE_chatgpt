# Load model directly
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

#inputs = tokenizer("Hello this is a test", return_tensors='pt')
inputs = tokenizer("Today is a rainy day, please be careful!", return_tensors='pt')
embed = model.generate(inputs.input_ids, max_length=30, output_hidden_states=True)
print(embed)
