# Load model directly
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
relation_lis = []
with open('./data/FB15k-237/relation2text.txt') as f:
    for line in f:
        relation_dic = {"relation": None, "embed": None}
        relation = line.split('\t')
        relation_trans = str(relation[1])
        inputs = tokenizer(relation_trans, return_tensors='pt').to(device)
        model = model.to(device)
        embed = model.generate(inputs.input_ids, max_length=32, output_hidden_states=True)
        relation_dic["relation"] = relation[0]
        relation_dic["embed"] = embed
        relation_lis.append(relation_dic)
relation_file = open('relation_embed.pickle','wb')
pickle.dump(relation_lis, relation_file)
relation_file.close()
