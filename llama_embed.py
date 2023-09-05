# Load model directly
import torch
import pickle
from transformers import LlamaForCausalLM, LlamaTokenizer
import random
import numpy as np

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
model = model.bfloat16()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_data_batch(inputs, batch_size=None, shuffle=False):
    '''
    循环产生批量数据batch
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    # rows,cols=inputs.shape
    rows = len(inputs)
    indices = list(range(rows))
    if shuffle:
        random.seed(100)
        random.shuffle(indices)

    while True:
        batch_indices = indices[0:batch_size]  # 产生一个batch的index
        indices = indices[batch_size:]  # 循环移位，以便产生下一个batch
        batch_data = []
        temp_data = find_list(batch_indices, inputs)
        batch_data.append(temp_data)
        yield batch_data

def find_list(indices, data):
    out = []
    for i in indices:
        out.append(data[i])
    return out

def get_next_batch(batch):
    return batch.__next__()

def load_relation_file(filename):
    relation_lis = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            relation_dic = {"relation": None, "sentence": None}
            relation = line.rstrip().split('\t')
            relation_dic["relation"] = str(relation[0])
            relation_dic["sentence"] = str(relation[1])
            relation_lis.append(relation_dic)
    return relation_lis

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    relation_lis = []
    file1 = './data/FB15k-237/relation2text.txt'
    relation_lis = load_relation_file(file1)

    iter = 5
    batch_size = len(relation_lis) // iter + 1
    batch = get_data_batch(inputs=relation_lis, batch_size=batch_size, shuffle=False)

    embed_lis = []
    len_embed = 0
    for i in range(iter):
        embed_dic = {"relation": None, "embed": None}
        batch_relation = get_next_batch(batch)
        rela_dic = batch_relation[0]
        for j in rela_dic:
            rela_sentence = j["sentence"]
            inputs = tokenizer(rela_sentence, return_tensors='pt').to(device)
            model = model.to(device)
            embed = model.generate(inputs.input_ids, max_length=64, output_hidden_states=True)
            embed = embed.cpu().tolist()[0]
            embed_lis.append(embed)

    relation_file = open('relation_embed.pickle','wb')
    pickle.dump(embed_lis, relation_file)
    relation_file.close()
