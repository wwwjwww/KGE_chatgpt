import numpy as np
import torch
from torch.nn.init import xavier_normal_, zeros_, xavier_uniform_
import torch.nn.functional as F
import pickle
from torch.autograd import Variable


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.entity_embed = None
        self.relation_embed = None
        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)

        if kwargs["entity_embedding"] != "":
            self.entity_embed = torch.tensor(np.load(kwargs["entity_embedding"], allow_pickle=True))
            self.W_E = torch.nn.Linear(self.entity_embed.size(1), d1, bias=True) 
            self.E = torch.nn.Embedding.from_pretrained(F.relu(self.W_E(self.entity_embed)), freeze=False)

        if kwargs["relation_embedding"] != "":
            self.relation_embed = torch.tensor(np.load(kwargs["relation_embedding"], allow_pickle=True))
            self.W_R = torch.nn.Linear(self.relation_embed.size(1), d2, bias=True)
            self.R = torch.nn.Embedding.from_pretrained(F.relu(self.W_R(self.relation_embed)), freeze=False)

        self.triple_embedding = torch.tensor(np.load("../../../KGE_chatgpt/embed/triple_embedding_llama_avg_10000.npy", allow_pickle=True))
        self.fc = torch.nn.Linear(d1+d2+d1, self.triple_embedding.size(1), bias=True)

        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])

        self.loss_1 = torch.nn.BCELoss()
        self.loss_2 = torch.nn.CosineEmbeddingLoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        
    def init(self):
        xavier_uniform_(self.fc.weight)
        zeros_(self.fc.bias)
        if self.entity_embed == None:
            xavier_normal_(self.E.weight.data)
        else:
            xavier_uniform_(self.W_E.weight)
            zeros_(self.W_E.bias)

        if self.relation_embed == None:
            xavier_normal_(self.R.weight.data)
        else:
            xavier_uniform_(self.W_R.weight)
            zeros_(self.W_R.bias)

    def get_matrix(self, idx_pairs):
        entity_embedding = self.E
        relation_embedding = self.R

        h_idx = torch.tensor(idx_pairs[:,0]).cuda()
        r_idx = torch.tensor(idx_pairs[:,1]).cuda()
        t_idx = torch.tensor(idx_pairs[:,2]).cuda()

        h_embed = entity_embedding(h_idx)
        r_embed = relation_embedding(r_idx)
        t_embed = entity_embedding(t_idx)

        concatenated_matrix = torch.cat((h_embed, r_embed, t_embed), 1)

        return concatenated_matrix

    def forward(self, e1_idx, r_idx, triple_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)

        if triple_idx != None:
            triple_matrix = torch.tensor(self.get_matrix(triple_idx))
            linear_matrix = self.fc(triple_matrix)
            matrix = torch.sigmoid(linear_matrix)
        else:
            matrix = None

        return pred, matrix

