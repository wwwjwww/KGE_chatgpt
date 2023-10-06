from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse

    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., relation_embedding='', entity_embedding=''):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "relation_embedding": relation_embedding, "entity_embedding": entity_embedding}
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_pretrain_vocab(self, data1, data2):
        er_vocab = defaultdict(list)
        for triple in data1:
            if (triple[0], triple[1], triple[2]) in data2:
                er_vocab[(triple[0], triple[1], triple[2])].append(1)
            else:
                er_vocab[(triple[0], triple[1], triple[2])].append(0)
        return er_vocab
    
    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def get_batch_pretrain(self, er_vocab, er_vocab_pairs, idx, pretrain_vocab_pairs, pretrain_vocab_values, triple_embedding):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        pretrain_batch = pretrain_vocab_pairs[idx:idx+self.batch_size] 
        pretrain_batch_values = pretrain_vocab_values[idx:idx+self.batch_size] 
        pretrain_matrix = triple_embedding[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets, np.array(pretrain_batch), np.array(pretrain_batch_values), pretrain_matrix

    def get_matrix(self, entity_embedding, relation_embedding, idx_pairs):

        dim = self.ent_vec_dim + self.rel_vec_dim + self.ent_vec_dim
        h_idx = torch.tensor(idx_pairs[:,0]).cuda()
        r_idx = torch.tensor(idx_pairs[:,1]).cuda()
        t_idx = torch.tensor(idx_pairs[:,2]).cuda()

        h_embed = entity_embedding(h_idx)
        r_embed = relation_embedding(r_idx)
        t_embed = entity_embedding(t_idx)

        concatenated_matrix = torch.cat((h_embed, r_embed, t_embed), 1)

        return concatenated_matrix

    
    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            triple_idx = None
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions, _ = model.forward(e1_idx, r_idx, triple_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))


    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        pretrain_data_idx = self.get_data_idxs(d.pretrain_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        pretrain_vocab = self.get_pretrain_vocab(train_data_idxs, pretrain_data_idx)
        pretrain_vocab_pairs = list(pretrain_vocab.keys())
        pretrain_vocab_values = list(pretrain_vocab.values())

        triple_embedding = torch.tensor(np.load("../../../KGE_chatgpt/embed/triple_embedding_llama_avg_10000.npy", allow_pickle=True))

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets, pretrain_batch, pretrain_batch_values, pretrain_matrix = self.get_batch_pretrain(er_vocab, er_vocab_pairs, j, pretrain_vocab_pairs, pretrain_vocab_values, triple_embedding)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1]) 
                triple_idx = None
                if (pretrain_batch_values == 1).any():
                    triple_idx = torch.tensor([pretrain_batch[i] for i in range(len(pretrain_batch)) if pretrain_batch_values[i]==1])

                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    predictions, matrix = model.forward(e1_idx, r_idx, triple_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))

                loss_1 = model.loss_1(predictions, targets)
                loss=loss_1
                if matrix != None:
                    pretrain_matrix = pretrain_matrix.cuda()
                    targets_new = Variable(torch.Tensor(matrix.size(0)).cuda().fill_(1.0))
                    loss_2 = model.loss_2(matrix, pretrain_matrix, targets_new)
                    loss = 0.2 * loss_1 + 0.8 * loss_2

                #np.save('./cmp/structual_entity_embedding_%d.npy'%j, model.E.weight.data.detach().cpu().numpy())

                #loss = args.loss_parameter * loss_1 + (1-args.loss_parameter) * loss_2

                loss.backward(retain_graph=True)
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(it)
            print(time.time()-start_train)    
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data)
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data)
                    print(time.time()-start_test)
           
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    #parser.add_argument("--loss_parameter", type=float, default=0.5, nargs="?",
                    #help="loss parameter.")
    parser.add_argument("--relation_embedding", type=str, default='', nargs="?",
                        help="The file contains pretrained embeddings of relations. ")
    parser.add_argument("--entity_embedding", type=str, default='', nargs="?",
                        help="The file contains pretrained embeddings of relations. ")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    print(data_dir)
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=False)
    print(len(d.train_data))

    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, relation_embedding=args.relation_embedding, entity_embedding=args.entity_embedding)
    experiment.train_and_eval()
                

