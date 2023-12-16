import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AttentionEncoder(nn.Module):
    """scaled dot-product attention
        used in the Aggregation sub-layer of the GCN layer
        aggregate the information from the k-nearest neighbors
        Eq.9 in the paper, ATTN
    """    
    
    def __init__(self, hidden_dim):
        super(AttentionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        # self.Wquery = nn.Linear(hidden_dim, hidden_dim)
        # self.Wkey = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h_n, neighbor):
        '''
        @param h_n: (batch_size, node_num, hidden_dim)
        @param neighbor: (batch_size, node_num, k, hidden_dim)
        '''
        
        # h_n = self.Wquery(h_n)
        # neighbor = self.Wkey(neighbor)
        
        h_n = h_n.unsqueeze(2) # (batch_size, node_num, 1, hidden_dim)
        neighbor = neighbor.permute(0, 1, 3, 2) # (batch_size, node_num, hidden_dim, k)
        attn_score = F.softmax(torch.matmul(h_n, neighbor) / np.sqrt(self.hidden_dim), dim=-1) # torch.matmul return a tensor with (batch_size, node_num, 1, k), softmax along the last dimension
        weighted_neighbor = attn_score * neighbor # (batch_size, node_num, hidden_dim, k)
        agg = h_n.squeeze(2) + torch.sum(weighted_neighbor, dim=-1) # (batch_size, node_num, hidden_dim)
        return agg
    
    
class AttentionDecoder(nn.Module):
    """the pointer mechanism in the decoder
    """    
    
    def __init__(self, hidden_dim, use_tanh=False):
        super(AttentionDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        
        self.Wquery = nn.Linear(hidden_dim, hidden_dim)    # the query matrix, which is used to embed the hidden state of the GRU
        self.Wkey = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)    # the key matrix, which is used to embed the resulting node embeddings
        self.C = 10    # the scaling factor of the attention score
        self.tanh = nn.Tanh()
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        v = torch.FloatTensor(hidden_dim).to(device)    # the parameter v in the attention mechanism
        self.v = nn.Parameter(v)  
        self.v.data.uniform_(-0.08, 0.08)
        
    def forward(self, h_n, gru_hidden):
        '''
        @param gru_hidden: (batch_size, hidden_dim)
        @param h_n: (node_num, batch_size, hidden_dim)
        '''
        # print('h_n.shape: ', h_n.shape)
        # print("gru_hidden.shape: ", gru_hidden.shape)
        h_n = h_n.permute(1, 2, 0)    # (batch_size, hidden_dim, node_num), since the Conv1d only works on the dimension -2
        query = self.Wquery(gru_hidden).unsqueeze(2)    # (batch_size, hidden_dim, 1)
        key = self.Wkey(h_n)    # (batch_size, hidden_dim, node_num)
        # print("query.shape: ", query.shape)
        # print("key.shape: ", key.shape)
        
        query = query.repeat(1, 1, key.size(-1))    # (batch_size, hidden_dim, node_num), keep the same dimension with key
        # print("query.shape: ", query.shape)
        # self.v.unsqueeze(0) results in (1, hidden_dim)
        # self.v.unsqueeze(0).expand(query.size(0), len(self.v)) results in (batch_size, hidden_dim)
        # self.v.unsqueeze(0).expand(query.size(0), len(self.v)).unsqueeze(1) results in (batch_size, 1, hidden_dim)
        v = self.v.unsqueeze(0).expand(query.size(0), len(self.v)).unsqueeze(1)    # (batch_size, 1, hidden_dim)
        # batch matrix multiplication, (batch_size, 1, hidden_dim) * (batch_size, hidden_dim, node_num) results in (batch_size, 1, node_num)
        u = torch.bmm(v, self.tanh(query + key)).squeeze(1)    # (batch_size, node_num)
        if self.use_tanh: logits = self.C * self.tanh(u)    # (batch_size, node_num)
        else: logits = u
        return key, logits