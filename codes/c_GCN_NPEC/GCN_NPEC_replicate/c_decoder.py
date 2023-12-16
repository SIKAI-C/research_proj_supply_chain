from .b_attention import AttentionDecoder
from .b_context import ContextLayer

import torch
import torch.nn as nn
import numpy as np    

class SequencialDecoderSupervised(nn.Module):
    """from the resulting embeddings to the resulting routes
    @param hidden_dim: the dimension of the hidden state, which is the same as the dimension of the node embedding
    @param decode_type: the type of the decoder, including 'greedy', 'sampling'
    @param use_cuda: whether to use cuda
    """    
    
    def __init__(self, hidden_dim, gru_num_layers=2):
        super(SequencialDecoderSupervised, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hidden_dim = hidden_dim
        
        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_num_layers)    # the GRU layer
        self.tanh = nn.Tanh()
        self.pointer = AttentionDecoder(hidden_dim, use_tanh=True)    # the pointer layer
        
    def forward(self, h_n, last_node, gru_hidden, mask):
        """
        @param h_n the node embedding (batch_size, node_num, hidden_dim)
        @param last_node the last node selected in the route (batch_size, 1)c
        @param gru_hidden the hidden state of the GRU (gru_num_layers, batch_size, hidden_dim)
        @param env the environment info, an Environment Object
        @param mask the mask of the node (batch_size, node_num)
        """
        batch_size = h_n.size(0)
        batch_idx = torch.arange(0, batch_size).unsqueeze(1).long().to(self.device)   # the idx, 0, 1, 2, ..., batch_size-1
        last_h_n = h_n[batch_idx, last_node].permute(1, 0, 2) # (gru_num_layers, batch_size, hidden_dim)
        _, gru_hidden = self.gru(last_h_n, gru_hidden) # (gru_num_layers, batch_size, hidden_dim)
        gru_hidden_pointer = gru_hidden[-1] # (batch_size, hidden_dim), select the last layer
        # Eq.15 
        _, u = self.pointer(h_n.permute(1,0,2), gru_hidden_pointer)
        # Eq.16
        u = u.masked_fill_(mask, -np.inf) # (batch_size, node_num)
        probs = self.softmax(u)
        probs = probs.clamp(min=1e-8)
        return probs, gru_hidden
        
class SequencialDecoder(nn.Module):
    """from the resulting embeddings to the resulting routes
    @param hidden_dim: the dimension of the hidden state, which is the same as the dimension of the node embedding
    @param decode_type: the type of the decoder, including 'greedy', 'sampling'
    @param use_cuda: whether to use cuda
    """    
    
    def __init__(self, hidden_dim, decode_type, gru_num_layers=2):
        super(SequencialDecoder, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hidden_dim = hidden_dim
        self.decode_type = decode_type
        
        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_num_layers)    # the GRU layer
        self.tanh = nn.Tanh()
        self.pointer = AttentionDecoder(hidden_dim, use_tanh=True)    # the pointer layer

        # combine the env info and the gru hidden state as the context vector
        # take the context vector to make the next decision
        # self.context_layer = ContextLayer(hidden_dim)
        
    # encourage exploration
    def boltzman_exploration(self, logits, tau=2):
        probs = torch.exp(logits / tau)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs
        
    def forward(self, h_n, last_node, gru_hidden, env, mask):
        """
        @param h_n the node embedding (batch_size, node_num, hidden_dim)
        @param last_node the last node selected in the route (batch_size, 1)c
        @param gru_hidden the hidden state of the GRU (gru_num_layers, batch_size, hidden_dim)
        @param env the environment info, an Environment Object
        @param mask the mask of the node (batch_size, node_num)
        """
        batch_size = h_n.size(0)
        batch_idx = torch.arange(0, batch_size).unsqueeze(1).long().to(self.device)   # the idx, 0, 1, 2, ..., batch_size-1
        last_h_n = h_n[batch_idx, last_node].permute(1, 0, 2) # (1, batch_size, hidden_dim)
        _, gru_hidden = self.gru(last_h_n, gru_hidden) # (gru_num_layers, batch_size, hidden_dim)
        gru_hidden_pointer = gru_hidden[-1] # (batch_size, hidden_dim), select the last layer
        # Eq.15 
        # _, u = self.pointer(h_n.permute(1,0,2), context)
        _, u = self.pointer(h_n.permute(1,0,2), gru_hidden_pointer)
        # Eq.16
        u = u.masked_fill_(mask, -np.inf) # (batch_size, node_num)
        probs = self.softmax(u)
        probs = probs.clamp(min=1e-8)
        if self.decode_type == "sampling":
            # probs = self.boltzman_exploration(u, 1.1)
            # probs = probs.clamp(min=1e-8)
            idx = torch.multinomial(probs, 1) # (batch_size, 1)
        elif self.decode_type == "greedy":
            # probs = self.softmax(u)
            # probs = probs.clamp(min=1e-8)
            idx = torch.argmax(probs, dim=1).unsqueeze(1) # (batch_size, 1)
        prob = probs[batch_idx, idx].squeeze(1) # (batch_size)
        return idx, prob, gru_hidden

    def beamForward(self, h_n, last_node, gru_hidden, env, mask):
        """
        @param h_n the node embedding (batch_size, node_num, hidden_dim)
        @param last_node the last node selected in the route (batch_size, 1)
        @param gru_hidden the hidden state of the GRU (gru_num_layers, batch_size, hidden_dim)
        @param env the environment info, an Environment Object (the env is not called in the function)
        @param mask the mask of the node (batch_size, node_num)
        """
        batch_size = h_n.size(0)
        batch_idx = torch.arange(0, batch_size).unsqueeze(1).long().to(self.device)   # the idx, 0, 1, 2, ..., batch_size-1
        last_h_n = h_n[batch_idx, last_node].permute(1, 0, 2) # (1, batch_size, hidden_dim)
        _, gru_hidden = self.gru(last_h_n, gru_hidden) # (gru_num_layers, batch_size, hidden_dim)
        gru_hidden_pointer = gru_hidden[-1] # (batch_size, hidden_dim), select the last layer
        # Eq.15 
        # _, u = self.pointer(h_n.permute(1,0,2), context)
        _, u = self.pointer(h_n.permute(1,0,2), gru_hidden_pointer)
        # Eq.16
        u = u.masked_fill_(mask, -np.inf) # (batch_size, node_num)
        # probs = self.softmax(u)
        # probs = probs.clamp(min=1e-8)
        return u, gru_hidden

class ClassificationDecoder(nn.Module):
    """from the edge embedding to the edge classification
    return 1 if the edge is considered as a edge appearing in the resulting routes
    """    
    
    def __init__(self, hidden_dim, hidden_dim_MLP, num_layers_MLP):
        super(ClassificationDecoder, self).__init__()
        MLP_Layers = []
        for i in range(num_layers_MLP):
            if i == 0:
                MLP_Layers.append(nn.Linear(hidden_dim, hidden_dim_MLP))
            else:
                MLP_Layers.append(nn.Linear(hidden_dim_MLP, hidden_dim_MLP))
            MLP_Layers.append(nn.ReLU())
        MLP_Layers.append(nn.Linear(hidden_dim_MLP, 2))
        self.MLP = nn.Sequential(*MLP_Layers)
        # self.softmax = nn.Softmax(dim=-1)
        # self.softmax = nn.functional.log_softmax(dim=-1)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    def forward(self, h_e):
        '''
        @param h_e: (batch_size, node_num, node_num, hidden_dim)
        '''
        # Eq.20 in the paper
        h_e = self.MLP(h_e) # (batch_size, node_num, node_num, 2)
        # h_e = h_e.squeeze(-1) # (batch_size, node_num, node_num)
        
        # batch_size, node_num, _, _ = h_e.shape
        # mask = torch.eye(node_num).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 2).to(self.device)
        # mask = mask * torch.tensor([0,1]).float().to(self.device)
        # h_e = h_e.masked_fill_(mask.bool(), float("-inf"))
        
        # h_e = self.softmax(h_e) # (batch_size, node_num, node_num, 2)
        h_e = nn.functional.log_softmax(h_e, dim=-1)
        # h_e = torch.sigmoid(h_e)
        return h_e