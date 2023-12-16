from .b_attention import AttentionEncoder

import torch
import torch.nn as nn
import numpy as np

class GCN(nn.Module):
    
    def __init__(
                    self,
                    hidden_dim,             # dimension of the hidden nodes in GCN layer
                    gcn_num_layers,         # number of GCN layers
                    k,                      # k-nearest neighbors, used to update the node embedding in the GCN layer
                    node_info_dim = 1       # the dimension of the node information, like demand, reward, ...
                ):
        super(GCN, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.hidden_dim = hidden_dim
        self.gcn_num_layers = gcn_num_layers
        self.k = k
        
        self.W1 = nn.Linear(2, hidden_dim)                      # $W_1$ in the paper, encoding the coordinates of the depot
        self.W2 = nn.Linear(2, hidden_dim // 2)                 # $W_2$ in the paper, encoding the coordinates of the customers
        self.W3 = nn.Linear(node_info_dim, hidden_dim // 2)     # $W_3$ in the paper, encoding the demands of the customers, TODO: add reward infomation
        self.W4 = nn.Linear(1, hidden_dim // 2)                 # $W_4$ in the paper, encoding distance of an edge
        self.W5 = nn.Linear(1, hidden_dim // 2)                 # $W_5$ in the paper, encoding the adjacency of an edge
        
        self.gcn_initializer_n = nn.Linear(hidden_dim, hidden_dim)    # $W_{E1}$ in the paper, used to initialize the node embedding in the GCN layer
        self.gcn_initializer_e = nn.Linear(hidden_dim, hidden_dim)    # $W_{E2}$ in the paper, used to initialize the edge embedding in the GCN layer
        
        self.gcn_layers = nn.ModuleList([GCNLayer(hidden_dim) for _ in range(gcn_num_layers)])
        
        self.relu = nn.ReLU()

    def getAdjacency(self, dist):
        """
        find the k-nearest neighbors of each node and return the adjacency matrix accordingly
        @param dist (node_num, node_num): distance matrix
        return a tensor with shape (node_num, node_num), the adjacency matrix
        """
        # Find the indices of the k nearest neighbors
        _, idx = torch.topk(dist, self.k+1, largest=False, dim=1)
        # Create an empty adjacency matrix
        adjacency = torch.zeros_like(dist).to(self.device)
        # Create a tensor with the row indices
        row_indices = torch.arange(dist.size(0)).unsqueeze(-1).expand_as(idx).to(self.device)
        # Use the row indices and the neighbor indices to fill the adjacency matrix
        adjacency[row_indices, idx] = 1.0
        # fill the diagonal elements to 0
        adjacency.fill_diagonal_(-1)
        return adjacency
    
    def getNeighbor(self, dist):
        """
        find the k-nearest neighbors of each node and return the indices of the neighbors
        @param dist (batch_size, node_num, node_num): distance matrix
        return a tensor with shape (batch_size, node_num, k), the indices of the neighbors
        """
        # Find the indices of the k+1 nearest neighbors
        _, idx = torch.topk(dist, self.k+1, largest=False, dim=-1)
        # Exclude the closest one (the first one), which is itself.
        idx = idx[:, :, 1:]
        return idx


    def forward(self, n_coor, n_info, dist):
        """
        @param n_coor: node coordination (batch_size, node_num(N+1), 2)
        @param n_info: node information (demand, reward, load_time) (batch_size, node_num(N+1), 3)
        @param dist: distance matrix (batch_size, node_num(N+1), node_num(N+1))
        """       
        # Eq.2 node information embedding, concatenate the coordinates and the demand(reward) information
        x_0 = self.relu(self.W1(n_coor[:, :1, :]))    # (batch_size, 1, hidden_dim)
        x_i = self.relu(torch.cat(((self.W2(n_coor[:, 1:, :])), (self.W3(n_info[:, 1:, :]))), dim=-1)) # concatenate the two embeddings each with shape (batch_size, node_num(N), hidden_dim // 2)
        x = torch.cat((x_0, x_i), dim=1)               # (batch_size, node_num(N+1), hidden_dim)
        
        
        # Eq.3, the adjacency matrix
        # print(dist)
        # print(dist.shape)
        # print(type(dist))
        adj_mat = torch.stack([self.getAdjacency(dist[i]) for i in range(dist.shape[0])]).to(self.device) # (batch_size, node_num(N+1), node_num(N+1))
        
        # Eq.4, edge information embedding, concatenate the distance and the adjacency information
        y = self.relu(torch.cat(((self.W4(dist.unsqueeze(3))), (self.W5(adj_mat.unsqueeze(3)))), dim=-1)) # (batch_size, node_num(N+1), node_num(N+1), hidden_dim)
        
        # Eq.5, Eq.6, initialize the node and edge embedding for GCN layer
        h_n = self.gcn_initializer_n(x)    # (batch_size, node_num(N+1), hidden_dim)
        h_e = self.gcn_initializer_e(y)    # (batch_size, node_num(N+1), node_num(N+1), hidden_dim)
        
        # print("GCN | before layers | h_n - {} | h_e - {}".format(h_n.shape, h_e.shape))
        
        # gcn layers
        neighbor = self.getNeighbor(dist)   # (batch_size, node_num(N+1), k)
        for gcn_layer in self.gcn_layers:
            h_n, h_e = gcn_layer(h_n, h_e, neighbor)
        
        # print("GCN | after layers | h_n - {} | h_e - {}".format(h_n.shape, h_e.shape))
        
        return h_n, h_e    



class GCNLayer(nn.Module):
    
    def __init__(self, hidden_dim):
        super(GCNLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # node GCN Layer
        self.W_node_agg = nn.Linear(hidden_dim, hidden_dim)    # $W_I^l$, aggregation sub-layer
        self.V_node_com = nn.Linear(hidden_dim, hidden_dim)    # $V_I^l$, combination sub-layer
        self.V_node = nn.Linear(2*hidden_dim, hidden_dim)      # feed forward after combination sub-layer
        self.attn = AttentionEncoder(hidden_dim)               # $ATTN$, attention layer used in the aggregation sub-layer
        self.relu = nn.ReLU()
        self.node_agg_layer_norm = nn.LayerNorm(hidden_dim)    # each sublayer in the node GCN layer has a skip connection and layer normalization
        self.node_com_layer_norm = nn.LayerNorm(hidden_dim)
        
        # edge GCN Layer
        self.W_edge_agg = nn.Linear(hidden_dim, hidden_dim)     # $W_E^l$, aggregation sub-layer
        self.W_edge_agg_1 = nn.Linear(hidden_dim, hidden_dim)   # $W_{e1}^l$, in the aggregation sub-layer
        self.W_edge_agg_2 = nn.Linear(hidden_dim, hidden_dim)   # $W_{e2}^l$, in the aggregation sub-layer
        self.W_edge_agg_3 = nn.Linear(hidden_dim, hidden_dim)   # $W_{e3}^l$, in the aggregation sub-layer
        self.V_edge_com = nn.Linear(hidden_dim, hidden_dim)     # $V_E^l$, combination sub-layer
        self.V_edge = nn.Linear(2*hidden_dim, hidden_dim)       # feed forward after combination sub-layer
        # self.relu = nn.ReLU()
        self.edge_agg_layer_norm = nn.LayerNorm(hidden_dim)     # each sublayer in the edge GCN layer has a skip connection and layer normalization
        self.edge_com_layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h_n, h_e, neighbor_index):
        '''
        @param h_n: (batch_size, node_num(N+1), hidden_dim)
        @param h_e: (batch_size, node_num(N+1), node_num(N+1), hidden_dim)
        @param neighbor_index: (batch_size, node_num(N+1), k)
        '''
        
        batch_size, node_num, hidden_dim = h_n.size()
        _, _, k = neighbor_index.size()
        
        # node embedding
        # Eq.7/9
        # aggregation sub-layer for node info
        # neighbor_index = neighbor_index.unsqueeze(3).repeat(1, 1, 1, hidden_dim)    # (batch_size, node_num, k, hidden_dim), copy the origin data k times along the forth dimension
        # h_n_repeated = h_n.unsqueeze(1).repeat(1, node_num, 1, 1)    # (batch_size, node_num, node_num, hidden_dim), copy the origin data node_num times along the second dimension
        
        # # print("GCN Layers | neighbor_index - {} | h_n_repeated - {}".format(neighbor_index.shape, h_n_repeated.shape))

        # print("h_n_repeated.shape: ", h_n_repeated.shape)
        # # print("neighbor_index.shape: ", neighbor_index.shape)
        # h_n_neighbor = h_n_repeated.gather(2, neighbor_index)    # (batch_size, node_num, k, hidden_dim), gather the neighbor node info
        # # print(neighbor_index[0,0,:,0])
        # print(h_n_neighbor.shape)
        # print(h_n_neighbor[0,0,:,0])
        
        neighbor_index = neighbor_index.unsqueeze(3).expand(-1, -1, -1, hidden_dim)    # (batch_size, node_num, k, hidden_dim), copy the origin data k times along the forth dimension
        h_n_repeated = h_n.unsqueeze(2).expand(-1, -1, k, -1)    # (batch_size, node_num, k, hidden_dim), copy the origin data node_num times along the second dimension
        h_n_neighbor = h_n_repeated.gather(1, neighbor_index)    # (batch_size, node_num, k, hidden_dim), gather the neighbor node info 
        
        h_n_agg = self.attn(h_n, h_n_neighbor)    # aggregation sub-layer
        h_n_agg = h_n + self.relu(self.W_node_agg(h_n_agg))    # skip connection after aggregation sub-layer
        h_n_agg = self.node_agg_layer_norm(h_n_agg)    # layer normalization after aggregation sub-layer
        
        # Eq.8/12
        h_n_com = torch.cat([self.V_node_com(h_n), h_n_agg], dim=-1)    # combination sub-layer
        h_n_com = h_n_agg + self.relu(self.V_node(h_n_com))
        h_n_com = self.node_com_layer_norm(h_n_com)
        
        h_n_next = h_n_com
        
        # edge embedding
        # Eq.7/10/11 
        # aggregation sub-lay for egde info
        h_n_from = h_n.unsqueeze(2).repeat(1, 1, node_num, 1)    # (batch_size, node_num, node_num, hidden_dim), copy the origin data node_num times along the third dimension
        h_n_to = h_n.unsqueeze(1).repeat(1, node_num, 1, 1)    # (batch_size, node_num, node_num, hidden_dim), copy the origin data node_num times along the second dimension
        
        # h_n_from have same value along the third dimension, therefore, self.W_edge_agg_2(h_n_from) has same value along the third dimension, that is j
        # h_n_to have same value along the second dimension, therefore, self.W_edge_agg_3(h_n_to) has same value along the second dimension, that is i
        h_e_agg = self.W_edge_agg_1(h_e) + self.W_edge_agg_2(h_n_from) + self.W_edge_agg_3(h_n_to)    # aggregation sub-layer
        h_e_agg = h_e + self.relu(self.W_edge_agg(h_e_agg))    # skip connection after aggregation sub-layer
        h_e_agg = self.edge_agg_layer_norm(h_e_agg)    # layer normalization after aggregation sub-layer
        
        # Eq.8/13
        h_e_com = torch.cat([self.V_edge_com(h_e), h_e_agg], dim=-1)    # combination sub-layer
        h_e_com = h_e_agg + self.relu(self.V_edge(h_e_com))
        h_e_com = self.edge_com_layer_norm(h_e_com)
        
        h_e_next = h_e_com
        
        return h_n_next, h_e_next
        