import torch
import torch.nn as nn

class ContextLayer(nn.Module):
    """the context node
    """    
    
    def __init__(self, hidden_dim):
        super(ContextLayer, self).__init__()
        
        env_layer = []
        env_layer.append(nn.Linear(1, hidden_dim))
        env_layer.append(nn.BatchNorm1d(hidden_dim))
        env_layer.append(nn.ReLU())
        for _ in range(10):
            env_layer.append(nn.Linear(hidden_dim, hidden_dim))
            env_layer.append(nn.BatchNorm1d(hidden_dim))
            env_layer.append(nn.ReLU())
        self.env_layer = nn.Sequential(*env_layer)
        
        context_layer = []
        context_layer.append(nn.Linear(hidden_dim*2, hidden_dim))
        context_layer.append(nn.BatchNorm1d(hidden_dim))
        context_layer.append(nn.ReLU())
        for _ in range(5):
            context_layer.append(nn.Linear(hidden_dim, hidden_dim))
            context_layer.append(nn.BatchNorm1d(hidden_dim))
            context_layer.append(nn.ReLU())
        self.context_layer = nn.Sequential(*context_layer)
        
    def forward(self, remaining_capacity, gru_hidden_pointer):
        """
        remaining_capacity (batch_size, 1)
        gru_hidden_pointer (batch_size, hidden_dim) this info integrates the info about current node
        """        
        # env = torch.cat((remaining_capacity), dim=1)
        env = self.env_layer(remaining_capacity)
        context = torch.cat((env, gru_hidden_pointer), dim=1)
        context = self.context_layer(context)
        return context