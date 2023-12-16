from .c_GCN import GCN
from .c_decoder import SequencialDecoder, ClassificationDecoder
import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(
        self,
        hidden_dim, 
        gcn_num_layers,
        k = 10,
        node_info_dim = 1,
        gru_num_layers = 2,
        mlp_num_layers = 2
    ):
        super(Model, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.GCNEncoder = GCN(
            hidden_dim=hidden_dim,
            gcn_num_layers=gcn_num_layers,
            k=k,
            node_info_dim=node_info_dim
        )
        self.sequentialDecoderSample = SequencialDecoder(
            hidden_dim=hidden_dim,
            decode_type="sampling",
            gru_num_layers=gru_num_layers
        ).to(self.device)
        # Rollout
        self.sequentialDecoderGreedy = SequencialDecoder(
            hidden_dim=hidden_dim,
            decode_type="greedy"
        ).to(self.device)
        self.classificationDecoder = ClassificationDecoder(
            hidden_dim=hidden_dim,
            hidden_dim_MLP=hidden_dim,
            num_layers_MLP=mlp_num_layers
        )
        self.gru_num_layers = gru_num_layers
        self.hidden_dim = hidden_dim
        
    def seqDecoderForward(self, env, h_n, decode_type="sampling"):
        """decide the routes based on the sequential decoder

        @param env the Environment object
        @param h_n (batch_size, hidden_dim), the node embedding from the GCN encoder
        @param decode_type decode type, "sampling" or "greedy" Defaults to "sampling".
        
        @return total_dist (batch_size), the total distance of the routes
        @return log_prob (batch_size), the log probability of the routes
        @return matrix (batch_size, node_num, node_num), the adjacency matrix of the routes
        """
        batch_size = env.batch_size
        node_num = env.node_num
        env.reset()
        last_node = torch.zeros((batch_size, 1)).long().to(self.device) # (batch_size, 1)
        gru_hidden = torch.zeros((self.gru_num_layers, batch_size, self.hidden_dim)).to(self.device) # (gru_num_layers, batch_size, hidden_dim)
        mask = torch.zeros((batch_size, node_num)).bool().to(self.device) # (batch_size, node_num)
        mask[:, 0] = True # depot
        skip_logprob = torch.zeros((batch_size, 1), dtype=bool).to(self.device)
        log_prob = 0
        # c = 0
        # print(env.complete(), c)
        while env.complete() == False:
            # c += 1
            # print(env.complete(), c)
            # idx: (batch_size, 1)
            # prob: (batch_size)
            # gru_hidden: (gru_num_layers, batch_size, hidden_dim)
            if decode_type == "sampling": 
                idx, prob, gru_hidden = self.sequentialDecoderSample.forward(h_n, last_node, gru_hidden, env, mask)
            elif decode_type == "greedy":
                # with torch.no_grad():
                idx, prob, gru_hidden = self.sequentialDecoderGreedy.forward(h_n, last_node, gru_hidden, env, mask)
            env.step(idx)
            last_node = idx
            # print("prob: ", prob.shape, prob)
            # log_prob = log_prob + torch.log(prob)
            log_prob += torch.log(prob.masked_fill(skip_logprob, 1.0))
            # print("log_prob: ", log_prob.shape, log_prob)
            # mask = env.get_mask(idx)
            mask, skip_logprob = env.get_mask(idx)
        total_dist = env.get_time()
        matrix = env.decode_routes()
        return total_dist, log_prob, matrix
    
    def forward(self, env):
        """
        make a forward
        encode the env info by GCN encoder
        decode the embedding info and get the routes
        """        
        n_coor = env.coor 
        # print("n_coor: ", n_coor.shape)
        # node info is the env.demand, env.reward, and env.loat_time, each with size (batch_size, node_num), the result should be (batch_size, node_num, 3)
        # first make the env.demand, env.reward, and env.loat_time to (batch_size, node_num, 1)
        # then stack them together
        n_info = env.demand.unsqueeze(2)
        # print("n_info: ", n_info.shape)
        dist = env.dist
        
        # GCN encoder
        h_n, h_e = self.GCNEncoder.forward(n_coor, n_info, dist)
        batch_size, node_num, hidden_dim = h_n.shape
        
        # sequential decoder
        # sampling
        sampling_total_dist, sampling_log_prob, sampling_matrix = self.seqDecoderForward(env, h_n, decode_type="sampling")
        # greedy
        greedy_total_dist, greedy_log_prob, greedy_matrix = self.seqDecoderForward(env, h_n, decode_type="greedy")
        # predict matrix
        predict_matrix = self.classificationDecoder.forward(h_e)
        # baseline sol
        baseline_matrix = env.decode_baseline_sol()
        baseline_dist = env.baseline_obj
        
        return sampling_log_prob, sampling_total_dist, greedy_total_dist, sampling_matrix, predict_matrix, greedy_matrix, baseline_matrix, baseline_dist