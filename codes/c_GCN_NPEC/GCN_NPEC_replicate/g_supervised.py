from .a_config import train_parser, load_pkl, Config
from .a_utilities import countPredictedOnes, decode_baseline_sol
from .c_GCN import GCN
from .c_decoder import SequencialDecoderSupervised, ClassificationDecoder
from .d_env import Environment
from .e_dataset import MyDataloader2

import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

class SupervisedModel(nn.Module):
    
    def __init__(self, hidden_dim, gcn_num_layers, k, node_info_dim, gru_num_layers, mlp_num_layers):
        super(SupervisedModel, self).__init__()
        
        self.gru_num_layers = gru_num_layers
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.GCN = GCN(hidden_dim=hidden_dim, gcn_num_layers=gcn_num_layers, k=k, node_info_dim=node_info_dim).to(self.device)
        self.sequential_decoder = SequencialDecoderSupervised(hidden_dim=hidden_dim, gru_num_layers=gru_num_layers).to(self.device)
        self.classification_decoder = ClassificationDecoder(hidden_dim=hidden_dim, hidden_dim_MLP=hidden_dim, num_layers_MLP=mlp_num_layers).to(self.device)
    
    def forward(self, env, or_tools_sol, criterion_sequential, criterion_classification):
        
        # GCN
        h_n, h_e = self.GCN(env.coor, env.demand.unsqueeze(2), env.dist)
        batch_size, node_num, hidden_dim = h_n.shape
        
        # sequential decoder
        L_s = 0
        env.reset()
        gru_hidden = torch.zeros((self.gru_num_layers, batch_size, self.hidden_dim)).to(self.device) # (gru_num_layers, batch_size, hidden_dim)
        for t in range(or_tools_sol.shape[1]-1):
            last_node = or_tools_sol[:, t].unsqueeze(1) # (batch_size, 1)
            target_node = or_tools_sol[:, t+1]
            mask, _ = env.get_mask(last_node)
            probs, gru_hidden = self.sequential_decoder.forward(h_n, last_node, gru_hidden, mask)
            L_s += criterion_sequential(probs, target_node)
            
        # classification decoder
        predict_matrix = self.classification_decoder.forward(h_e)
        num_ones = countPredictedOnes(predict_matrix)
        target_matrix = decode_baseline_sol(batch_size, node_num, or_tools_sol, device)
        target_matrix[:, 0, 0] = 0
        predict_matrix = predict_matrix.view(-1)
        target_matrix = target_matrix.view(-1)
        L_c = criterion_classification(predict_matrix.to(device), target_matrix.float().to(device))
        
        return L_s, L_c, num_ones
        
    def testForward(self, env, decode_type="greedy"):
        env.reset()
        # GCN
        h_n, h_e = self.GCN(env.coor, env.demand.unsqueeze(2), env.dist)
        batch_size, node_num, hidden_dim = h_n.shape
        # sequential decoder
        last_node = torch.zeros((batch_size, 1)).long().to(self.device) # (batch_size, 1)
        gru_hidden = torch.zeros((self.gru_num_layers, batch_size, self.hidden_dim)).to(self.device) # (gru_num_layers, batch_size, hidden_dim)
        mask = torch.zeros((batch_size, node_num)).bool().to(self.device) # (batch_size, node_num)
        mask[:, 0] = True # depot
        while env.complete() == False:
            probs, gru_hidden = self.sequential_decoder.forward(h_n, last_node, gru_hidden, mask)
            if decode_type == "greedy":
                idx = torch.argmax(probs, dim=1).unsqueeze(1)
            elif decode_type == "sampling":
                idx = torch.multinomial(probs, 1)
            env.step(idx)
            last_node = idx
            mask, _ = env.get_mask(last_node)
        total_dist = env.get_time()
        return total_dist
        

def supervisedPreTrain(mode, epoch, data_loader, model, criterion_sequential, criterion_classification, optimizer, result_path, batch_steps):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if mode == "train": model.train()
    elif mode == "test": model.eval()
    else: 
        print("ERROR")
        return None
    
    loss_total = []
    loss_sequential = []
    loss_classification = []
    num_predicted_ones = []

    for b, item in enumerate(data_loader):
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        
        L_s, L_c, num_ones = model.forward(env, or_tools_sol, criterion_sequential, criterion_classification)
        
        L_s *= 0.01
        loss = L_s
        # loss = L_s + L_c
        loss_total.append(loss.item())
        loss_sequential.append(L_s.item())
        loss_classification.append(0)
        # loss_classification.append(L_c.item())
        num_predicted_ones.append(num_ones)
        
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            # if b % 100 == 0:
            #     info = "e {:>4d}|b {:>4d}/{:>4d}|L_s {}|L_c {}|L {}|num_predicted_ones {}".format(epoch, b, batch_steps, loss_sequential[-1], loss_classification[-1], loss_total[-1], num_predicted_ones[-1])
            #     print(info)
            #     with open(result_path, "a") as f:
            #         f.write(info + "\n")
        
    loss_total = np.array(loss_total)
    loss_sequential = np.array(loss_sequential)
    loss_classification = np.array(loss_classification)
    num_predicted_ones = np.array(num_predicted_ones)
    info = "L_s {}|L_c {}|L {}|num_predicted_ones {}".format(np.mean(loss_sequential), np.mean(loss_classification), np.mean(loss_total), np.mean(num_predicted_ones))
    print(info)
    with open(result_path, "a") as f:
        f.write(info + "\n")
    
    if mode == "test":
        total_dist_greedy = model.testForward(env, decode_type="greedy")
        total_dist_sampling = model.testForward(env, decode_type="sampling")
        info = "TESTING | total_dist_greedy - {} | total_dist_sampling - {} | or_tools_obj - {}".format(total_dist_greedy.mean().item(), total_dist_sampling.mean().item(), or_tools_obj.float().mean().item())
        print(info)
        with open(result_path, "a") as f:
            f.write(info + "\n")
        
    return None


if __name__ == "__main__":
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    arg_train = train_parser()
    cfg = load_pkl(arg_train.path)
    train_data_loader = MyDataloader2(arg_train.data_path, batch_size=cfg.batch).dataloader()
    test_data_loader = MyDataloader2(arg_train.test_data_path, batch_size=cfg.batch).dataloader()
    
    model = SupervisedModel(
        hidden_dim=cfg.hidden_dim,
        gcn_num_layers=cfg.gcn_num_layers,
        k=cfg.k,
        node_info_dim=cfg.node_info_dim,
        gru_num_layers=cfg.gru_num_layers,
        mlp_num_layers=cfg.mlp_num_layers
    ).to(device)
    
    if arg_train.model_path_gcn != "":
        model.GCN.load_state_dict(torch.load(arg_train.model_path_gcn, map_location=device))
        print("GCN is loaded from - {}".format(arg_train.model_path_gcn))
        
    if arg_train.model_path_sequential_decoder != "":
        model.sequential_decoder.load_state_dict(torch.load(arg_train.model_path_sequential_decoder, map_location=device))
        print("sequential decoder is loaded from - {}".format(arg_train.model_path_sequential_decoder))
        
    if arg_train.model_path_classification_decoder != "":
        model.classification_decoder.load_state_dict(torch.load(arg_train.model_path_classification_decoder, map_location=device))
        print("classification decoder is loaded from - {}".format(arg_train.model_path_classification_decoder))
    print()
    
    criterion_sequential = CrossEntropyLoss()
    criterion_classification = BCELoss()
    optimizer = Adam(model.parameters(), lr=cfg.pretrain_lr)
    
    task = 'VRP%d_'%(cfg.n_customer)
    dump_date = datetime.now().strftime('%m%d_%H_%M')
    os.makedirs(cfg.result_dir+"pretrain/", exist_ok = True)
    result_path = cfg.result_dir + "pretrain/" + task + dump_date + ".txt"
    
    num_epoch = cfg.pretrain_epoch
    for i in range(num_epoch):
        
        supervisedPreTrain(
            mode="test",
            epoch=i+1,
            data_loader=test_data_loader,
            model=model,
            criterion_sequential=criterion_sequential,
            criterion_classification=criterion_classification,
            optimizer=optimizer,
            result_path=result_path,
            batch_steps=cfg.batch_steps
        )
        
        supervisedPreTrain(
            mode="train",
            epoch=i+1,
            data_loader=train_data_loader,
            model=model,
            criterion_sequential=criterion_sequential,
            criterion_classification=criterion_classification,
            optimizer=optimizer,
            result_path=result_path,
            batch_steps=cfg.batch_steps
        )
        
        
        if i % 10 == 0:
            os.makedirs(cfg.weight_dir+"pretrain/", exist_ok = True)
            weights_file_path = cfg.weight_dir + "pretrain/" + "checkpoints_epoch_" + str(i+1)
            torch.save(model.GCN.state_dict(), weights_file_path + "_GCN.pth")
            torch.save(model.sequential_decoder.state_dict(), weights_file_path + "_sequential_decoder.pth")
            torch.save(model.classification_decoder.state_dict(), weights_file_path + "_classification_decoder.pth")
    