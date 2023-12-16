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
from torch.nn import CrossEntropyLoss, BCELoss, NLLLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

class SupervisedModel(nn.Module):
    
    def __init__(self, hidden_dim, gcn_num_layers, k, node_info_dim, gru_num_layers, mlp_num_layers):
        super(SupervisedModel, self).__init__()
        
        self.gru_num_layers = gru_num_layers
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.GCN = GCN(hidden_dim=hidden_dim, gcn_num_layers=gcn_num_layers, k=k, node_info_dim=node_info_dim).to(self.device)
        self.classification_decoder = ClassificationDecoder(hidden_dim=hidden_dim, hidden_dim_MLP=hidden_dim, num_layers_MLP=mlp_num_layers).to(self.device)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, env, or_tools_sol, criterion_classification):
        
        # GCN
        h_n, h_e = self.GCN(env.coor, env.demand.unsqueeze(2), env.dist)
        batch_size, node_num, hidden_dim = h_n.shape
            
        # classification decoder
        predict_matrix = self.classification_decoder.forward(h_e)
        num_ones = countPredictedOnes(predict_matrix)
        target_matrix = decode_baseline_sol(batch_size, node_num, or_tools_sol, device)
        target_matrix[:, 0, 0] = 0
        predict_matrix = predict_matrix.view(-1, 2)
        target_matrix = target_matrix.view(-1)
        L_c = criterion_classification(predict_matrix.to(device), target_matrix.long().to(device))
        
        return L_c, num_ones

    def forwardTest(self, env, or_tools_sol, criterion_classification):
        
        # GCN
        h_n, h_e = self.GCN(env.coor, env.demand.unsqueeze(2), env.dist)
        batch_size, node_num, hidden_dim = h_n.shape
            
        # classification decoder
        predict_matrix = self.classification_decoder.forward(h_e)
        num_ones = countPredictedOnes(predict_matrix)
        target_matrix = decode_baseline_sol(batch_size, node_num, or_tools_sol, device)

        # construct the routes
        env.reset()
        last_node = torch.zeros((batch_size, 1)).long().to(self.device) # (batch_size, 1)
        mask = torch.zeros((batch_size, node_num)).bool().to(self.device) # (batch_size, node_num)
        
        while env.complete() == False:
            probs = predict_matrix[torch.arange(batch_size), last_node.squeeze(), :, 1].squeeze() # (batch_size, 1)
            probs.masked_fill_(mask, -np.inf)
            probs = self.softmax(probs)
            probs = probs.clamp(min=1e-8)
            idx = torch.argmax(probs, dim=1).unsqueeze(1) # (batch_size, 1)
            env.step(idx)
            last_node = idx
            mask, _ = env.get_mask(idx)
        total_dist = env.get_time()        

        # compute the loss
        target_matrix[:, 0, 0] = 0
        predict_matrix = predict_matrix.view(-1, 2)
        target_matrix = target_matrix.view(-1)
        L_c = criterion_classification(predict_matrix.to(device), target_matrix.long().to(device))
        
        return L_c, num_ones, total_dist

def supervisedPreTrain(mode, epoch, data_loader, model, criterion_classification, optimizer, result_path, batch_steps):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if mode == "train": model.train()
    elif mode == "test": model.eval()
    else: 
        print("ERROR")
        return None
    
    loss_total = []
    loss_classification = []
    num_predicted_ones = []
    route_dist = []

    for b, item in enumerate(data_loader):
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        
        if mode == "train":
            L_c, num_ones = model.forward(env, or_tools_sol, criterion_classification)
        elif mode == "test":
            with torch.no_grad():
                L_c, num_ones, total_dist = model.forwardTest(env, or_tools_sol, criterion_classification)
            route_dist.append(np.mean(total_dist.squeeze().detach().cpu().numpy()))
        
        loss = L_c
        loss_total.append(loss.item())
        loss_classification.append(L_c.item())
        num_predicted_ones.append(num_ones)
        
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            if b % 100 == 0:
                info = "e {:>4d}|b {:>4d}/{:>4d}|L_c {}|L {}|num_predicted_ones {}".format(epoch, b, batch_steps, loss_classification[-1], loss_total[-1], num_predicted_ones[-1])
                print(info)
                with open(result_path, "a") as f:
                    f.write(info + "\n")
        
    loss_total = np.array(loss_total)
    loss_classification = np.array(loss_classification)
    num_predicted_ones = np.array(num_predicted_ones)
    if mode == "test": route_dist = np.array(route_dist).reshape(-1)
    if mode == "train":
        info = "L_c {}|L {}|num_predicted_ones {}".format(np.mean(loss_classification), np.mean(loss_total), np.mean(num_predicted_ones))
    elif mode == "test":
        info = "L_c {}|L {}|num_predicted_ones {}|route_dist {}".format(np.mean(loss_classification), np.mean(loss_total), np.mean(num_predicted_ones), np.mean(route_dist))
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
        
    if arg_train.model_path_classification_decoder != "":
        model.classification_decoder.load_state_dict(torch.load(arg_train.model_path_classification_decoder, map_location=device))
        print("classification decoder is loaded from - {}".format(arg_train.model_path_classification_decoder))

    # criterion_classification = BCELoss()
    n = cfg.n_customer + 1
    weight_0 = n**2 / ((n**2 - 2*n) * 2)
    weight_1 = n**2 / (2*n * 2)
    
    # if n == 6:
    #     weight_0 = n**2 / ((n**2 - 7.16) * 2)
    #     weight_1 = n**2 / (7.16 * 2)
    # else:
    #     tot_edge = n + n/7
    #     weight_0 = n**2 / ((n**2 - tot_edge) * 2)
    #     weight_1 = n**2 / (tot_edge * 2)
    
    edge_class_weight = torch.tensor([weight_0, weight_1], dtype=torch.float, device=device)
    criterion_classification = CrossEntropyLoss(weight=edge_class_weight)
    optimizer = Adam(model.parameters(), lr=cfg.pretrain_lr)
    
    task = 'VRP%d_'%(cfg.n_customer)
    dump_date = datetime.now().strftime('%m%d_%H_%M')
    os.makedirs(cfg.result_dir+"pretrain/", exist_ok = True)
    result_path = cfg.result_dir + "pretrain/" + task + dump_date + ".txt"
    
    num_epoch = cfg.pretrain_epoch
    for i in range(num_epoch):
        
        supervisedPreTrain(
            mode="train",
            epoch=i+1,
            data_loader=train_data_loader,
            model=model,
            criterion_classification=criterion_classification,
            optimizer=optimizer,
            result_path=result_path,
            batch_steps=cfg.batch_steps
        )
        
        supervisedPreTrain(
            mode="test",
            epoch=i+1,
            data_loader=test_data_loader,
            model=model,
            criterion_classification=criterion_classification,
            optimizer=optimizer,
            result_path=result_path,
            batch_steps=cfg.batch_steps
        )
        
        if i % 10 == 0:
            os.makedirs(cfg.weight_dir+"pretrain/", exist_ok = True)
            # weights_file_path = cfg.weight_dir + "pretrain/" + "checkpoints_epoch_" + str(i+1)
            weights_file_path = cfg.weight_dir + "pretrain/" + "checkpoints"
            torch.save(model.GCN.state_dict(), weights_file_path + "_GCN.pth")
            torch.save(model.classification_decoder.state_dict(), weights_file_path + "_classification_decoder.pth")
    