from .a_config import train_parser, load_pkl, Config
from .a_utilities import countPredictedOnes, init_weights
from .c_GCN import GCN
from .c_decoder import ClassificationDecoder
from .d_env import Environment

import torch
from torch.nn import BCELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import numpy as np

def pretrainClassificationDecoderTrain(epoch, train_data_loader, model, criterion, optimizer, result_path, batch_steps):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.train()
    
    model.GCNEncoder.requires_grad_(False)
    model.sequentialDecoderSample.requires_grad_(False)
    model.sequentialDecoderGreedy.requires_grad_(False)
    model.classificationDecoder.requires_grad_(True)
    
    loss = []
    predicted_ones = []
    
    for b, item in enumerate(train_data_loader):
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        sampling_logprob, sampling_dist, greedy_dist, target_matrix, predict_matrix, greedy_matrix, baseline_matrix, baseline_dist = model(env)
        
        # ones
        predicted_ones.append(countPredictedOnes(predict_matrix))
        
        # loss
        predict_matrix = predict_matrix.view(-1)
        target_matrix[:, 0, 0] = 0
        target_matrix = target_matrix.view(-1)
        
        a_loss = criterion(predict_matrix.to(device), target_matrix.float().to(device))
        loss.append(a_loss.item())
        
        # step
        optimizer.zero_grad()
        a_loss.backward()
        optimizer.step()
        
        if b % 100 == 0:
            info = " PRETRAIN |e {:>4d}|b {:>4d}/{:>4d}|ones {}|loss {}".format(epoch, b, batch_steps, predicted_ones[-1], loss[-1])
            print(info)
            with open(result_path, "a") as f:
                f.write(info + "\n")
        
    train_info = " PRETRAIN |e {:>4d}|ones {}|loss {}".format(epoch, np.mean(predicted_ones), np.mean(loss))
    print(train_info)
    with open(result_path, "a") as f:
        f.write(train_info + "\n")
    return None

def pretrainClassificationDecoderTest(test_data_loader, model, criterion, result_path):    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    
    loss = []
    predicted_ones = []
    
    for b, item in enumerate(test_data_loader):
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        sampling_logprob, sampling_dist, greedy_dist, target_matrix, predict_matrix, greedy_matrix, baseline_matrix, baseline_dist = model(env)
        
        # ones
        predicted_ones.append(countPredictedOnes(predict_matrix))
        
        # loss
        predict_matrix = predict_matrix.view(-1)
        target_matrix[:, 0, 0] = 0
        target_matrix = target_matrix.view(-1)
        
        a_loss = criterion(predict_matrix.to(device), target_matrix.float().to(device))
        loss.append(a_loss.item())
        
    test_info = " PRETRAIN |e     |ones {}|loss {}".format(np.mean(predicted_ones), np.mean(loss))
    print(test_info)
    with open(result_path, "a") as f:
        f.write(test_info + "\n")
    return None


if __name__ == "__main__":
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    arg_train = train_parser()
    cfg = load_pkl(arg_train.path)
    train_data_loader = MyDataloader2(arg_train.data_path, batch_size=cfg.pretrain_batch).dataloader()
    test_data_loader = MyDataloader2(arg_train.test_data_path, batch_size=cfg.pretrain_batch).dataloader()
    
    print(" ######################################## ")
    print(" ### PRE-TRAIN CLASSIFICATION DECODER ### ")
    print(" ######################################## ")
    print()
    print(" *** DEVICE {} *** ".format(device))
    print()
    
    GCNEncoder = GCN(
        hidden_dim=cfg.hidden_dim,
        gcn_num_layers=cfg.gcn_num_layers,
        k=cfg.k,
        node_info_dim=cfg.node_info_dim        
    )
    GCNEncoder.requires_grad_(False)
    
    model = ClassificationDecoder(
        hidden_dim=cfg.hidden_dim,
        hidden_dim_MLP=cfg.hidden_dim,
        num_layers_MLP=cfg.mlp_num_layers
    ).to(device)
    
    if arg_train.model_path_classification_decoder == "":
        print("*** CLASSIFICATION DECODER - training start from initialization")
        model.apply(init_weights)
    else:
        print("*** CLASSIFICATION DECODER - training continues from the model - ", arg_train.model_path)
        model.load_state_dict(torch.load(arg_train.model_path_classification_decoder, map_location=device))
    print()
    
    if arg_train.model_path_gcn == "":
        print("*** GCN - just the random initial GCN")
        GCNEncoder.apply(init_weights)
    else:
        print("*** GCN - load the existing GCN - ")
        GCNEncoder.load_state_dict(torch.load(arg_train.model_path_gcn), map_location=device)
    print()
    
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=cfg.pretrain_lr)
    
    task = 'VRP%d_'%(cfg.n_customer)
    dump_date = datetime.now().strftime('%m%d_%H_%M')
    result_path = cfg.result_dir + "/pretrain_classification_decoder/" + task + dump_date + ".txt"
    
    num_epoch = cfg.pretrain_epoch
    for i in range(num_epoch):
        
        train(
            epoch = i+1,
            train_data_loader = train_data_loader,
            GCN = GCNEncoder,
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            result_path = result_path,
            batch_steps = cfg.pretrain_batch_steps
        )
        
        test(
            test_data_loader = test_data_loader,
            GCN = GCNEncoder,
            model = model,
            criterion = criterion,
            result_path = result_path
        )
        
        if i % 10 == 0:
            weights_file_path = cfg.weight_dir + "/pretrain_classification_decoder/checkpoints_epoch_" + str(i) + ".pth"
            torch.save(model.state_dict(), weights_file_path)