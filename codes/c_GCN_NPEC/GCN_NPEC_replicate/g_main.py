import os
from datetime import datetime

from .a_config import train_parser, load_pkl, Config
from .a_utilities import init_weights
from .d_model import Model
from .e_generate_instance import *
from .e_dataset import MyDataloader2
from .f_train_test import train, test, trainWithGradientAccumulation, testWithPolt
from .f_pretrain import pretrainClassificationDecoderTrain, pretrainClassificationDecoderTest
from torch.nn import CrossEntropyLoss, BCELoss, NLLLoss
from torch.optim import Adam

if __name__ == "__main__":
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("#####################")
    print("### DEVICE {} ###".format(device))
    print("#####################")
    
    arg_train = train_parser()
    cfg = load_pkl(arg_train.path)
    train_data_loader = MyDataloader2(arg_train.data_path, batch_size=cfg.batch).dataloader()
    test_data_loader = MyDataloader2(arg_train.test_data_path, batch_size=cfg.batch).dataloader()
    if arg_train.plot_data_path != '':
        test_data_loader_for_plot = MyDataloader2(arg_train.plot_data_path, batch_size=1).dataloader(shuffle=False)
    else:
        test_data_loader_for_plot = MyDataloader2(arg_train.test_data_path, batch_size=1).dataloader(shuffle=False)
        

    model = Model(
        hidden_dim=cfg.hidden_dim,
        gcn_num_layers=cfg.gcn_num_layers,
        k=cfg.k,
        node_info_dim=cfg.node_info_dim,
        gru_num_layers=cfg.gru_num_layers,
        mlp_num_layers=cfg.mlp_num_layers
    ).to(device)
            
    if arg_train.model_path == "":
        print("*** training start from initialization")
        model.apply(init_weights)
    else:
        print("*** training continues from the model - ", arg_train.model_path)
        model.load_state_dict(torch.load(arg_train.model_path, map_location=device))
        
    if arg_train.model_path_gcn != "":
        model.GCNEncoder.load_state_dict(torch.load(arg_train.model_path_gcn, map_location=device))
        
    if arg_train.model_path_sequential_decoder != "":
        model.sequentialDecoderSample.load_state_dict(torch.load(arg_train.model_path_sequential_decoder, map_location=device))
        model.sequentialDecoderGreedy.load_state_dict(torch.load(arg_train.model_path_sequential_decoder, map_location=device))
        
    if arg_train.model_path_classification_decoder != "":
        model.classificationDecoder.load_state_dict(torch.load(arg_train.model_path_classification_decoder, map_location=device))
    
    # criterion = BCELoss()
    n = cfg.n_customer + 1
    weight_0 = n**2 / ((n**2 - 2*n) * 2)
    weight_1 = n**2 / (2*n * 2)
    edge_class_weight = torch.tensor([weight_0, weight_1], dtype=torch.float, device=device)
    # criterion = NLLLoss(weight=edge_class_weight)
    criterion = CrossEntropyLoss(weight=edge_class_weight)

    # cycle learning rate
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False)

    task = '_VRP%d_'%(cfg.n_customer)
    dump_date = datetime.now().strftime('%m%d_%H_%M')
    result_path = cfg.result_dir + "training" + task + dump_date + ".txt"

    num_epoch = cfg.epochs
    lr = cfg.lr
    beta = cfg.beta
    for i in range(num_epoch):
        
        ############################################
        ### PRE-TRAIN THE CLASSIFICATION DECODER ###
        ############################################
        # pretrain_num_epoch = cfg.pretrain_epoch - 5 * i 
        # pretrain_num_epoch = 1
        # if pretrain_num_epoch > 0:
        #     pretrain_lr = cfg.pretrain_lr
        #     optimizer = Adam(model.classificationDecoder.parameters(), lr=pretrain_lr)
        #     for j in range(pretrain_num_epoch):
                
        #         pretrainClassificationDecoderTrain(
        #             epoch=j+1,
        #             train_data_loader=train_data_loader,
        #             model=model,
        #             criterion=criterion,
        #             optimizer=optimizer,
        #             result_path=result_path,
        #             batch_steps=cfg.batch_steps
        #         )
                
        #         pretrainClassificationDecoderTest(
        #             test_data_loader=test_data_loader,
        #             model=model,
        #             criterion=criterion,
        #             result_path=result_path
        #         ) 

        #############################
        ### TRAIN THE WHOLE MODEL ###
        #############################
        if lr <= 3 * 10 ** (-6): pass
        else: lr *= 0.96
        optimizer = Adam(model.parameters(), lr=lr)
        update = train(
            epoch=i+1, 
            train_data_loader=train_data_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=cfg.batch,
            result_path=result_path,
            batch_steps=cfg.batch_steps,
            alpha=cfg.alpha,
            beta=beta,
        )
        
        # update = trainWithGradientAccumulation(
        #     epoch=i+1, 
        #     train_data_loader=train_data_loader,
        #     model=model,
        #     criterion=criterion,
        #     optimizer=optimizer,
        #     batch_size=cfg.batch,
        #     result_path=result_path,
        #     batch_steps=cfg.batch_steps,
        #     alpha=cfg.alpha,
        #     beta=beta,
        #     accumulation_steps=4
        # )
        
        test(
            test_data_loader=test_data_loader,
            model=model,
            alpha=cfg.alpha,
            beta=beta,
            criterion=criterion,
            result_path=result_path,
        )
        
        # image_dir = cfg.result_dir + "image/"
        # os.makedirs(image_dir, exist_ok=True)
        # testWithPolt(
        #     epoch=i+1,
        #     test_data_loader_for_plot=test_data_loader_for_plot,
        #     model=model,
        #     image_dir=image_dir
        # )

        # if beta >= 0.001: beta *= 0.96
        
        # wheteher update the rollout policy
        # print(update)
        if update:
            model.sequentialDecoderGreedy.load_state_dict(model.sequentialDecoderSample.state_dict())
        #     weights_file_path = cfg.weight_dir + "updates_epoch_" + str(i+1) + ".pth"
        #     torch.save(model.state_dict(), weights_file_path)
            
        if i % 10 == 0:
            # weights_file_path = cfg.weight_dir + "checkpoints_epoch_" + str(i+1) + ".pth"
            weights_file_path = cfg.weight_dir + "checkpoints.pth"
            torch.save(model.state_dict(), weights_file_path)