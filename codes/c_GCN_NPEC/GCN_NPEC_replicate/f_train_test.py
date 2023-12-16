from .a_utilities import countPredictedOnes, countBaselineOnes, plotDuringTraining
from .d_env import Environment

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from scipy.stats import ttest_ind
import numpy as np

def train(epoch, train_data_loader, model, criterion, optimizer, batch_size, result_path, batch_steps, alpha=1, beta=1):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.train()

    t_tests_stat = []    # paried t-test to determine whether update the rollout policy
    t_tests_pvalue = []  # paried t-test to determine whether update the rollout policy
    s_dist = []    # reward of the sampling policy
    g_dist = []  # reward of the greedy policy
    b_dist = []      # the objective of the baseline value
    r_loss = []     # loss of the REINFORCE
    s_loss = []     # loss of the SUPERVISED
    t_loss = []     # total loss
    num_predicted_ones = []
    num_baseline_ones = []

    # if epoch % 2 == 0:
    #     model.GCNEncoder.requires_grad_(False)
    #     model.sequentialDecoderSample.requires_grad_(False)
    #     model.classificationDecoder.requires_grad_(True)
    #     alpha *= 0
    #     beta *= 10
    # else:
    #     model.GCNEncoder.requires_grad_(True)
    #     model.sequentialDecoderSample.requires_grad_(True)
    #     model.classificationDecoder.requires_grad_(True)

    model.GCNEncoder.requires_grad_(True)
    model.sequentialDecoderGreedy.requires_grad_(False)
    model.sequentialDecoderSample.requires_grad_(True)
    model.classificationDecoder.requires_grad_(True)

    # for item in tqdm(train_data_loader, "train"):
    # torch.autograd.set_detect_anomaly(True)
    # print(len(train_data_loader))
    for b, item in enumerate(train_data_loader):        
        
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        sampling_logprob, sampling_dist, greedy_dist, target_matrix, predict_matrix, greedy_matrix, baseline_matrix, baseline_dist = model(env)
        # print("baseline_matrix.shape: ", baseline_matrix.shape)
        # print("baseline_obj.shape: ", baseline_obj.shape)
        # print("sampling_logprob.shape: ", sampling_logprob.shape)
        # print("sampling_dist.shape: ", sampling_dist.shape)
        
        # predict matrix
        num_predicted_ones.append(countPredictedOnes(predict_matrix))
        predict_matrix = predict_matrix.view(-1, 2)
        # target matrix
        target_matrix[:, 0, 0] = 0
        target_matrix = target_matrix.view(-1)
        # baseline matrix
        baseline_matrix[:, 0, 0] = 0
        num_baseline_ones.append(countBaselineOnes(baseline_matrix))
        baseline_matrix = baseline_matrix.view(-1)
        # Eq. 21
        advantage = (sampling_dist - greedy_dist).detach().squeeze(dim=1)
        reinforce = advantage * sampling_logprob
        
        L_r = reinforce.mean()
        # Eq. 22
        # penalty for the difference of the classification decoder and the sequential decoder
        L_s = criterion(predict_matrix.to(device), target_matrix.long().to(device))
        # print("L_r: ", alpha * L_r.item())
        # print("L_s: ", beta * L_s.item())
        # Eq. 23
        loss = alpha * L_r + beta * L_s
        # print("loss: ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        # record the reward of the sampling policy and the greedy policy
        s_dist.append(np.mean(sampling_dist.squeeze().detach().cpu().numpy()))
        g_dist.append(np.mean(greedy_dist.squeeze().detach().cpu().numpy()))
        b_dist.append(np.mean(baseline_dist.squeeze().detach().cpu().numpy()))
        r_loss.append(L_r.item() * alpha)
        s_loss.append(L_s.item() * beta)
        t_loss.append(loss.item())

        if b % 100 == 0:
            info = "e {:>4d}|b {:>4d}/{:>4d}|s_dist {:.6f}|g_dist {:.6f}|b_dist {:.6f}|l {:.6f}|l_r {:.6f}|l_s {:.6f}|d {:.6f}|b_d {:.6f}"\
                .format(epoch, b, batch_steps, s_dist[-1], g_dist[-1], b_dist[-1], t_loss[-1], r_loss[-1], s_loss[-1], num_predicted_ones[-1], num_baseline_ones[-1])
            print(info)
            # record the reward
            with open(result_path, "a") as f:
                f.write(info + "\n")
    
    # paried t-test
    s_dist = np.array(s_dist).reshape(-1)
    g_dist = np.array(g_dist).reshape(-1)
    b_dist = np.array(b_dist).reshape(-1)
    r_loss = np.array(r_loss).reshape(-1)
    s_loss = np.array(s_loss).reshape(-1)
    t_loss = np.array(t_loss).reshape(-1)
    # print("s_reward shape: ", s_reward.shape)
    # print("g_reward.shape: ", g_reward.shape)
    ttest_stats, ttest_two_side_pval = ttest_ind(s_dist, g_dist)
    ttest_pval = ttest_two_side_pval / 2
    # print("mean of sampling reward", np.mean(s_reward))
    # print("mean of greedy reward", np.mean(g_reward))
    # print(ttest_stats)
    # print(ttest_pval)
    # is_update = ((np.mean(s_dist) < np.mean(g_dist)) and (ttest_pval < 0.05))
    is_update = (np.mean(s_dist) < np.mean(g_dist))
    print("*** UPDATE INFO ***")
    update_info = "s_dist {:.6f}|g_dist {:.6f}|b_dist {:.6f}|l {:.6f}|l_r {:.6f}|l_s {:.6f}|d {:.6f}|b_d {:.6f}|is_update {}"\
                .format(np.mean(s_dist), np.mean(g_dist), np.mean(b_dist), np.mean(t_loss), np.mean(r_loss), np.mean(s_loss), np.mean(num_predicted_ones), np.mean(num_baseline_ones), is_update)
    with open(result_path, "a") as f:
        f.write(update_info + "\n")
    print(update_info)
    return  is_update

def trainWithGradientAccumulation(epoch, train_data_loader, model, criterion, optimizer, batch_size, result_path, batch_steps, alpha=1, beta=1, accumulation_steps=4):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.train()
    optimizer.zero_grad()

    t_tests_stat = []    # paried t-test to determine whether update the rollout policy
    t_tests_pvalue = []  # paried t-test to determine whether update the rollout policy
    s_dist = []    # reward of the sampling policy
    g_dist = []  # reward of the greedy policy
    b_dist = []      # the objective of the baseline value
    r_loss = []     # loss of the REINFORCE
    s_loss = []     # loss of the SUPERVISED
    t_loss = []     # total loss
    num_predicted_ones = []
    num_baseline_ones = []

    model.GCNEncoder.requires_grad_(True)
    model.sequentialDecoderGreedy.requires_grad_(False)
    model.sequentialDecoderSample.requires_grad_(True)
    model.classificationDecoder.requires_grad_(True)

    for b, item in enumerate(train_data_loader):        
        
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        sampling_logprob, sampling_dist, greedy_dist, target_matrix, predict_matrix, greedy_matrix, baseline_matrix, baseline_dist = model(env)
        
        # predict matrix
        num_predicted_ones.append(countPredictedOnes(predict_matrix))
        predict_matrix = predict_matrix.view(-1)
        # target matrix
        target_matrix[:, 0, 0] = 0
        target_matrix = target_matrix.view(-1)
        # baseline matrix
        baseline_matrix[:, 0, 0] = 0
        num_baseline_ones.append(countBaselineOnes(baseline_matrix))
        baseline_matrix = baseline_matrix.view(-1)
        # Eq. 21
        advantage = (sampling_dist - greedy_dist).detach().squeeze(dim=1)
        reinforce = advantage * sampling_logprob
        
        L_r = reinforce.mean()
        # Eq. 22
        # penalty for the difference of the classification decoder and the sequential decoder
        L_s = criterion(predict_matrix.to(device), target_matrix.float().to(device))
        # Eq. 23
        loss = alpha * L_r + beta * L_s
        
        loss /= accumulation_steps
        loss.backward()
        
        if (b+1) % accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            optimizer.zero_grad()
        
        # record the reward of the sampling policy and the greedy policy
        s_dist.append(np.mean(sampling_dist.squeeze().detach().cpu().numpy()))
        g_dist.append(np.mean(greedy_dist.squeeze().detach().cpu().numpy()))
        b_dist.append(np.mean(baseline_dist.squeeze().detach().cpu().numpy()))
        r_loss.append(L_r.item() * alpha)
        s_loss.append(L_s.item() * beta)
        t_loss.append(loss.item() * accumulation_steps)

        # if b % 100 == 0:
        #     info = "e {:>4d}|b {:>4d}/{:>4d}|s_dist {:.6f}|g_dist {:.6f}|b_dist {:.6f}|l {:.6f}|l_r {:.6f}|l_s {:.6f}|d {:.6f}|b_d {:.6f}"\
        #         .format(epoch, b, batch_steps, s_dist[-1], g_dist[-1], b_dist[-1], t_loss[-1], r_loss[-1], s_loss[-1], num_predicted_ones[-1], num_baseline_ones[-1])
        #     print(info)
        #     # record the reward
        #     with open(result_path, "a") as f:
        #         f.write(info + "\n")
    
    # paried t-test
    s_dist = np.array(s_dist).reshape(-1)
    g_dist = np.array(g_dist).reshape(-1)
    b_dist = np.array(b_dist).reshape(-1)
    r_loss = np.array(r_loss).reshape(-1)
    s_loss = np.array(s_loss).reshape(-1)
    t_loss = np.array(t_loss).reshape(-1)
    ttest_stats, ttest_two_side_pval = ttest_ind(s_dist, g_dist)
    ttest_pval = ttest_two_side_pval / 2
    is_update = ((np.mean(s_dist) < np.mean(g_dist)) and (ttest_pval < 0.05))
    print("*** UPDATE INFO ***")
    update_info = "s_dist {:.6f}|g_dist {:.6f}|b_dist {:.6f}|l {:.6f}|l_r {:.6f}|l_s {:.6f}|d {:.6f}|b_d {:.6f}|is_update {}"\
                .format(np.mean(s_dist), np.mean(g_dist), np.mean(b_dist), np.mean(t_loss), np.mean(r_loss), np.mean(s_loss), np.mean(num_predicted_ones), np.mean(num_baseline_ones), is_update)
    with open(result_path, "a") as f:
        f.write(update_info + "\n")
    print(update_info)
    return  is_update

def test(test_data_loader, model, alpha, beta, criterion, result_path):
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    
    s_dist = []    # reward of the sampling policy
    g_dist = []  # reward of the greedy policy
    b_dist = []      # the objective of the baseline value
    r_loss = []     # loss of the REINFORCE
    s_loss = []     # loss of the SUPERVISED
    t_loss = []     # total loss
    num_predicted_ones = []
    num_baseline_ones = []
    
    for b, item in enumerate(test_data_loader):    
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        with torch.no_grad():
            sampling_logprob, sampling_dist, greedy_dist, target_matrix, predict_matrix, greedy_matrix, baseline_matrix, baseline_dist = model(env)
        # print("baseline_matrix.shape: ", baseline_matrix.shape)
        # print("baseline_obj.shape: ", baseline_obj.shape)
        
        # predict matrix
        num_predicted_ones.append(countPredictedOnes(predict_matrix))
        predict_matrix = predict_matrix.view(-1, 2)
        # target matrix
        target_matrix[:, 0, 0] = 0
        target_matrix = target_matrix.view(-1)
        # baseline matrix
        baseline_matrix[:, 0, 0] = 0
        num_baseline_ones.append(countBaselineOnes(baseline_matrix))
        baseline_matrix = baseline_matrix.view(-1)
        # Eq. 21
        advantage = (sampling_dist - greedy_dist).detach().squeeze(dim=1)
        reinforce = advantage * sampling_logprob
        
        L_r = reinforce.mean()
        # Eq. 22
        # penalty for the difference of the classification decoder and the sequential decoder
        L_s = criterion(predict_matrix.to(device), target_matrix.long().to(device))
        # print("L_r: ", alpha * L_r.item())
        # print("L_s: ", beta * L_s.item())
        # Eq. 23
        loss = alpha * L_r + beta * L_s
        # print("loss: ", loss.item())
        # record the reward of the sampling policy and the greedy policy
        
        s_dist.append(np.mean(sampling_dist.squeeze().detach().cpu().numpy()))
        g_dist.append(np.mean(greedy_dist.squeeze().detach().cpu().numpy()))
        b_dist.append(np.mean(baseline_dist.squeeze().detach().cpu().numpy()))
        r_loss.append(L_r.item() * alpha)
        s_loss.append(L_s.item() * beta)
        t_loss.append(loss.item())
    
    s_dist = np.array(s_dist).reshape(-1)
    g_dist = np.array(g_dist).reshape(-1)
    b_dist = np.array(b_dist).reshape(-1)
    r_loss = np.array(r_loss).reshape(-1)
    s_loss = np.array(s_loss).reshape(-1)
    t_loss = np.array(t_loss).reshape(-1)

    print("*** TESTING INFO ***")
    testing_info = "s_dist {:.6f}|g_dist {:.6f}|b_dist {:.6f}|l {:.6f}|l_r {:.6f}|l_s {:.6f}|d {:.6f}|b_d {:.6f}"\
                .format(np.mean(s_dist), np.mean(g_dist), np.mean(b_dist), np.mean(t_loss), np.mean(r_loss), np.mean(s_loss), np.mean(num_predicted_ones), np.mean(num_baseline_ones))
    print(testing_info)
    with open(result_path, "a") as f:
        f.write(testing_info + "\n")
    return  None

def testWithPolt(epoch, test_data_loader_for_plot, model, image_dir):
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    
    for b, item in enumerate(test_data_loader_for_plot):        
        dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj =\
            item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device), item[6].to(device)
        env = Environment(dimension, capacity, location, distance, demand, or_tools_sol, or_tools_obj)
        sampling_logprob, sampling_dist, greedy_dist, target_matrix, predict_matrix, greedy_matrix, baseline_matrix, baseline_dist = model(env)
        
        image_path = image_dir + "e_" + str(epoch) + "_b_" + str(b) + ".png"
        plotDuringTraining(dimension.item(), demand[0].cpu().numpy().tolist(), location[0].cpu().numpy().tolist(), target_matrix[0].cpu().numpy().tolist(), predict_matrix[0].squeeze().cpu().detach().numpy().tolist(), image_path)
        if b >= 10: break