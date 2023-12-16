import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from math import floor

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import matplotlib.pyplot as plt

def init_weights(m):
    """Initialize the weights of the model
    """    
    if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                   nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
        nn.init.uniform_(m.weight.data, -0.08, 0.08)
        if m.bias is not None:
            nn.init.uniform_(m.bias.data, -0.08, 0.08)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        nn.init.uniform_(m.weight.data, -0.08, 0.08)
        nn.init.uniform_(m.bias.data, -0.08, 0.08)
    elif type(m) in [nn.LSTM, nn.GRU, nn.RNN]:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.08, 0.08)
            else:  # 'bias' in name
                # Biases are usually initialized to zeros, but you can use uniform if you want.
                nn.init.uniform_(param.data, -0.08, 0.08)
                           
# the output of classification decoder 1s
def countPredictedOnes(predict_matrix):
    """count the number of ones in the predicted matrix
    @param predict_matrix (batch_size, node_num, node_num, 2), the output of the classification decoder
    """
    predicted_classes = torch.argmax(predict_matrix, dim=-1) # (batch_size, node_num, node_num)
    count_ones = (predicted_classes == 1).sum(dim=[1,2]).float() # (batch_size)
    # count_ones = (predict_matrix >= 0.5).sum(dim=[1,2]).float() # (batch_size)
    return torch.mean(count_ones).item()

# the output of classification decoder probs
def computePredictedProbs(predict_matrix, target_matrix):
    """based on the target_matrix
    compute the probs of the 1 in the target_matrix
    compute the probs of the 0 in the target_matrix

    Args:
        predict_matrix (batch_size * node_num * node_num, 2)
        target_matrix  (batch_size * node_num * node_num, 1)
    """
    
    idx_0 = (target_matrix == 0).nonzero(as_tuple=True)[0]
    idx_1 = (target_matrix == 1).nonzero(as_tuple=True)[0]
    
    avg_prob_0_class_0 = predict_matrix[idx_0, 0].mean()
    avg_prob_1_class_0 = predict_matrix[idx_0, 1].mean()
    avg_prob_0_class_1 = predict_matrix[idx_1, 0].mean()
    avg_prob_1_class_1 = predict_matrix[idx_1, 1].mean()
    
    predict_matrix_ndarray = predict_matrix.detach().cpu().numpy()
    target_matrix_ndarray = target_matrix.detach().cpu().numpy()
    auc = roc_auc_score(target_matrix_ndarray, predict_matrix_ndarray[:, 1])
    
    return (avg_prob_0_class_0.item(), avg_prob_1_class_0.item(), avg_prob_0_class_1.item(), avg_prob_1_class_1.item(), auc)

# the baseline solution computed by or-tools
# will be used in the beginning of the training process 
def ortoolsSolution(demand, distance, capacity):
    """call the or-tools to compute the baseline solution
    @param demand (node_num, ), the demand of each node
    @param distance (node_num, node_num), the distance between each pair of nodes
    @param capacity (int), the capacity of the vehicle
    """    
    
    # or-tools can only handle the integer
    demand = [int(d) for d in demand]
    distance = [[int(d) for d in row] for row in distance]
    capacity = int(capacity)
    
    
    def _returnSolution(data, manager, routing, solution):
        result = []
        for vehicle_id in range(data['num_vehicles']):
            sub_result = []
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                sub_result.append(node_index)
                index = solution.Value(routing.NextVar(index))
            sub_result.append(0)
            result.append(sub_result)
        return result

    data  = {}
    data["demands"]            = demand
    data["distance_matrix"]    = distance
    data["num_vehicles"]       = int(2 * sum(demand) / capacity) + 1
    data["vehicle_capacities"] = [capacity] * data["num_vehicles"]
    data["depot"]              = 0

    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)
    solution = routing.SolveWithParameters(search_parameters)
    return _returnSolution(data, manager, routing, solution), solution.ObjectiveValue()

# decode the baseline solution
def decode_baseline_sol(batch_size, node_num, baseline_sol, device):
    """from the baseline solution, decode the baseline matrix
    @param batch_size (int) hyper-parameter
    @param node_num (int) hyper-parameter
    @param baseline_sol (batch_size, sol_max_len), the baseline solution
    @param device (torch.device)
    """    
    matrix = torch.zeros((batch_size, node_num, node_num), dtype=torch.float, device=device)    # (batch_size, node_num, node_num)
    sol_max_len = baseline_sol.shape[1]
    idx = torch.arange(start=0, end=batch_size, step=1, device=device).unsqueeze(-1)    # (batch_size, 1)
    for i in range(1, sol_max_len):
        prev_step = baseline_sol[:, (i-1):i]
        curr_step = baseline_sol[:, i:(i+1)]
        matrix[idx, prev_step, curr_step] = 1
    return matrix.long()

# count 1s in the baseline matrix
def countBaselineOnes(baseline_matrix):
    count_ones = (baseline_matrix == 1).sum(dim=[1,2]).float() # (batch_size)
    return torch.mean(count_ones).item()

# draw the classification decoder's result for the toy example
def plotDuringTraining(dimension, demand, location, target_matrix, predict_matrix, image_path):
    """compare the result of the classification decoder and the result of sequential decoder
    @param dimension (int), the number of nodes (n+1)
    @param demand (dimension, ), the demand of each node
    @param location (dimension, 2), the location of each node
    @param target_matrix (dimension, dimension), the result of sequential decoder
    @param predict_matrix (dimension, dimension), the result of classification decoder
    @param image_path (str), the path to store the image
    """    
    
    # plot the shelfs
    for i in range(dimension):
        if i == 0: c = "red"
        else: c = "blue"
        plt.scatter(location[i][0], location[i][1], c=c)
        plt.text(location[i][0], location[i][1], str(demand[i]), c=c)
        
    # plot the routes
    for i in range(dimension):
        for j in range(dimension):
            if target_matrix[i][j] == 1:
                plt.plot([location[i][0], location[j][0]], [location[i][1], location[j][1]], c="black", linestyle="--")
                plt.text((location[i][0]+location[j][0])/2, (location[i][1]+location[j][1])/2, str(round(predict_matrix[i][j], 2)))
            if predict_matrix[i][j] >= 0.5:
                plt.plot([location[i][0], location[j][0]], [location[i][1], location[j][1]], c="yellow", alpha=predict_matrix[i][j])
                plt.text((location[i][0]+location[j][0])/2, (location[i][1]+location[j][1])/2, str(round(predict_matrix[i][j], 2)), c="brown")
    
    # the plot
    # plt.show()   
    plt.savefig(image_path)
    plt.clf()
