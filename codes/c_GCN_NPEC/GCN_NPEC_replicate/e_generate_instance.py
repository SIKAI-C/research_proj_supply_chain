from .a_config import data_parser, load_pkl, Config
from .a_utilities import ortoolsSolution

import numpy as np
import torch
import math
from tqdm import tqdm

def generateDataset_stage1(cfg, file_name, size=10240):
    
    def euclidean_distance(n1, n2):
        x1, y1, x2, y2 = n1[0], n1[1], n2[0], n2[1]
        if isinstance(n1, torch.Tensor):
            return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
        elif isinstance(n1, (list, np.ndarray)):
            return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
        else:
            raise TypeError
    
    node_num = cfg.n_customer
    dimension = node_num + 1
    capacity = 1500
    time_limit = 280000
    
    location = np.random.randint(0, 50000, size=(size, node_num+1, 2), dtype=np.int64)
    distance = np.zeros((size, node_num+1, node_num+1))
    for i in tqdm(range(size), desc="make the distance"):
        for j in range(node_num+1):
            for k in range(node_num+1):
                distance[i][j][k] = euclidean_distance(location[i][j], location[i][k])
    
    demand = np.random.randint(0, 200, size=(size, node_num))
    depot_demand = np.zeros((size, 1))
    demand = np.concatenate((depot_demand, demand), axis=1)
    
    reward_coef = np.random.uniform(0.3, 0.5, size=(size, node_num+1))
    reward = reward_coef * demand
    
    load_time = np.random.normal(600, 60, size=(size, node_num))
    depot_load_time = np.zeros((size, 1))
    load_time = np.concatenate((depot_load_time, load_time), axis=1)
    
    res_path = file_name + ".npz"

    dimension = np.array([dimension]*size)
    capacity = np.array([capacity]*size)
    time_limit = np.array([time_limit]*size)
    
    np.savez(res_path, dimension=dimension, capacity=capacity, location=location, distance=distance, demand=demand, reward=reward, load_time=load_time, time_limit=time_limit)

def generateDataset_stage2(cfg, file_name, size=10240):
    """generateDataset_stage1 + ortoolsSoluiton
    """    
    
    or_tools_sol_list = []
    or_tools_obj_list = []
    
    def euclidean_distance(n1, n2):
        x1, y1, x2, y2 = n1[0], n1[1], n2[0], n2[1]
        if isinstance(n1, torch.Tensor):
            return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
        elif isinstance(n1, (list, np.ndarray)):
            return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
        else:
            raise TypeError
    
    try: node_num = cfg.n_customer
    except: node_num = cfg
    dimension = node_num + 1
    capacity = 30
    
    demand = np.random.randint(0, 9, size=(size, node_num))    # for normal training
    # demand = np.random.randint(0, 20, size=(size, node_num))    # for toy example
    depot_demand = np.zeros((size, 1))
    demand = np.concatenate((depot_demand, demand), axis=1)
    
    res_path = file_name + ".npz"

    dimension = np.array([dimension]*size)
    capacity = np.array([capacity]*size)
    
    location = np.random.randint(0, 85000, size=(size, node_num+1, 2), dtype=np.int64)
    distance = np.zeros((size, node_num+1, node_num+1))
    for i in tqdm(range(size), desc="make the distance"):
        for j in range(node_num+1):
            for k in range(node_num+1):
                distance[i][j][k] = int(euclidean_distance(location[i][j], location[i][k]))
        
        # # ortools solution
        # or_tools_sol, or_tools_obj = ortoolsSolution(demand[i],  distance[i], capacity[i])
        # sol = []
        # for sol_i in or_tools_sol:
        #     if len(sol_i) > 2: sol += sol_i
        # or_tools_sol_list.append(sol)
        # or_tools_obj_list.append(or_tools_obj)
        
        # max_len = max(len(lst) for lst in or_tools_sol_list)
        # padded_or_tools_sol_list = [lst + [0]*(max_len - len(lst)) for lst in or_tools_sol_list]
        
        padded_or_tools_sol_list =  [[0, 0]]*size
        # or_tools_obj_list = np.array([4.69]*size)
        or_tools_obj_list = np.array([9.67]*size)
    
    location = location / np.sqrt(1e5) 
    distance = distance / 1e5
    # or_tools_obj_list = np.array(or_tools_obj_list, dtype=float) / 1e5
    
    np.savez(
                res_path, 
                dimension=dimension,
                capacity=capacity,
                location=location,
                distance=distance,
                demand=demand,
                or_tools_sol=np.array(padded_or_tools_sol_list),
                or_tools_obj=or_tools_obj_list,
            )


if __name__ == "__main__":
    cfg = load_pkl(data_parser().path)
    
    size = cfg.batch * cfg.batch_steps
    file_name = cfg.data_dir + "data_stage2_" + str(cfg.n_customer) + "_training"
    generateDataset_stage2(cfg, file_name, size=size)
    
    file_name = cfg.data_dir + "data_stage2_" + str(cfg.n_customer) + "_testing"
    generateDataset_stage2(cfg, file_name, size=int(0.2*size))
    