import torch
import numpy as np

class Environment:
    
    def __init__(self, dimension, capacity, coor, dist, demand, or_tools_sol, or_tools_obj):
        """
        @param dimension: (batch_size) the number of nodes
        @param capacity: (batch_size) the capacity of vehicle
        @param coor: (batch_size, node_num(N+1), 2)
        @param dist: (batch_size, node_num(N+1), node_num(N+1))
        @param demand: (batch_size, node_num(N+1))
        @param or_tools_sol: (batch_size, 1), just like routes
        @param or_tools_obj: (batch_size, 1), the objective value of or_tools_sol
        """
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.coor = coor
        self.dist = dist
        self.demand = demand
        
        self.batch_size = coor.shape[0]
        self.node_num = dimension[0]
        self.capacity = capacity
        
        self.baseline_sol = or_tools_sol
        self.baseline_obj = or_tools_obj
        
        # self.coor = self.coor / np.sqrt(1e5)
        # self.dist = self.dist / 1e5
        # self.baseline_obj = self.baseline_obj / 1e5
        
        self.reset()
        
    def init_state(self):
        """
        routes: (batch_size, 1)
        remaining_capacity: (batch_size, 1)
        remaining_time: (batch_size, 1)
        collected_demand: (batch_size, 1)
        shelf_demand: (batch_size, node_num)
        visited: (batch_size, node_num)
        """
        routes = torch.full((self.batch_size, 1), 0, dtype=torch.long, device=self.device)    # (batch_size, 1)
        remaining_capacity = self.capacity.unsqueeze(1).clone().float().to(self.device)    # (batch_size, 1)        
        used_time = torch.full((self.batch_size, 1), 0, dtype=torch.float, device=self.device)    # (batch_size, 1
        collected_demand = torch.full((self.batch_size, 1), 0, dtype=torch.float, device=self.device)    # (batch_size, 1)
        shelf_demand = self.demand.clone().float().to(self.device)   # (batch_size, node_num)
        visited = torch.zeros((self.batch_size, self.node_num), dtype=torch.bool, device=self.device)    # (batch_size, node_num)
        visited[:, 0] = True    # depot is visited
        return routes, remaining_capacity, used_time, collected_demand, shelf_demand, visited
    
    def reset(self):
        self.routes, self.remaining_capacity, self.used_time, self.collected_demand, \
            self.shelf_demand, self.visited = self.init_state()
        
    def step(self, action):
        """update customer and vehicle states
        @param action: (batch_size, 1), the index of the next node to visit
        1. routes += action
        2. remaining_capacity = 
            if idx == 0: initial_capacity
            otherwise: max(0, remaining_capacity - demands[idx])
        3. shelf_demands[idx] = 0
        5. remaining_time -= dist[routes[-2], routes[-1]] + load_time[routes[-1]]
        6. collected_demand += demands[idx]
        8. visited[idx] = True
        """
        action = action.squeeze(-1)    # (batch_size, )
        # print("action.shape: ", action.shape)
        # print("self.remaining_capacity.shape: ", self.remaining_capacity.shape)
        # print("self.capacity.shape: ", self.capacity.shape)
        # 1. routes += action
        self.routes = torch.cat((self.routes, action.unsqueeze(-1)), dim=-1)    # (batch_size, time_step+1)
        # 2. remaining_capacity
        prev_capacity = self.remaining_capacity    # (batch_size, 1)
        curr_demand = self.shelf_demand.gather(1, action.unsqueeze(-1))    # (batch_size, 1)
        self.remaining_capacity[action==0] = self.capacity.unsqueeze(1).float().to(self.device)[action==0]    # (batch_size, 1)
        self.remaining_capacity[action!=0] = torch.maximum(torch.zeros(self.batch_size, 1, device=self.device)[action!=0], prev_capacity[action!=0] - curr_demand[action!=0])    # (batch_size, 1)
        # 3. remaining_demand
        self.shelf_demand.scatter_(1, action.unsqueeze(-1), 0)    # (batch_size, node_num)
        # 5. remaining_time
        prev_step = self.routes[:, -2:-1]    # (batch_size, 1)
        curr_step = self.routes[:, -1:]    # (batch_size, 1)
        self.used_time += self.dist_per_step(prev_step, curr_step)    # (batch_size, 1)
        # 6. collected_demand
        self.collected_demand += curr_demand    # (batch_size, 1)
        # 8. visited
        self.visited.scatter_(1, action.unsqueeze(1), True)
        
    def dist_per_step(self, prev_step, curr_step):
        """
        @param prev_step: (batch_size, 1)
        @param curr_step: (batch_size, 1)
        @return: distance of single step (batch_size, 1)
        """
        idx = torch.arange(start=0, end=self.batch_size, step=1).unsqueeze(1)
        dist = self.dist[idx, prev_step, curr_step]
        return dist
    
    def get_time(self):
        '''
        loss is defined as the total distance
        '''
        return self.used_time.clone()
        
    def get_mask(self, last_action):
        ''' compute the mask for current states
        cannot visit the visited node
        @param last_action: (batch_size, 1)
        1. if shelf_demands[idx] >= remaining_capacity: set idx mask True
        2. if last_idx == 0: set the warehouse mask True
        4. if mask is all True: set the warehouse mask False
        '''
        mask = self.visited.clone()
        mask[:, 0] = False
        last_action = last_action.squeeze(-1)    # (batch_size, )
        # 1. 
        mask[(self.shelf_demand>=self.remaining_capacity)] = True
        # 2.
        mask[last_action==0, 0] = True
        skip_logprob = mask.all(dim=1)
        # 4.
        mask[mask.all(dim=1), 0] = False
        return mask, skip_logprob
    
    def complete(self):
        """
        @return: whether all the nodes are visited
        """
        visit_all_nodes = (self.visited == True).all()
        back_to_depot = (self.routes[:, -1] == 0).all()
        return (visit_all_nodes and back_to_depot)
    
    def decode_routes(self):
        """decode route sequence_to_matrix, will be used in the classification decoder
        @return (batch_size, node_num, node_num)
        """
        matrix = torch.zeros((self.batch_size, self.node_num, self.node_num), dtype=torch.float, device=self.device)    # (batch_size, node_num, node_num)
        idx = torch.arange(start=0, end=self.batch_size, step=1, device=self.device).unsqueeze(-1)    # (batch_size, 1)
        for i in range(1, self.routes.size(-1)):
            prev_step = self.routes[:, (i-1):i]    # (batch_size, 1)
            curr_step = self.routes[:, i:(i+1)]    # (batch_size, 1)
            matrix[idx, prev_step, curr_step] = 1
        return matrix.long()

    def decode_baseline_sol(self):
        matrix = torch.zeros((self.batch_size, self.node_num, self.node_num), dtype=torch.float, device=self.device)    # (batch_size, node_num, node_num)
        sol_max_len = self.baseline_sol.shape[1]
        idx = torch.arange(start=0, end=self.batch_size, step=1, device=self.device).unsqueeze(-1)    # (batch_size, 1)
        for i in range(1, sol_max_len):
            prev_step = self.baseline_sol[:, (i-1):i]
            curr_step = self.baseline_sol[:, i:(i+1)]
            matrix[idx, prev_step, curr_step] = 1
        return matrix.long()
            

        
        
        