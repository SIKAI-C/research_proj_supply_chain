from math import floor
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class InitialSolution:

    def __init__(self, para):
        self.para = para

    def _addCurNode(self, cur_result, cur_node):
        costs = [0]*len(cur_result)
        for i in range(len(cur_result)):
            prev = cur_result[i][-1]
            costs[i] = self.para["graph"]["distance"][prev][cur_node]+self.para["graph"]["distance"][cur_node][0]-self.para["graph"]["distance"][prev][0] 
        return costs

    def _greedyInitialSolution(self):
        result      = [[0] for _ in range(self.para["graph"]["trucks"])]
        tot_demand  = [0 for _ in range(self.para["graph"]["trucks"])]
        for node in range(1,self.para["graph"]["dimension"]):
            # print(tot_demand)
            costs = self._addCurNode(result, node)
            costs_id = sorted(range(len(costs)), key=lambda k:costs[k])
            add_node, idx = False, 0
            while idx < len(costs):
                if add_node == False and tot_demand[costs_id[idx]] + self.para["graph"]["demand"][node] <= self.para["graph"]["capacity"]: 
                    # print(costs_id[idx], idx)
                    # print(result)
                    result[costs_id[idx]].append(node)
                    # print(result)
                    tot_demand[costs_id[idx]] += self.para["graph"]["demand"][node]
                    add_node = True
                    break
                idx += 1
            if add_node == False:
                result[-1].append(node)
                tot_demand[-1] += self.para["graph"]["demand"][node]
        for route in result: route.append(0)
        return result

    def _OrtoolsSolution(self, num_vehicle="default"):
        
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
        
        graph = self.para["graph"]
        data  = {}
        data["demands"]            = graph["demand"]
        data["distance_matrix"]    = graph["distance"]
        if num_vehicle == "default":
            data["num_vehicles"]   = graph["trucks"]
        elif num_vehicle == "large":
            data["num_vehicles"]   = 50
        else:
            print("unknown num_vehicle")
        data["vehicle_capacities"] = [graph["capacity"]]*data["num_vehicles"]
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
        search_parameters.time_limit.FromSeconds(self.para["hyper_para"]["initial"]["ortools_time_limits"])
        solution = routing.SolveWithParameters(search_parameters)
        return _returnSolution(data, manager, routing, solution)

    def __FengESolutionParser(self, sol):
        distance = self.para["graph"]["distance"]
        duration = self.para["graph"]["duration"]
        demand   = self.para["graph"]["demand"]
        profit   = self.para["graph"]["profit"]
        route_distance = [0]*len(sol)
        route_duration = [0]*len(sol)
        route_demand   = [0]*len(sol)
        route_profit   = [0]*len(sol)
        for r_idx in range(len(sol)):
            r = sol[r_idx]
            for i in range(len(r)-1):
                route_distance[r_idx] += distance[r[i]][r[i+1]]
                route_duration[r_idx] += duration[r[i]][r[i+1]]
                route_demand[r_idx]   += demand[r[i]]
                route_profit[r_idx]   += profit[r[i]]
            route_duration[r_idx] += (len(r)-2)*600
        return route_distance, route_duration, route_demand, route_profit

    def __FengESolutionKnapsack(self, dura_list, prft_list):
        time_limit = 28000
        dp = [0 for i in range(time_limit+1)]
        res = [[] for i in range(time_limit+1)]
        for i in range(len(dura_list)):
            for j in range(time_limit, floor(dura_list[i]-1), -1):
                if dp[j] < dp[floor(j-dura_list[i])]+prft_list[i]:
                    dp[j] = dp[floor(j-dura_list[i])]+prft_list[i]
                    res[j] = res[floor(j-dura_list[i])]+[i]
        return dp[time_limit], res[time_limit]
    
    # def __FengESolutionGreedyJoint(self, sol, dist_list, dura_list, demd_list, prft_list):
    #     pass

    def _FengESolution(self):
        # 设置 num_vehicle 为一个经验上保证有feasible solution但是不是很大的值
        # 得到的线路，大概的demand都在capacity附近
        # 对得到的解进行背包问题
        try:
            sol = self._OrtoolsSolution(num_vehicle="default")
        except:
            sol = self._OrtoolsSolution(num_vehicle="large")
        dist_list, dura_list, demd_list, prft_list = self.__FengESolutionParser(sol)
        max_prft, knapsack_sol = self.__FengESolutionKnapsack(dura_list, prft_list)
        return [dist_list, dura_list, demd_list, prft_list, max_prft, knapsack_sol, sol, self.para["graph"]["capacity"]]

    def getInitialSolution(self):
        if self.para["hyper_para"]["initial"]["type"] == "default":
            return self._greedyInitialSolution()
        elif self.para["hyper_para"]["initial"]["type"] == "ortools":
            return self._OrtoolsSolution()
        elif self.para["hyper_para"]["initial"]["type"] == "FengE":
            return self._FengESolution()
        elif self.para["hyper_para"]["initial"]["type"] == "FengE_joint":
            return self._FengESolutionJoint()
        else:
            print("unknown initial option")
            return None

def printFengESolution(sol):
    print("\033[1;30;47m{}\033[0m".format("distance - "), end = " ")
    for i in range(len(sol[0])):
        if i in sol[5]: print("\033[37;41m{}\033[0m".format(sol[0][i]), end=" ")
        else: print(sol[0][i], end=" ")
    print()
    print("\033[1;30;47m{}\033[0m".format("duration - "), end = " ")
    for i in range(len(sol[1])):
        if i in sol[5]: print("\033[37;41m{}\033[0m".format(sol[1][i]), end=" ")
        else: print(sol[1][i], end=" ")
    print()
    print("\033[1;30;47m{}\033[0m".format("demand - "), end = " ")
    for i in range(len(sol[2])):
        if i in sol[5]: print("\033[37;41m{}\033[0m".format(sol[2][i]), end=" ")
        else: print(sol[2][i], end=" ")
    print()
    print("\033[1;30;47m{}\033[0m".format("profit - "), end = " ")
    for i in range(len(sol[3])):
        if i in sol[5]: print("\033[37;41m{}\033[0m".format(sol[3][i]), end=" ")
        else: print(sol[3][i], end=" ")
    print()
    print("\033[1;30;47m{}\033[0m".format("max profit - "), end = " ")
    print("\033[37;43m{}\033[0m".format(sol[4]))
    print("\033[1;30;47m{}\033[0m".format("selected route - "), end = " ")
    print(sol[5])
    print("\033[1;30;47m{}\033[0m".format("solution - "))
    for i in range(len(sol[6])):
        if i in sol[5]: print("        ","\033[37;41m{}\033[0m".format(sol[6][i]))
        # else: print("        ", sol[6][i])
    print("\033[1;30;47m{}\033[0m".format("capacity - "), end = " ")
    print(sol[7])