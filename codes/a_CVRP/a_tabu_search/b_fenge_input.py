import json
import math
from copy import deepcopy
import sys
# sys.path.append("../")

class FengEInfo:
    
    def __init__(self):
        routes_path   = r""
        distance_path = r""
        with open(routes_path, "r") as f: 
            routes_dict = json.load(f)
        with open(distance_path, "r") as f: 
            distance_dict = json.load(f)
        self.routes   = routes_dict
        self.distance = distance_dict
        self.capacity_demand_ratio = 2
        self.distance_upperbound = 99999999
        self.duration_upperbound = 99999999

    def readRouteInstance(self, index):
        index = str(index)
        route = self.routes[index]
        missing = False

        # initialize the nodes, demand, profit, distance, capacity, trucks
        node = [None]*(len(route["shelf"])+1)
        node_demand = [0.0]*(len(route["shelf"])+1)
        node_profit = [0.0]*(len(route["shelf"])+1)
        node_distance = [[0.0]*(len(route["shelf"])+1) for _ in range((len(route["shelf"])+1))]
        node_duration = [[0.0]*(len(route["shelf"])+1) for _ in range((len(route["shelf"])+1))]
        capacity = 0.0
        trucks = 0.0

        # capacity, delivery_vehicle
        delivery_vehicle = route["manager"]["delivery_vehicle"]
        capacity = route["manager"]["max_transport"]

        # demand and profit 
        shelf_keys = list(route["shelf"].keys())
        node[0] = route["manager"]["warehouse_id"]
        for i in range(1, (len(route["shelf"])+1)):
            node[i] = shelf_keys[i-1]
            node_demand[i] = int(route["shelf"][node[i]]["demand"])
            node_profit[i] = float(route["shelf"][node[i]]["profit"])
        
        # num of trucks
        trucks = math.ceil((sum(node_demand) / capacity) * self.capacity_demand_ratio)

        # the distance matrix
        for i in range((len(route["shelf"])+1)):
            for j in range((len(route["shelf"])+1)):
                if i == j: pass
                else: 
                    try:
                        if delivery_vehicle == 1:
                            node_distance[i][j] = self.distance[node[i]][node[j]]["distance"]["pedelec"]
                            node_duration[i][j] = self.distance[node[i]][node[j]]["duration"]["pedelec"]
                        else:
                            node_distance[i][j] = self.distance[node[i]][node[j]]["distance"]["car"]
                            node_duration[i][j] = self.distance[node[i]][node[j]]["duration"]["car"]
                    except:
                        node_distance[i][j] = self.distance_upperbound
                        node_duration[i][j] = self.duration_upperbound
                        missing = True

        # the best route suggested by fenge currently
        fenge_best = deepcopy(route["route"])
        for k in fenge_best.keys():
            r = fenge_best[k]
            if r[0] != route["manager"]["warehouse_id"]:
                r = [route["manager"]["warehouse_id"]] + r
            r.append(r[0])
            for n in r:
                if n not in node: missing = True
            fenge_best[k] = r
        num_of_suggestions = len(list(fenge_best.keys()))

        # the profit and demand of the above suggested route
        fenge_best_demand   = []
        fenge_best_profit   = []
        fenge_best_distance = []
        fenge_best_duration = []
        if missing == False:
            for k in fenge_best.keys(): 
                a_profit   = 0
                a_demand   = 0
                a_distance = 0
                a_duration = 0
                r = fenge_best[k]
                for i in range(len(r)-1):
                    a_profit += node_profit[node.index(r[i])]
                    a_demand += node_demand[node.index(r[i])]
                    a_distance += node_distance[node.index(r[i])][node.index(r[i+1])]
                    a_duration += node_duration[node.index(r[i])][node.index(r[i+1])]
                fenge_best_profit.append(a_profit)
                fenge_best_demand.append(a_demand)
                fenge_best_distance.append(a_distance)
                fenge_best_duration.append(a_duration)

        # for d in fenge_best_demand:
        #     if d > capacity:
        #         # print("in the instance - {} | the capacity - {} | the demand of suggested route - {}".format(index, capacity, d))
        #         # if len(fenge_best_demand) == 1: 
        #         #     print(route["day"], route["manager"], fenge_best, fenge_best_demand)
        #         #     print("---"*10)
        #         missing = True

        # scale the demand list
        # demand_avg_in_record = sum(fenge_best_demand)/len(fenge_best_demand)
        # scale = capacity/demand_avg_in_record
        # scale = capacity/max(fenge_best_demand)
        # node_demand = [scale*d for d in node_demand]
        # fenge_best_demand = [scale*d for d in fenge_best_demand]
        
        result_dict  = {}
        result_dict["name"]      = index
        result_dict["capacity"]  = capacity
        result_dict["dimension"] = len(node_demand)
        result_dict["distance"]  = node_distance
        result_dict["duration"]  = node_duration
        result_dict["demand"]    = node_demand
        result_dict["profit"]    = node_profit
        result_dict["trucks"]    = trucks

        result_dict["nodes"]               = node
        result_dict["fenge_best"]          = fenge_best
        result_dict["fenge_best_demand"]   = fenge_best_demand
        result_dict["fenge_best_profit"]   = fenge_best_profit
        result_dict["fenge_best_distance"] = fenge_best_distance
        result_dict["fenge_best_duration"] = fenge_best_duration
        result_dict["num_of_suggestion"]   = num_of_suggestions
        result_dict["missing"]             = missing

        return result_dict


