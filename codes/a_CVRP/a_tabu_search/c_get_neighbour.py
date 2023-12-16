from copy import deepcopy
class Neighborhood:
    
    def __init__(self, para):
        self.distance = para["graph"]["distance"]
        self.num_nodes = para["graph"]["dimension"]
        self.num_route = para["graph"]["trucks"]

    def insertCurNode(self, route, node):
        # insert current node based only on distance cost
        # we insert a node in a route 
        # we determine the location by minimizing the additional distance
        # additional cost = dist(node, prev) + dist(node, next) - dist(prev, next)
        def _computeCost(prev_n, next_n):
            return self.distance[prev_n][node] + self.distance[next_n][node] - self.distance[prev_n][next_n]
        cost_list = list(map(_computeCost, route[:-1], route[1:]))
        min_cost_loc = cost_list.index(min(cost_list)) + 1
        route.insert(min_cost_loc, node)
        return route

    def getExchangeNeighbour(self, cur_sol):
        result, result_idx = [None]*self.num_nodes*self.num_route, 0
        for remove_idx in range(len(cur_sol)):
            if len(cur_sol[remove_idx]) > 2:
                remove_route = deepcopy(cur_sol[remove_idx])
                for remove_node_idx in range(1,len(cur_sol[remove_idx])-1):
                    new_remove_route = remove_route[:remove_node_idx]+remove_route[remove_node_idx+1:]
                    remove_node = remove_route[remove_node_idx]
                    for insert_idx in range(len(cur_sol)):
                        if remove_idx != insert_idx:
                            insert_route = deepcopy(cur_sol[insert_idx])
                            new_insert_route = self.insertCurNode(insert_route, remove_node)
                            sub_result = [None]*len(cur_sol)
                            for i in range(len(cur_sol)):
                                if i == remove_idx: sub_result[i] = new_remove_route
                                elif i == insert_idx: sub_result[i] = new_insert_route
                                else: sub_result[i] = cur_sol[i]
                            # print(sub_result, result_idx)
                            result[result_idx] = (sub_result, (remove_node, remove_idx, insert_idx))
                            result_idx += 1
        none_idx = 0
        while result[none_idx-1] == None: none_idx -= 1
        result = result[:none_idx]
        return result