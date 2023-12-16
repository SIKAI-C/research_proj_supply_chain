import numpy as np

class Node:

    def __init__(self):
        self.ids                    = None       # will be a [], contains basic nodes in this logic nodes (or super nodes)
        self.entry                  = None       # will be an integer, denote the first basic nodes in this logic nodes
        self.exit                   = None       # will be an integer, denote the last basic nodes in this logic nodes
        self.internal_distance      = None       # will be a real number, denote the internal distance in this logic nodes, if the logic nodes only have one node, the internal distance is 0
        self.demand                 = None       # total demand in this logic node
        self.profit                 = None       # total profit in this logic node
        self.basic                  = True       # default: basic nodes

def _computeDist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

class NodeAggregation:

    def __init__(self):
        self.para = {}
        hyper_para = {}
        hyper_para["threshold_distance"] = 0.2  # aggregation condition on distance d_{xy} \leq this_item * \bar{d}
        hyper_para["threshold_demand"]   = 0.2  # aggregation condition on demand q_{x} + q_{y} \leq this_item * Q
        self.para["hyper_para"] = hyper_para

    def _handleInput(self, input_dict):
        self.para["nodes"]          = list(range(input_dict["dimension"]))
        self.para["location"]       = [None]*input_dict["dimension"]
        self.para["demand"]         = input_dict["demand"]
        self.para["profit"]         = input_dict["profit"]
        self.para["avg_d"]          = sum(self.para["demand"]) / (input_dict["dimension"]-1)
        self.para["capacity"]       = input_dict["capacity"]
        self.para["origin_info"]    = input_dict

        # compute the node location
        for idx in range(input_dict["dimension"]):
            self.para["location"][idx] = (input_dict["distance"][idx][1], input_dict["distance"][idx][2])
        
        # compute the threshold on the distance & the threshold on the demand
        self.para["threshold_distance"] = self.para["hyper_para"]["threshold_distance"] * self.para["avg_d"]
        self.para["threshold_demand"]   = self.para["hyper_para"]["threshold_demand"] * self.para["capacity"]

    def _twoNodesDistance(self, node_1, node_2):
        if node_1.basic and node_2.basic:
            return [_computeDist(self.para["location"][node_1.entry][0], self.para["location"][node_1.entry][1], self.para["location"][node_2.entry][0], self.para["location"][node_2.entry][1]), ("exit", "entry")]
        else:
            connection = [("entry", "entry"), ("entry", "exit"), ("exit", "entry"), ("exit", "exit")]
            distance = [
            _computeDist(self.para["location"][node_1.entry][0], self.para["location"][node_1.entry][1], self.para["location"][node_2.entry][0], self.para["location"][node_2.entry][1]),
            _computeDist(self.para["location"][node_1.entry][0], self.para["location"][node_1.entry][1], self.para["location"][node_2.exit][0], self.para["location"][node_2.exit][1]),
            _computeDist(self.para["location"][node_1.exit][0], self.para["location"][node_1.exit][1], self.para["location"][node_2.entry][0], self.para["location"][node_2.entry][1]),
            _computeDist(self.para["location"][node_1.exit][0], self.para["location"][node_1.exit][1], self.para["location"][node_2.exit][0], self.para["location"][node_2.exit][1]),
            ]
            min_distance = min(distance)
            connection_type = connection[distance.index(min_distance)]
            return [min_distance, connection_type]

    def _addNodes(self, node_1, node_2, connection_type=None):
        def _addTwoNodes(node_1, node_2):
            new_node = Node()
            new_node.ids = node_1.ids + node_2.ids
            new_node.entry = node_1.entry
            new_node.exit = node_2.exit
            new_node.internal_distance = node_1.internal_distance + node_2.internal_distance + _computeDist(self.para["location"][node_1.exit][0], self.para["location"][node_1.exit][1], self.para["location"][node_2.entry][0], self.para["location"][node_2.entry][1])
            new_node.demand = node_1.demand + node_2.demand
            new_node.profit = node_1.profit + node_2.profit
            new_node.basic = False
            return new_node
        if not connection_type:
            connection_type = self._twoNodesDistance(node_1, node_2)[1]
        if connection_type == ("entry", "entry"):
            node_1.ids = node_1.ids[::-1]
            tmp_entry, tmp_exit = node_1.entry, node_1.exit
            node_1.entry, node_1.exit = tmp_exit, tmp_entry
            return _addTwoNodes(node_1, node_2)
        elif connection_type == ("entry", "exit"):
            return _addTwoNodes(node_2, node_1)
        elif connection_type == ("exit", "entry"):
            return _addTwoNodes(node_1, node_2)
        elif connection_type == ("exit", "exit"):
            node_2.ids = node_2.ids[::-1]
            tmp_entry, tmp_exit = node_2.entry, node_2.exit
            node_2.entry, node_2.exit = tmp_exit, tmp_entry
            return _addTwoNodes(node_1, node_2)
        else:
            print("ERROR: ADD TWO NODES WITH MEANINGLESS CONNECTION TYPE")
            return None

    def _satisfyCondition(self, node_1, node_2):
        min_distance = self._twoNodesDistance(node_1, node_2)[0]
        if min_distance <= self.para["threshold_distance"] and (node_1.demand + node_2.demand <= self.para["threshold_demand"]):
            return True
        else: return False


    '''
    output: the resulting distance matrix after node aggregation & a info dictionary 
    the info dictionary contains the aggregate information and will be used to convert the routes on the aggregated nodes 
    into the routes on the original nodes
    '''
    def _nodeAggregation(self):
        print("procesing the node aggregation ---- ")

        self.info = {}

        # record the nodes in the system
        nodes = set()
        for node in self.para["nodes"]:
            if node == 0: pass
            else:
                new_node = Node()
                new_node.ids, new_node.entry, new_node.exit, new_node.internal_distance, new_node.demand, new_node.profit =\
                    [node], node, node, 0, self.para["demand"][node], self.para["profit"][node]
                nodes.add(new_node)

        # iteratively aggregate nodes
        iteration = 0
        while True:
            min_dist, min_dist_connection, min_dist_pair = None, None, None
            for node_1 in nodes:
                for node_2 in nodes:
                    if node_1 == node_2: pass
                    else:
                        if self._satisfyCondition(node_1, node_2):
                            cur = self._twoNodesDistance(node_1, node_2)
                            cur_dist, cur_dist_connection, cur_pair = cur[0], cur[1], (node_1, node_2)
                            if min_dist == None:
                                min_dist, min_dist_connection, min_dist_pair = cur_dist, cur_dist_connection, cur_pair
                            else:
                                if cur_dist < min_dist:
                                    min_dist, min_dist_connection, min_dist_pair = cur_dist, cur_dist_connection, cur_pair
            cur_info = {}
            if min_dist == None: break
            else:
                cur_info["min_dist"] = min_dist
                cur_info["min_dist_connection"] = min_dist_connection
                cur_info["n1_ids"] = min_dist_pair[0].ids
                cur_info["n2_ids"] = min_dist_pair[1].ids
                cur_info["n1_entry"] = min_dist_pair[0].entry
                cur_info["n2_entry"] = min_dist_pair[1].entry
                cur_info["n1_exit"] = min_dist_pair[0].exit
                cur_info["n2_exit"] = min_dist_pair[1].exit
                self.info[iteration] = cur_info
                
                new_node = self._addNodes(min_dist_pair[0], min_dist_pair[1], min_dist_connection)
                nodes.remove(min_dist_pair[0])
                nodes.remove(min_dist_pair[1])
                nodes.add(new_node)
            iteration += 1
        
        # return the resulting nodes
        return nodes

    def _handleOutput(self, aggregated_nodes):
        aggregated_nodes = list(aggregated_nodes)
        result_dict = {}
        result_dict["name"]                 = self.para["origin_info"]["name"]
        result_dict["capacity"]             = self.para["origin_info"]["capacity"] 
        # result_dict["distance"]             = distance
        # result_dict["demand"]               = demand
        # result_dict["profit"]               = profit
        result_dict["trucks"]               = self.para["origin_info"]["trucks"]
        
        ids_list                    = [[]]*self.para["origin_info"]["dimension"]
        internal_distance_list      = [0]*self.para["origin_info"]["dimension"]
        demand_list                 = [0]*self.para["origin_info"]["dimension"]
        profit_list                 = [0]*self.para["origin_info"]["dimension"]
        
        node_idx = 1
        for node in aggregated_nodes:
            ids_list[node_idx] = node.ids
            internal_distance_list[node_idx] = node.internal_distance
            demand_list[node_idx] = node.demand
            profit_list[node_idx] = node.profit
            node_idx += 1

        depot = Node()
        depot.entry = depot.exit = 0
        aggregated_nodes = [depot] + aggregated_nodes
        distance = [[0]*node_idx for _ in range(node_idx)]
        for i in range(node_idx):
            for j in range(node_idx):
                if i == j: pass
                else: distance[i][j] = self._twoNodesDistance(aggregated_nodes[i], aggregated_nodes[j])[0]
        
        result_dict["demand"]           = demand_list[:node_idx]
        result_dict["profit"]           = profit_list[:node_idx]
        result_dict["distance"]         = distance
        result_dict["inner_dist"]       = internal_distance_list[:node_idx]
        result_dict["ids"]              = ids_list[:node_idx]

        result_dict["dimension"]            = len(result_dict["demand"])
        result_dict["aggregation_info"]     = self.info
        result_dict["origin_info"]          = self.para["origin_info"]
        return result_dict

    def haveInput(self, input_dict):
        self._handleInput(input_dict)

    def run(self):
        aggregated_nodes = self._nodeAggregation()
        return self._handleOutput(aggregated_nodes)