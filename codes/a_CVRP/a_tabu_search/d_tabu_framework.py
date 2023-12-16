import time
from .b_fenge_input import *
from .c_evaluate_neighborhood import *
from .c_get_neighbour import *
from .c_initial_solution import *
from .c_report import *
from .d_summary import *
from tqdm import tqdm



class TabuSearch:

    def __init__(self):
        self.para = {}
        self.handle_graph_input = FengEInfo()

    def handleInput(self, graph_idx, input_hyper_para=None):
        self._setHyperPara(input_hyper_para)
        graph_para = self.handle_graph_input.readRouteInstance(graph_idx)
        self.para["graph"] = graph_para
        self.initial = InitialSolution(self.para)
        self.init_sol = self._initialSolution()
        self.evaluator = Evaluator(self.para)
        self.neighborhood = Neighborhood(self.para)

    def _setHyperPara(self, input_hyper_para):
        hyper_para = {}
        hyper_para["c_d"]                      = 0.2          # coefficient of cost of disance
        hyper_para["c_u"]                      = 0.8          # coefficient of cost of unit    
        hyper_para["c_i"]                      = 0.3          # coefficient of cost of infeasible
        hyper_para["r_d"]                      = -0.2         # coefficient of reward of diversity
        hyper_para["r_p"]                      = -0.5         # coefficient of reward of profit
        hyper_para["tabu_tenure"]              = 10           # the tabu tenure
        hyper_para["terminal_max"]             = 1000         # terminal because reach the maximal iteration numbers
        hyper_para["tabu_good"]                = 10           # record a good solution every n iterations
        hyper_para["inference"]                = 10           # record the search result, every n iterations
        hyper_para["logging"]                  = True         # print the log or not
        hyper_para["aggregation"]              = False        # default not to aggregate nodes
        hyper_para["aggregation_logging"]      = False
        
        hyper_para["delta"]                    = 0.5         # adjust the above 4 costs/rewards dynamically
        hyper_para["up_delta"]                 = 0.01
        hyper_para["down_delta"]               = 0.5
        hyper_para["dynamic"]                  = "productive"    # "productive" or "additive"

        hyper_para["initial"]                   = {"type": "default"}       
        # hyper_para["initial"]                   = {"type": "ortools", "ortools_time_limits": 100}      
        # hyper_para["initial"]                   = {"type": "FengE", "ortools_time_limits": 100}       
        # hyper_para["initial"]                   = {"type": "FengE_joint", "ortools_time_limits": 100}      
        if input_hyper_para == None: pass
        else:
            for key in input_hyper_para.keys():
                if key in hyper_para:
                    hyper_para[key] = input_hyper_para[key]
                else:
                    self.para[key] = input_hyper_para[key]
        self.para["hyper_para"] = hyper_para
        return None

    def changeHyperPara(self, input_hyper_para):
        for key in input_hyper_para.keys():
            if key in self.para["hyper_para"]:
                self.para["hyper_para"][key] = input_hyper_para[key]
            else:
                self.para[key] = input_hyper_para[key]
    
    def _initialSolution(self):
        sol = self.initial.getInitialSolution()
        self.para["graph"]["init_distance"]     = sol[0]
        self.para["graph"]["init_duration"]     = sol[1]
        self.para["graph"]["init_demand"]       = sol[2]
        self.para["graph"]["init_profit"]       = sol[3]
        self.para["graph"]["init_max_profit"]   = sol[4]
        self.para["graph"]["init_knapsack_sol"] = sol[5]
        self.para["graph"]["init_sol"]          = sol[6]

        self.para["graph"]["num_of_suggestion"] = len(self.para["graph"]["init_knapsack_sol"])
        self.para["graph"]["trucks"]            = len(self.para["graph"]["init_sol"]) 

        return self.para["graph"]["init_sol"]

    def _evaluateFunction(self, s):
        return self.evaluator.evaluateFunction(s, self.tabu_good)

    def _evaluateFunctionValueOnly(self, s):
        return self.evaluator.evaluateFunction(s, self.tabu_good)[0]

    def _getSolOnly(self, s):
        return s[0]

    def _findNeighborhood(self, s):
        return self.neighborhood.getExchangeNeighbour(s)

    def _anIteration(self, s, iter):
        neighborhood = self._findNeighborhood(s)
        neighborhood_sol_only = map(self._getSolOnly, neighborhood)
        evaluations = list(map(self._evaluateFunctionValueOnly, neighborhood_sol_only))
        evaluations_id = sorted(range(len(evaluations)), key=lambda k:evaluations[k])
        node_insert = False
        for idx in evaluations_id:
            action = neighborhood[idx][1]
            insert_move = (action[0], action[2])
            if insert_move not in self.tabu_list:
                node_insert = True
                remove_move = (action[0], action[1])
                self._holdTabuList(remove_move, iter)
                self.info["cur_f"], self.info["cur_f_info"] = self._evaluateFunction(neighborhood[idx][0])
                return [neighborhood[idx][0], False]
            else: pass
        if node_insert == False:
            return [s, True]

    def _tabuGood(self, s):
        # add the current solution into the set of "good solution"
        # will be used to compute the evaluating funciton
        # reward of diversity
        # and this will print the log if self.para["hyper_para"]["logging"] is True
        for route in range(len(s)):
            for node in s[route]:
                if node != 0:
                    if node not in self.tabu_good[route]: self.tabu_good[route][node] = 1
                    else: self.tabu_good[route][node] += 1 
        self.tabu_good["times"] += 1

    def _holdTabuList(self, remove_move, iteration):
        self.tabu_list[remove_move] = iteration
        release_list = []
        for k in self.tabu_list.keys():
            if iteration > self.para["hyper_para"]["tabu_tenure"] + self.tabu_list[k]: release_list.append(k)
        for k in release_list: self.tabu_list.pop(k)

    def _reset(self):
        self.info = {}
        
        self.info["maximal_profit"] = 0
        self.info["maximal_profit_solution"] = None

        self.info["cur_feasible_profit"] = 0
        self.info["cur_feasible_solution"] = None

        self.tabu_good = {"times": 0}
        for i in range(self.para["graph"]["trucks"]):
            self.tabu_good[i] = {}
        # print(self.tabu_good)
        self.tabu_list = {}
        self.record = {}
        self.tabu_list = {}
        self.record = {}

    def _inference(self, cur_iteration):
        self.record[cur_iteration] = deepcopy(self.info)

    def _report(self, option):
        return currentReport(option, self.info, self.para)

    def _adjustmentAndRecordCur(self):
        if self.info["cur_f_info"]["tot_infeasible"] != 0:
            if self.para["hyper_para"]["dynamic"] == "productive":
                self.para["hyper_para"]["c_i"] *=(self.para["hyper_para"]["delta"]+1)
            else:
                self.para["hyper_para"]["c_i"] += self.para["hyper_para"]["up_delta"]
        else: # if the solution is feasible, make less penalty, and update the current feasible objective value, and the optimal objective value
            if self.para["hyper_para"]["dynamic"] == "productive":
                self.para["hyper_para"]["c_i"] /=(self.para["hyper_para"]["delta"]+1)
            else:
                self.para["hyper_para"]["c_i"] -= self.para["hyper_para"]["down_delta"]
            
            if self.info["cur_f_info"]["tot_duration"] == 0:
                self.info["cur_feasible_solution"] = self.info["cur_sol"]
                self.info["cur_feasible_profit"] = self.info["cur_f_info"]["maximal_profit"]
                if self.info["cur_f_info"]["maximal_profit"] > self.info["maximal_profit"]:
                    self.info["maximal_profit"] = self.info["cur_f_info"]["maximal_profit"]
                    self.info["maximal_profit_solution"] = self.info["cur_sol"]

    def _tabuSearch(self, is_tqdm=True):
        cur_sol = self.init_sol
        cur_iteration = 0
        self.info["cur_sol"], self.info["cur_iteration"] = cur_sol, cur_iteration
        self.info["cur_feasible_profit"] = self.para["graph"]["init_max_profit"]
        self.info["maximal_profit"] = self.para["graph"]["init_max_profit"]
        self._inference(cur_iteration)
        if self.para["hyper_para"]["logging"]: self._report("initial")
        
        if is_tqdm == True:
            for cur_iteration in tqdm(range(1, self.para["hyper_para"]["terminal_max"]+1)):
                # make an iteration
                cur_sol, sudden_terminal = self._anIteration(cur_sol, cur_iteration)
                self.info["cur_sol"], self.info["cur_iteration"], self.info["sudden_terminal"] = cur_sol, cur_iteration, sudden_terminal
                # feasibility, if the solution is infeasible, make more penalty
                self._adjustmentAndRecordCur()
                # keep the tabu_good dict
                if cur_iteration % self.para["hyper_para"]["tabu_good"] == 0: 
                    self._tabuGood(cur_sol)
                # record the search result
                if cur_iteration % self.para["hyper_para"]["inference"] == 0:
                    self._inference(cur_iteration)
                    if self.para["hyper_para"]["logging"]: self._report("iteration")
                if sudden_terminal: 
                    if self.para["hyper_para"]["logging"]: self._report("sudden_terminal")
                    return cur_sol
        else:
            for cur_iteration in range(1, self.para["hyper_para"]["terminal_max"]+1):
                # make an iteration
                cur_sol, sudden_terminal = self._anIteration(cur_sol, cur_iteration)
                self.info["cur_sol"], self.info["cur_iteration"], self.info["sudden_terminal"] = cur_sol, cur_iteration, sudden_terminal
                # feasibility, if the solution is infeasible, make more penalty
                self._adjustmentAndRecordCur()
                # keep the tabu_good dict
                if cur_iteration % self.para["hyper_para"]["tabu_good"] == 0: 
                    self._tabuGood(cur_sol)
                # record the search result
                if cur_iteration % self.para["hyper_para"]["inference"] == 0:
                    self._inference(cur_iteration)
                    if self.para["hyper_para"]["logging"]: self._report("iteration")
                if sudden_terminal: 
                    if self.para["hyper_para"]["logging"]: self._report("sudden_terminal")
                    return cur_sol
        if self.para["hyper_para"]["logging"]: self._report("normal_terminal")
        return cur_sol

    def run(self, is_tqdm=True):
        self._reset()
        self._tabuSearch(is_tqdm)