import json
import random
import string
import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from copy import deepcopy
import time
import sys

sys.path.append("..//1_graph")
from .a_generate_instance import GenerateGraph
from .a_gurobi_solution_parser import solJoint, computeObj, capacityCheck, timeCheck, GurobiSolutionParser

class LagrangianMultipliers:

    def __init__(self):
        self.generate_graph = GenerateGraph()
        self.logging = False
    
    def generateGraph(self):
        para             = self.generate_graph.generateAGraph()
        para["name"]     = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        para["V"]        = list(range(para["dimension"]))              # shelves
        para["E"]        = [(i, j) for i in para["V"]  for j in para["V"] if i != j]     # edges
        para["duration"] = {(i,j): para["duration"][i][j] for i,j in para["E"]}
        para["T_c"]      = 28000                                        # time constraint
        para["T"]        = [1,2,3,4]             # time periods
        para["K"]        = [1,2]
        para["I_lb"]     = 0
        para["I_ub"]     = 280 if np.max(np.array(para["demand"] ))+50 < 280 else np.max(np.array(para["demand"] ))+50
        I_0              = [0 for _ in range(len(para["V"] ))]
        I_0[0]           = 0
        para["I_0"]      = I_0
        para["M"]        = 100000
        return para
    
    def readGraph(self, graph_path):
        with open(graph_path, "r") as f:
            para = json.load(f)
        para["E"]        = [(i, j) for i in para["V"]  for j in para["V"] if i != j]     # edges
        para["duration"] = {(i,j): para["duration"][i][j] for i,j in para["E"]}
        return para

    def _initialParas(self, para):
        V = para["V"]
        K = para["K"]
        T = para["T"]
        capacity = para["capacity"]
        demand = para["demand"]
        duration = para["duration"]
        T_c = para["T_c"]

        LB = 0
        tau = 1
        mu = {(i,t): random.uniform(0,1) for i in V for t in T}
        sigma = 2
        # construct a initial solution
        OBJ = 0
        sol = [[[] for _ in range(len(K))] for _ in range(len(T))]
        for t in range(len(T)):
            cur_idx = 1             # the cur_idx shelf
            cur_location = 0        
            cur_vehicle = 0
            remaining_amount = [capacity for _ in range(len(K))]
            remaining_time = T_c
            while cur_vehicle < len(K):
                # print("current location - ", cur_location)
                # print("current shelf - ", cur_idx)
                # print("current vehicle - ", cur_vehicle)
                # print("remaining amount - ", remaining_amount)
                # print("remaining time - ", remaining_time)
                # print("***"*5)
                # if int(remaining_amount[cur_vehicle]) > 0:
                if remaining_time >= duration[cur_location,cur_idx]+duration[cur_idx,0]+600:
                    sol[t][cur_vehicle].append(cur_idx)
                    remaining_time -= duration[cur_location, cur_idx]
                    remaining_time -= 600
                    if remaining_amount[cur_vehicle] > demand[t][cur_idx]:
                        remaining_amount[cur_vehicle] -= demand[t][cur_idx]
                        cur_location = cur_idx
                        cur_idx += 1
                    else: 
                        OBJ += (demand[t][cur_idx] - remaining_amount[cur_vehicle])
                        remaining_time -= duration[cur_idx,0]
                        remaining_amount[cur_vehicle] = 0
                        cur_idx += 1
                        cur_location = 0
                        cur_vehicle += 1
                else:
                    if cur_location != 0: 
                        remaining_time -= duration[cur_location,0]
                        cur_location = 0
                    break   
            while cur_idx < para["dimension"]: 
                OBJ += demand[t][cur_idx]   
                cur_idx += 1
        UB = OBJ
        X = [[[0]+sol[t][k]+[0] for k in range(len(K))] for t in range(len(T))]
        para["LB"] = LB
        para["UB"] = UB
        para["tau"] = tau
        para["mu"] = mu
        para["sigma"] = sigma
        para["X"] = X
        return para
    
    # The IA-P 
    def _IAPModel(self, para):
        mu = para["mu"]
        V = para["V"]
        T = para["T"]
        I_0 = para["I_0"]
        demand = para["demand"]
        I_lb = para["I_lb"]
        I_ub = para["I_ub"]
        M = para["M"]

        start = time.time()
        m = Model("IA-P")
        z = m.addVars(V, T, vtype=GRB.CONTINUOUS, name="replenish_amount")
        I = m.addVars(V, T, vtype=GRB.CONTINUOUS, name="inventory_level")
        B = m.addVars(V, T, vtype=GRB.CONTINUOUS, name="unsatisfied_demand")
        r = m.addVars(V, T, vtype=GRB.BINARY, name="whether_unsatisfied_demand")
        # constraints about inventory level
        m.addConstrs((I[i,1] - B[i,1] == I_0[i] + z[i,1] - demand[0][i] for i in V if i != 0), name="inventory_level_for_two_adjacent_days_0")
        m.addConstrs((I[i,t] - B[i,t] == I[i,t-1] + z[i,t] - demand[t-1][i] for i in V if i != 0 for t in T if t != 1), name="inventory_level_for_two_adjacent_days_1")
        m.addConstrs((I[i,t] >= I_lb for i in V if i != 0 for t in T), name="inventory_lower_bound")
        m.addConstrs((I_0[i]+z[i,1] <= I_ub for i in V if i != 0), name="inventory_upper_bound_0")
        m.addConstrs((I[i,t-1] + z[i,t] <= I_ub for i in V if i != 0 for t in T if t != 1), name="inventory_upper_bound_1")
        m.addConstrs((z[0,t] == 0 for t in T), name="replenishment_amount_depot")
        m.addConstrs((r[0,t] == 0 for t in T), name="whether_unsatisfied_demand_depot")
        m.addConstrs((B[0,t] == 0 for t in T), name="unsatisfied_demand_depot")
        m.addConstrs((I[0,t] == 0 for t in T), name="inventory_level_depot")
        m.addConstrs((B[i,t] <= M*r[i,t] for i in V if i != 0 for t in T), name="unsatisfied_demand_is_positive")
        m.addConstrs((I[i,t] <= M*(1-r[i,t]) for i in V if i != 0 for t in T), name="inventory_level_is_positive")
        m.setObjective(quicksum(B[i,t]+z[i,t]*mu[i,t] for i in V if i != 0 for t in T), GRB.MINIMIZE)
        end = time.time()
        m.params.LogToConsole=self.logging # 显示求解过程
        m.Params.MIPGap=0.01 # 百分比界差
        m.Params.TIME_LIMIT=50 # 限制求解时间为 50s
        m.optimize()
        optimal_status = m.status
        solver_res = {}
        solver_res["T"] = T
        solver_res["val"] = m.objVal
        solver_res["z"] = m.getAttr('x', z)
        solver_res["I"] = m.getAttr('x', I)
        solver_res["B"] = m.getAttr('x', B)
        solver_res["r"] = m.getAttr('x', r)
        solver_res["I0"] = I_0
        solver_res["time"] = end-start
        solver_res["optimal_status"] = optimal_status
        para["IA-P"] = solver_res
        return para
    
    # The RT-P
    def _RTPModel(self, para):
        mu = para["mu"]
        V = para["V"]
        E = para["E"]
        capacity = para["capacity"]
        duration = para["duration"]
        T_c = para["T_c"]
        K = para["K"]
        T = para["T"]

        start = time.time()
        m = Model("RT-P")
        # add variables
        x = m.addVars(duration, K, T, vtype=GRB.BINARY, name="edge_var")
        y = m.addVars(V, K, T, vtype=GRB.BINARY, name="vertex_var")
        Q = m.addVars(duration, K, T, vtype=GRB.CONTINUOUS, name="continuous_factor")
        # capacity
        m.addConstrs((Q[i,j,k,t] <= capacity*x[i,j,k,t] for i,j in E for k in K for t in T), name="capacity")
        # time limits
        m.addConstrs((quicksum((600*(y.sum('*',k,t)-1) + quicksum(duration[i,j]*x[i,j,k,t] for i,j in duration.keys())) for k in K) <= T_c for t in T), name="time_limit")
        # routing constraints
        m.addConstrs((y[0,k,t] <= 1 for k in K for t in T), name="depot")
        m.addConstrs((y.sum(i,'*',t) <= 1 for i in V if i != 0 for t in T), name="shelf")
        m.addConstrs((x.sum(i,'*',k,t) == y[i,k,t] for i in V for k in K for t in T), name="leaving_vertex")
        m.addConstrs((x.sum('*',j,k,t) == y[j,k,t] for j in V for k in K for t in T), name="arriving_vertex")
        m.setObjective(quicksum(mu[j,t]*(Q.sum(j,'*','*',t)-Q.sum('*',j,'*',t)) for j in V if j != 0 for t in T), GRB.MINIMIZE)
        end = time.time()
        m.params.LogToConsole=self.logging # 显示求解过程
        m.Params.MIPGap=0.01 # 百分比界差
        m.Params.TIME_LIMIT=50 # 限制求解时间为 50s
        m.optimize()
        optimal_status = m.status
        solver_res = {}
        solver_res["T"] = T
        solver_res["val"] = m.objVal
        solver_res["x"] = m.getAttr('x', x)
        solver_res["y"] = m.getAttr('x', y)
        solver_res["Q"] = m.getAttr('x', Q)
        solver_res["time"] = end-start
        solver_res["optimal_status"] = optimal_status
        para["RT-P"] = solver_res
        return para
    
    # construct a feasible solution
    def _feasibleSolution(self, para):

        def MultiDayRouting(para):
            capacity = para["capacity"]
            demand = para["IA-P"]["z"]
            T = para["T"]
            K = para["K"]
            V = para["V"]
            E = para["E"]
            T_c = para["T_c"]
            duration = para["duration"]
            start = time.time()
            m = Model("Multi_Periods_Routing")
            x = m.addVars(duration, K, T, vtype=GRB.BINARY, name="edge_var")
            y = m.addVars(V, K, T, vtype=GRB.BINARY, name="vertex_var")
            u = m.addVars(V, T, vtype=GRB.CONTINUOUS, name="continuous_factor")
            m.addConstrs((y[0,k,t] == 1 for k in K for t in T), name="depot")
            m.addConstrs((y.sum(i,'*',t) <= 1 for i in V if i != 0 for t in T), name="shelf")
            m.addConstrs((x.sum(i,'*',k,t) == y[i,k,t] for i in V for k in K for t in T), name="leaving_vertex")
            m.addConstrs((x.sum('*',j,k,t) == y[j,k,t] for j in V for k in K for t in T), name="arriving_vertex")
            m.addConstrs(((x[i,j,k,t] == 1) >> (u[i,t]+demand[j,t]<=u[j,t]) for i,j in E if j!= 0 for k in K for t in T), name="continuous_conditions")
            m.addConstrs((u[i,t] >= demand[i,t] for i in V for t in T), name="lb_continuous_conditions")
            m.addConstrs((u[i,t] <= capacity for i in V for t in T), name="ub_continuous_conditions")
            m.addConstrs((quicksum((600*(y.sum('*',k,t)-1) + quicksum(duration[i,j]*x[i,j,k,t] for i,j in duration.keys())) for k in K) <= T_c for t in T), name="time_limit")
            m.addConstrs((quicksum(y[i,k,t]*demand[i,t] for i in V) <= capacity for k in K for t in T), name="capacity")
            m.setObjective(quicksum(demand[i,t]*y[i,k,t] for i in V if i != 0 for k in K for t in T), GRB.MAXIMIZE)
            m.params.LogToConsole=False # 显示求解过程
            m.Params.MIPGap=0.01 # 百分比界差
            m.Params.TIME_LIMIT=30 # 限制求解时间为 50s
            m.optimize()
            end = time.time()
            solver_res = {}
            solver_res["val"] = m.objVal
            solver_res["x"] = m.getAttr('x', x)
            solver_res["y"] = m.getAttr('x', y)
            solver_res["u"] = m.getAttr('x', u)
            solver_res["time"] = end-start
            return solver_res

        
        T = para["T"]
        V = para["V"]
        K = para["K"]
        solver_res = MultiDayRouting(para)
        x = solver_res['x']
        res = [[[] for _ in range(len(K))] for _ in range(len(T))]
        for i,j,k,t in x.keys():
            if round(x[i,j,k,t]) == 1:
                res[t-1][k-1].append((i,j))
        res = [[solJoint(vehicle_res) for vehicle_res in daily_res] for daily_res in res]
        I_0 = deepcopy(para["I_0"])
        amount = para["IA-P"]["z"]
        demand = {(i,t): para["demand"][t-1][i] for t in para["T"] for i in para["V"]}
        # OBJ = computeObj(res, amount, demand, I_0, T, V)
        OBJ = computeObj(res, amount, demand, I_0, T, V)
        solver_res["replenishment"] = amount

        return res, OBJ, solver_res
    
