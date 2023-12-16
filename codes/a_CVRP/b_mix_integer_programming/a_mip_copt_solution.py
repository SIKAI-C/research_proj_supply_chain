import coptpy as COPT
from coptpy import Envr, quicksum
from coptpy import *

from ..a_tabu_search.b_fenge_input import FengEInfo

class MIPCoptSolution:

    def __init__(self):
        self.read_instance = FengEInfo()

    def _getSolution(self, idx):
        para = self.read_instance.readRouteInstance(idx)
        V        = list(range(para["dimension"]))
        E        = [(i, j) for i in V for j in V if i != j]
        capacity = para["capacity"]
        profit   = para["profit"]
        demand   = para["demand"]
        duration = para["duration"]
        duration = {(i,j): duration[i][j] for i,j in E}
        T        = 28000
        # K_list = [[i+1 for i in range(k)] for k in range(1,self.K+1)]
        # K_list = [[i+1 for i in range(self.K)]]
        if capacity >= 1500: K_list = [[1]]
        else: K_list = [[1], [1,2]]

        def MIPModel(V, E, capacity, profit, demand, duration, T, K):
            env = Envr()
            m = env.createModel("MIP_VRP")
            x = m.addVars(duration, K, vtype=COPT.BINARY, nameprefix="edge_var")
            y = m.addVars(V, K, vtype=COPT.BINARY, nameprefix="vertex_var")
            u = m.addVars(V, vtype=COPT.CONTINUOUS, nameprefix="continuous_factor")
            m.addConstrs((y[0,k] == 1 for k in K), nameprefix="depot")
            m.addConstrs((y.sum(i,'*') <= 1 for i in V if i != 0), nameprefix="shelf")
            m.addConstrs((x.sum(i,'*',k) == y[i,k] for i in V for k in K), nameprefix="leaving_vertex")
            m.addConstrs((x.sum('*',j,k) == y[j,k] for j in V for k in K), nameprefix="arriving_vertex")
            # m.addConstrs(((x[i,j,k] == 1) >> (u[i]+demand[j]==u[j]) for i,j in E if i != 0 and j!= 0 for k in K), nameprefix="continuous_conditions")
            for i,j in E:
                if i !=0 and j!= 0:
                    for k in K:
                        m.addGenConstrIndicator(x[i,j,k], True, u[i]+demand[j]==u[j])
            m.addConstrs((u[i] >= demand[i] for i in V), nameprefix="lb_continuous_conditions")
            m.addConstrs((u[i] <= capacity for i in V), nameprefix="ub_continuous_conditions")
            m.addConstr((quicksum((600*(y.sum('*',k)-1) + quicksum(duration[i,j]*x[i,j,k] for i,j in duration.keys())) for k in K) <= T), name="time_limit")
            m.addConstrs((quicksum(y[i,k]*demand[i] for i in V) <= capacity for k in K), nameprefix="capacity")
            m.setObjective(quicksum(profit[i]*y[i,k] for i in V for k in K), sense=COPT.MAXIMIZE)

            m.setParam(COPT.Param.Logging, 0)
            m.setParam(COPT.Param.RelGap, 0.01)
            m.setParam(COPT.Param.TimeLimit, 30)
            m.solve()
            return m.objval, m.getInfo(COPT.Info.Value, x), m.getInfo(COPT.Info.Value, y)

        max_profit = 0
        max_profit_x = None
        max_profit_y = None
        max_profit_K = None

        for K in K_list:
            try:
                cur_profit, x, y = MIPModel(V, E, capacity, profit, demand, duration, T, K)
                if cur_profit > max_profit:
                    max_profit = cur_profit
                    max_profit_x = x
                    max_profit_y = y
                    max_profit_K = K
            except:
                pass
        
        self.capacity = capacity
        self.profit   = profit
        self.demand   = demand
        self.duration = duration

        return max_profit, max_profit_x, max_profit_y, max_profit_K
    
    def _solutionParser(self, x, y, K):
        
        def solJoint(s):
            res = [0]
            while True:
                for i,j in s:
                    if i == res[-1]:
                        res.append(j)
                        break
                if res[-1] == 0: break
            return res
        
        sol = [[] for i in range(len(K))]
        for i,j,k in x.keys():
            if round(x[i,j,k]) == 1:
                sol[k-1].append((i,j))
        sol = [solJoint(s) for s in sol]

        sol_attr = {}
        sol_attr["capacity"]     = self.capacity
        sol_attr["duration"]     = []
        sol_attr["demand"]       = []
        sol_attr["profit"]       = []
        sol_attr["tot_duration"] = 0
        sol_attr["tot_profit"]   = 0

        for i in range(len(sol)):
            s = sol[i]
            dura, prft, demd = 0, 0, 0
            for node_idx in range(len(s)-1):
                dura += self.duration[s[node_idx], s[node_idx+1]]
                prft += self.profit[s[node_idx]]
                demd += self.demand[s[node_idx]]
            dura += 600*(len(s)-2)
            sol_attr["duration"].append(dura)
            sol_attr["profit"].append(prft)
            sol_attr["demand"].append(demd)
            sol_attr["tot_duration"] += dura
            sol_attr["tot_profit"]   += prft

        return sol, sol_attr

    def getSolution(self, idx):
        max_profit, max_profit_x, max_profit_y, max_profit_K = self._getSolution(idx)
        sol, sol_attr = self._solutionParser(max_profit_x, max_profit_y, max_profit_K)
        return max_profit, sol, sol_attr

