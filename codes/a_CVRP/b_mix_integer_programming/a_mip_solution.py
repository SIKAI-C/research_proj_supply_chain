from gurobipy import Model, GRB, quicksum

from ..a_tabu_search.b_fenge_input import FengEInfo

class MIPSolution:

    def __init__(self):
        self.read_instance = FengEInfo()
        self.K = 1

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
            # env = gp.Env(empty=True)
            # env.setParam("OutputFlag",0)
            # env.start()
            # m = Model("MIP_VRP", env=env)
            m = Model("MIP_VRP")
            x = m.addVars(duration, K, vtype=GRB.BINARY, name="edge_var")
            y = m.addVars(V, K, vtype=GRB.BINARY, name="vertex_var")
            u = m.addVars(V, vtype=GRB.CONTINUOUS, name="continuous_factor")
            m.addConstrs((y[0,k] == 1 for k in K), name="depot")
            m.addConstrs((y.sum(i,'*') <= 1 for i in V if i != 0), name="shelf")
            m.addConstrs((x.sum(i,'*',k) == y[i,k] for i in V for k in K), name="leaving_vertex")
            m.addConstrs((x.sum('*',j,k) == y[j,k] for j in V for k in K), name="arriving_vertex")
            m.addConstrs(((x[i,j,k] == 1) >> (u[i]+demand[j]==u[j]) for i,j in E if i != 0 and j!= 0 for k in K), name="continuous_conditions")
            m.addConstrs((u[i] >= demand[i] for i in V), name="lb_continuous_conditions")
            m.addConstrs((u[i] <= capacity for i in V), name="ub_continuous_conditions")
            m.addConstr((quicksum((600*(y.sum('*',k)-1) + quicksum(duration[i,j]*x[i,j,k] for i,j in duration.keys())) for k in K) <= T), name="time_limit")
            m.addConstrs((quicksum(y[i,k]*demand[i] for i in V) <= capacity for k in K), name="capacity")
            m.setObjective(quicksum(profit[i]*y[i,k] for i in V for k in K), GRB.MAXIMIZE)

            m.params.LogToConsole=False # 显示求解过程
            m.Params.MIPGap=0.01 # 百分比界差
            m.Params.TIME_LIMIT=30 # 限制求解时间为 50s
            m.optimize()

            return m.objVal, m.getAttr('x', x), m.getAttr('x', y)

        max_profit = 0
        max_profit_x = None
        max_profit_y = None
        max_profit_K = None

        for K in K_list:
            cur_profit, x, y = MIPModel(V, E, capacity, profit, demand, duration, T, K)
            if cur_profit > max_profit:
                max_profit = cur_profit
                max_profit_x = x
                max_profit_y = y
                max_profit_K = K

        
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

if __name__ == '__main__':
    a = MIPSolution()
    print(a.getSolution("187"))