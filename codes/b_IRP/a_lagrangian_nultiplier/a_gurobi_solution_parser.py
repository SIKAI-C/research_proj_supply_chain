# from edge variables to route decision
def solJoint(s):
    res = [0]
    while True:
        for i,j in s:
            if i == res[-1]:
                res.append(j)
                break
        if res[-1] == 0: break
    return res

def computeObj(sol, amount, deamnd, I_0, T, V):
    OBJ = 0
    I = I_0
    for t in T:
        daily_sol = sol[t-1]
        for i in V:
            if any([i in vehicle_daily_sol for vehicle_daily_sol in daily_sol]): I_new = I[i]+amount[i,t]-deamnd[i,t]
            else: I_new = I[i]-deamnd[i,t]
            if I_new > 0: I[i] = I_new
            else:
                I[i] = 0
                OBJ += (-I_new)
    return OBJ

def capacityCheck(sol, replenishment, T, K):
    if type(T) == list: T = len(T)
    if type(K) == list: K = len(K)
    sol_capacity = [[0 for _ in range(K)] for _ in range(T)]
    for t in range(T):
        for k in range(K):
            amount = 0
            for i in sol[t][k]: amount += replenishment[i,t+1]
            sol_capacity[t][k] = amount
    return sol_capacity

def timeCheck(sol, duration, T, K):
    if type(T) == list: T = len(T)
    if type(K) == list: K = len(K)
    sol_time = [0 for _ in range(T)]
    for t in range(T):
        t_duration = 0
        for k in range(K):
            sub_duration = 600*(len(sol[t][k])-2)
            for idx in range(len(sol[t][k])-1):
                sub_duration += duration[sol[t][k][idx],sol[t][k][idx+1]]
            t_duration += sub_duration
        sol_time[t] = t_duration
    return sol_time

class GurobiSolutionParser:

    def __init__(self):
        self.details = True
    
    def gurobiSolutionParser(self, graph_para, solver_para):
        return self._gurobiSolutionParser(graph_para, solver_para)

    def _gurobiSolutionParser(self, graph_para, solver_para):

        print("----------"*8)
        print()

        if solver_para == {}: 
            print("No feasible solution found.")
            print()
            return {}

        T   = len(solver_para["T"])
        K   = len(solver_para["K"])
        val = solver_para["val"]
        x   = solver_para["x"]
        y   = solver_para["y"]
        z   = solver_para["z"]
        I   = solver_para["I"]
        B   = solver_para["B"]
        Q   = solver_para["Q"]
        I_0 = solver_para["I0"]

        if self.details:
            print("########################################")
            print("### OBTAIN ROUTES FORM MIP SOLUTIONS ###")
            print("########################################")

        # parse x results
        def solJoint(s):
                res = [0]
                while True:
                    for i,j in s:
                        if i == res[-1]:
                            res.append(j)
                            break
                    if res[-1] == 0: break
                return res
        
        sol = [[[] for _ in range(K)] for _ in range(T)]
        for i,j,k,t in x.keys():
            if round(x[i,j,k,t]) == 1:
                sol[t-1][k-1].append((i,j))
        sol = [[solJoint(vehicle_sol) for vehicle_sol in daily_sol] for daily_sol in sol]

        text_colors = [31+i for i in range(T)]
        if self.details:
            print("**"*5)
            print("SOLUTIONS FROM x (edge_vars) VARIABLES")
            print("**"*5)
            print("shelves - ")
            for i in range(1,graph_para["dimension"]): print("{:>3d}".format(i), end=" ")
            print()
            for t in range(T): print("\033[{}m{:>2d} day\033[0m".format(text_colors[t], t+1), end = "  ")
            print()
            print("covered shelves - ")
        covered_shelves = []
        uncovered_shelves = []
        for i in range(1,graph_para["dimension"]):
            idc = -1
            for t in range(T):
                for k in range(K):
                    if i in sol[t][k]: idc = t
            if idc == -1: 
                uncovered_shelves.append(i)
                if self.details: print("{:>3d}".format(i), end=" ")
            else: 
                covered_shelves.append(i)
                if self.details: print("\033[{}m{:>3d}\033[0m".format(text_colors[idc], i), end = " ")
        if self.details:
            print()
            print("details - ")
            for t in range(T):
                for k in range(K):
                    print("\033[{}m{:>2d} day {:>2d} vehicle- {}\033[0m".format(text_colors[t], t+1, k+1, sol[t][k]))

        # route feasibility
        if self.details:
            print("**"*5)
            print("CHECK THE FEASIBILITY OF SOLUTIONS")
            print("**"*5)
        feasibility = True 
        if self.details: print("capacity - {} | time constraint - {}".format(graph_para["capacity"], 28000))
        sol_capacity = [[0 for _ in range(K)] for _ in range(T)]
        sol_time = [0 for _ in range(T)]

        # time
        for t in range(T):
            duration = 0
            for k in range(K):
                sub_duration = 600*(len(sol[t][k])-2)
                for idx in range(len(sol[t][k])-1):
                    sub_duration += graph_para["duration"][sol[t][k][idx]][sol[t][k][idx+1]]
                duration += sub_duration
            sol_time[t] = duration
        
        # capacity
        demand = graph_para["demand"]
        profit = graph_para["profit"]
        for t in range(T):
            for k in range(K):
                amount = 0
                for i in range(len(demand[0])):
                    amount += z[i,k+1,t+1]
                sol_capacity[t][k] = amount
        
        if self.details:
            for t in range(T):
                print("\033[{}m{:>2d} day \033[0m".format(text_colors[t], t+1))
                print("\033[{}m     duration - {} \033[0m".format(text_colors[t], sol_time[t]))
                if sol_time[t] > 28000: feasibility = False
                for k in range(K):
                    print("\033[{}m         vehicle - {:>1d} amount - {:>5.5f} \033[0m".format(text_colors[t], k+1, sol_capacity[t][k]))
                    if sol_capacity[t][k] > graph_para["capacity"]: feasibility = False
        
        # inventory level
        sol_inventory = [[0 for _ in range(graph_para["dimension"])] for _ in range(T)]
        sol_unsatisfied_demand = [[0 for _ in range(graph_para["dimension"])] for _ in range(T)]
        for t in range(T):
            for i in range(graph_para["dimension"]):
                sol_inventory[t][i] = max(0, I[i,t+1])
                sol_unsatisfied_demand[t][i] = max(0, B[i,t+1])
        replenishment = [[0 for _ in range(graph_para["dimension"])] for _ in range(T)]
        for t in range(T):
            for i in range(graph_para["dimension"]):
                a_replenishment = 0
                for k in range(K):
                    a_replenishment += z[i,k+1,t+1]
                replenishment[t][i] = a_replenishment

        if self.details:
            print("**"*5)
            print("CHECK THE INVENTORY LEVELS")
            print("**"*5)
            print("I_lb - 0.0 | I_ub - 1.1")
            
            shelves_output       = "SHELVES       -   |"
            inventory_output     = ["INVENTORY     - {} |".format(t) for t in range(T+1)]
            replenishment_output = ["REPLENISHMENT - {} |".format(t) for t in range(T)]
            demand_output        = ["DEMAND        - {} |".format(t+1) for t in range(T)]
            # profit_output    = ["PROFIT    - {} |".format(t+1) for t in range(T)]
            for i in range(1,graph_para["dimension"]): 
                if i != 0 and i % 25 == 0: 
                    print(shelves_output)
                    print("\033[{};46m{}\033[0m".format(30, inventory_output[0]))
                    for t in range(T): 
                        print("\033[{}m{}\033[0m".format(text_colors[t], replenishment_output[t]))
                        print("\033[{}m{}\033[0m".format(text_colors[t], demand_output[t]))
                        print("\033[{};46m{}\033[0m".format(text_colors[t], inventory_output[t+1]))
                        # print("\033[{}m{}\033[0m".format(text_colors[t], profit_output[t]))
                    print()
                    shelves_output       = "SHELVES       -   |"
                    inventory_output     = ["INVENTORY     - {} |".format(t) for t in range(T+1)]
                    replenishment_output = ["REPLENISHMENT - {} |".format(t) for t in range(T)]
                    demand_output        = ["DEMAND        - {} |".format(t+1) for t in range(T)]
                    # profit_output    = ["PROFIT    - {} |".format(t+1) for t in range(T)]
                
                shelves_output += " {:>4d} |".format(i)
                for t in range(T):
                    demand[t][i] = max(0, demand[t][i])
                    profit[t][i] = max(0, profit[t][i])
                    demand_output[t] += " {:.2f} |".format(demand[t][i])
                    replenishment_output[t] += " {:.2f} |".format(replenishment[t][i])
                    # profit_output[t] += " {:.2f} |".format(profit[t][i]*demand[t][i])
                for t in range(T+1): 
                    if t == 0: inventory_output[t] += " {:>.2f} |".format(I_0[i])
                    else: inventory_output[t] += " {:>.2f} |".format(sol_inventory[t-1][i])

        # parse y results
        sol_y = [[[] for _ in range(K)] for _ in range(T)]
        for i,k,t in y.keys():
            if round(y[i,k,t]) == 1: sol_y[t-1][k-1].append(i)
        if self.details:
            print("**"*5)
            print("SOLUTIONS FROM y (vertex_vars) VARIABLES, COMPARE WITH x")
            print("**"*5)
            for t in range(T):
                for k in range(K):
                    print("\033[{}m{:>2d} day {:>2d} vehicle- \033[0m".format(text_colors[t], t+1, k+1))
                    print("     y - {}".format(sorted(sol_y[t][k])))
                    print("     x - {}".format(sorted(sol[t][k][1:])))
        
        # check the objective
        if self.details: 
            print("**"*5)
            print("CHECK THE UNSATISFIED DEMAND")
            print("**"*5)
            unsatisfied_demand = []
            loss_profit = []
            total_demand = []
            for t in range(T):
                total_d, unsatisfied_d, loss_p = 0, 0, 0
                for i in range(graph_para["dimension"]):
                    total_d += demand[t][i]
                    if sol_unsatisfied_demand[t][i] > 0:
                        unsatisfied_d += sol_unsatisfied_demand[t][i]
                        loss_p += sol_unsatisfied_demand[t][i] * profit[t][i]
                total_demand.append(total_d)
                unsatisfied_demand.append(unsatisfied_d)
                loss_profit.append(loss_p)
                print("\033[{}m{:>2d} day - unsatisfied demand / total demand - {:>4.4f}/{:>4.4f} = {:>4.4f} | loss profit - {:>4.4f} \033[0m".format(text_colors[t], t+1, unsatisfied_d, total_d, unsatisfied_d/total_d, loss_p))                
        
        # a summary
        if self.details: 
            print("**"*5)
            print("A SUMMARY")
            print("**"*5)
        print("instance - {} | day - {} | time - {} | unsatisfied demand / total demand - {}/{} = {}".format(graph_para["name"], T, solver_para["time"], sum(unsatisfied_demand), sum(total_demand), sum(unsatisfied_demand)/sum(total_demand)))
        print()
        res = {}
        # res["instance"] = graph_para["name"]
        res["routes"] = sol
        res["shelves"] = sol_y
        res["sol_capacity"] = sol_capacity
        res["sol_duration"] = sol_time
        res["sol_feasibility"] = feasibility
        res["unsatisfied_demand"] = unsatisfied_demand
        res["loss_profit"] = loss_profit
        res["total_demand"] = total_demand
        return res

