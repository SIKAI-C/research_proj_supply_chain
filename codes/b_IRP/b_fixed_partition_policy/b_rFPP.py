from gurobipy import Model, GRB, quicksum
def rFPPSolution(T, N, N_p, E, E_p, K, U, Q, r, I_0, hc, c):
    '''
    T - time horizon
    N_p - set of customer
    N - set of customer + depot
    E_p - set of edges corresponding to customer
    E - set of edges corresponding to customer + depot
    K - set of vehicles
    U - capacity of inventory
    Q - capacity of vehicle
    r - daily consumption rate
    I_0 - initial inventory level
    '''
    
    m = Model("rFPP")
    I = m.addVars(N_p, T, vtype=GRB.CONTINUOUS, lb=0, name="inventory_level")
    z = m.addVars(N, K, T, vtype=GRB.BINARY, name="is_visited_node")
    q = m.addVars(N_p, K, T, vtype=GRB.CONTINUOUS, lb=0, name="quantity_delivered")
    y = m.addVars(E, K, T, vtype=GRB.BINARY, name="is_visited_edge")
    x = m.addVars(E, vtype=GRB.BINARY, name="masked_edge")

    # constraint about inventory level
    m.addConstrs((I[i,1] == I_0[i] + quicksum(q[i,k,1] for k in K) - r[1][i] for i in N_p), name="inventory_level_1")
    m.addConstrs((I[i,t] == I[i,t-1] + quicksum(q[i,k,t] for k in K) - r[t][i] for i in N_p for t in range(2,len(T)+1)), name="inventory_level_2")
    m.addConstrs((I[i,t] >= 0 for i in N_p for t in range(1,len(T)+1)), name="inventory_level_3")
    m.addConstrs((I[i,t] <= U for i in N_p for t in range(1,len(T)+1)), name="inventory_level_4")

    # constraint about quantity delivered
    m.addConstrs((quicksum(q[i,k,t] for i in N_p) <= Q*z[0,k,t] for k in K for t in T), name="quantity_delivered_1")
    m.addConstrs((quicksum(q[i,k,1] for k in K) <= U-I_0[i] for i in N_p), name="quantity_delivered_2")
    m.addConstrs((quicksum(q[i,k,t] for k in K) <= U-I[i, t-1] for i in N_p for t in range(2,len(T)+1)), name="quantity_delivered_3")
    m.addConstrs((q[i,k,t] <= U*z[i,k,t] for i in N_p for k in K for t in T), name="quantity_delivered_4")

    # constraint about routing
    m.addConstrs((quicksum(z[i,k,t] for k in K) <= 1 for i in N_p for t in T), name="routing_1")
    m.addConstrs(
        (quicksum(y[i,j,k,t] for j in N if (i,j) in E) == z[i,k,t] for i in N for k in K for t in T),
        name="routing_2"
    )
    m.addConstrs(
        (quicksum(y[j,i,k,t] for j in N if (j,i) in E) == z[i,k,t] for i in N for k in K for t in T),
        name="routing_3"
    )

    # constraint about masking
    m.addConstrs((x[i,j] >= y[i,j,k,t] for i,j in E for k in K for t in T), name="masking_1")
    m.addConstrs(
        (quicksum(x[i,j] for j in N if (i,j) in E) == 1 for i in N_p),
        name="masking_2"
    )
    m.addConstrs(
        (quicksum(x[j,i] for j in N if (j,i) in E) == 1 for i in N_p),
        name="masking_3"
    )

    # objective function
    m.setObjective(
        quicksum(hc[i]*I[i,t] for i in N_p for t in T) + quicksum(c[i,j]*y[i,j,k,t] for i,j in E for k in K for t in T),
        GRB.MINIMIZE
    )

    m._x = x
    m.Params.LogToConsole = True
    m.Params.lazyConstraints = 1

    def findDelta(S):
        delta = []
        visited = []
        for i in S:
            if i not in visited:
                visited.append(i)
                for j in N:
                    if (i,j) in E and j not in S:
                        delta.append((i,j))
        return delta

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            sol = model.cbGetSolution(model._x)
            edges = {(i,j): int(sol[i,j]) for i,j in model._x.keys() if sol[i,j] > 0.5}
            tours = subtour(edges)
            for tour in tours:
                if tour[0] != 0:
                    Delta = findDelta(tour)
                    model.cbLazy(quicksum(model._x[i,j] for i,j in Delta) >= 2)           

    def subtour(edges):
        tours = []
        all_edges = list(edges.keys())
        while all_edges:
            cycle = []
            start = all_edges[0][0]
            next = all_edges[0][1]
            cycle.append(start)
            cycle.append(next)
            all_edges.remove((start, next))
            while next != start:
                for i in range(len(all_edges)):
                    if all_edges[i][0] == next:
                        next = all_edges[i][1]
                        cycle.append(next)
                        all_edges.remove(all_edges[i])
                        break
            tours.append(cycle)
        return tours

    m.optimize(subtourelim)
    # m.optimize()

    return m, m.getAttr("x", y), m.getAttr("x", x), m.getAttr("x", z), m.getAttr("x", q), m.getAttr("x", I)

