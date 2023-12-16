class SolutionParser:

    def __init__(self):
        pass

    def solutionParser(self, cur_sol, para):

        distance = para["graph"]["distance"]
        duration = para["graph"]["duration"]
        demand = para["graph"]["demand"]
        profit = para["graph"]["profit"]
        num_of_suggestion = para["graph"]["num_of_suggestion"]

        tot_distance = []
        tot_duration = []
        tot_demand = []
        tot_profit = []
        tot_route = []

        for r in cur_sol:
            # if len(r) > 2:
            tot_route.append(r)
            dist, dura, demd, prft = 0, 0, 0, 0
            for node_idx in range(len(r)-1):
                dist += distance[r[node_idx]][r[node_idx+1]]
                dura += duration[r[node_idx]][r[node_idx+1]]
                demd += demand[r[node_idx]]
                prft += profit[r[node_idx]]

            dura += 10*60*(len(r)-2)
            tot_distance.append(dist)
            tot_duration.append(dura)
            tot_demand.append(demd)
            tot_profit.append(prft)

        rank_prft = sorted(range(len(tot_profit)), key=lambda k:tot_profit[k])[::-1]
        rank_prft = rank_prft[:num_of_suggestion]

        # print("dimension - {}".format(para["graph"]["dimension"]))
        # print("capacity - {}".format(para["graph"]["capacity"]))
        return_dura = 0
        return_profit = 0
        for i in rank_prft:
            return_dura += tot_duration[i]
            return_profit += tot_profit[i]
            print("route - {}".format(tot_route[i]))
            print("distance - {:>6.1f} | duration - {:>6.1f} | demand - {:>6.1f} | profit - {:>2.4f}"\
                .format(tot_distance[i], tot_duration[i], tot_demand[i], tot_profit[i]))
        return return_dura, return_profit



