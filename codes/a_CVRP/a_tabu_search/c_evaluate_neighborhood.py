class Evaluator:

    def __init__(self, para):
        self.para = para
        self.aggregation = para["hyper_para"]["aggregation"]
        self.inter_distance = para["graph"]["distance"]
        self.inter_duration = para["graph"]["duration"]
        if self.aggregation:
            self.inner_distance = para["graph"]["inner_dist"]
        self.demand = para["graph"]["demand"]
        self.profit = para["graph"]["profit"]
        self.capacity = para["graph"]["capacity"]
        self.num_of_suggestion = para["graph"]["num_of_suggestion"]

    def evaluateFunction(self, s, good_solution):
        cur_f   = 0
        info    = {}

        tot_dist = 0            # cost of distance
        tot_duration = 0        # cost of duration
        tot_violation = 0       # cost of infeasible
        tot_good_count = 0      # reward of diversity
        max_profit = 0          # reward of profit

        r_dist       = []
        r_duration   = []
        r_violation  = []
        r_good_count = []
        r_profit     = []

        for route_idx in range(len(s)):
            route = s[route_idx]
            rr_dist, rr_duration,  rr_demand, rr_profit, rr_good_count = 0, 0, 0, 0, 0
            for node_idx in range(len(route)-1):
                node = route[node_idx]
                # distance
                rr_dist += self.inter_distance[route[node_idx]][route[node_idx+1]]
                # duration
                rr_duration += self.inter_duration[route[node_idx]][route[node_idx+1]]
                # demand 
                rr_demand += self.demand[route[node_idx]]
                # profit 
                rr_profit += self.profit[route[node_idx]]
                # diversity
                if node in good_solution[route_idx]: rr_good_count += good_solution[route_idx][node]
            rr_duration += 600*(len(route)-2)
            r_dist.append(rr_dist)
            r_duration.append(rr_duration)
            r_violation.append(max(rr_demand-self.capacity, 0))
            r_profit.append(rr_profit)
            r_good_count.append(rr_good_count)

        rank_profit = sorted(range(len(r_profit)), key=lambda k:r_profit[k])[::-1]
        rank_profit = rank_profit[:self.num_of_suggestion]

        tot_dist = sum(r_dist)
        tot_duration = sum(r_duration[i] for i in rank_profit)
        # for i in rank_profit:
        #     tot_duration += r_duration[i]
        #     print(r_duration[i], "---", tot_duration)
        # print(tot_duration)
        tot_duration = max(tot_duration-28000, 0)
        # print(tot_duration)
        # print("------------------")
        tot_violation = sum(r_violation)
        tot_good_count = sum(r_good_count)
        max_profit = sum(r_profit[i] for i in rank_profit)

        cur_f += self.para["hyper_para"]["c_d"] * tot_dist
        info["cost_of_distance"] = self.para["hyper_para"]["c_d"] * tot_dist
        info["tot_distance"] = tot_dist

        cur_f += self.para["hyper_para"]["c_u"] * tot_duration
        info["cost_of_duration"] = self.para["hyper_para"]["c_u"] * tot_duration
        info["tot_duration"] = tot_duration

        cur_f += self.para["hyper_para"]["c_i"] * tot_violation
        info["cost_of_infeasible"] = self.para["hyper_para"]["c_i"] * tot_violation
        info["tot_infeasible"] = tot_violation
        
        cur_f += self.para["hyper_para"]["r_d"] * tot_good_count
        info["reward_of_diversity"] = self.para["hyper_para"]["r_d"] * tot_good_count
        info["tot_repeated_attributes"] = tot_good_count

        cur_f += self.para["hyper_para"]["r_p"] * max_profit
        info["reward_of_profit"] = self.para["hyper_para"]["r_p"] * max_profit
        info["maximal_profit"] = max_profit

        return [cur_f, info]