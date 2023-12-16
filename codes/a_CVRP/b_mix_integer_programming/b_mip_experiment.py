from .a_mip_solution import MIPSolution
from a_mip_copt_solution import MIPCoptSolution

import time

class MIPExperiment:

    def __init__(self):
        self.mip_solution = MIPSolution()

    def experiment(self, graph):
        self.result = {}
        idx = 0
        for graph_path in graph:
            idx += 1
            print("####################################################################################################################")
            print("############################################### THE {:>5d} EXPERIMENT ###############################################".format(idx))
            print("THE GRAPH - ", graph_path)
            start = time.time()
            max_profit, sol, sol_attr = self.mip_solution.getSolution(graph_path)
            end = time.time()
            cur_result = {}
            cur_result["max_profit"] = max_profit
            cur_result["sol"] = sol
            cur_result["sol_attr"] = sol_attr
            self.result[idx] = cur_result
            print("\033[1;30;47m{}\033[0m".format("the max profit         - "))
            print("\033[37;41m the sum of profit - {:>5.3f} \033[0m".format(max_profit))
            print("\033[1;30;47m{}\033[0m".format("the total duration         - "))
            print("\033[37;41m the sum of duration - {:>5.3f} \033[0m".format(sol_attr["tot_duration"]))
            print(cur_result)
            print("the TIME                     - {:<15.5f}".format(end-start))
            print("####################################################################################################################")
