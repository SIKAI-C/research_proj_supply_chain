import time
from .c_parser_solution import *
from .d_tabu_framework import *

class Experiment:

    def __init__(self):
        self.search_algorithm = TabuSearch()
        self.search_result_summary = SearchResultSummary()
        self.solution_parser = SolutionParser()

    def experiment(self, graph, default_settings, focus, plot=False):
        settings = []
        if focus == None:
            settings.append(default_settings)
            self._experiment(graph, settings,plot=plot)
        else:
            focus_para = focus[0]
            for focus_setting in focus[1:]:
                cur_setting = deepcopy(default_settings)
                for i in range(len(focus_para)):
                    # print(focus_para, len(focus_para))
                    # print(focus_setting, i)
                    cur_setting[focus_para[i]] = focus_setting[i]
                settings.append(cur_setting)
            self._experiment(graph, settings, focus_para,plot=plot)

    def _experiment(self, graph=[], settings=[], focus=None, plot=False):
        self.result = {}
        idx = 0
        for graph_path in graph:
            for para in settings:
                idx += 1
                cur_result = {}
                cur_result["graph"] = graph_path
                cur_result["para"] = para
                self.search_algorithm.handleInput(graph_path, para)
                print("####################################################################################################################")
                print("############################################### THE {:>5d} EXPERIMENT ###############################################".format(idx))
                print("THE GRAPH - ", graph_path)
                start = time.time()
                if idx == 1:
                    cur_info, cur_record = self._anExperiment(para, None)
                else: 
                    cur_info, cur_record = self._anExperiment(para, focus)
                end = time.time()
                cur_result["initial_solution"] = self.search_algorithm.init_sol
                cur_result["info"] = cur_info
                cur_result["record"] = cur_record
                self.result[idx] = cur_result
                print("\033[1;30;47m{}\033[0m".format("the initial solution         - "))
                # self.solution_parser.solutionParser(cur_result["initial_solution"], self.search_algorithm.para)
                init_dura = 0
                init_profit = 0
                for i in range(self.search_algorithm.para["graph"]["num_of_suggestion"]):
                    init_dura += self.search_algorithm.para["graph"]["init_duration"][self.search_algorithm.para["graph"]["init_knapsack_sol"][i]]
                    init_profit += self.search_algorithm.para["graph"]["init_profit"][self.search_algorithm.para["graph"]["init_knapsack_sol"][i]]

                    print("route - {}".format(self.search_algorithm.para["graph"]["init_sol"][self.search_algorithm.para["graph"]["init_knapsack_sol"][i]]))
                    print("distance - {:>6.1f} | duration - {:>6.1f} | demand - {:>6.1f} | profit - {:>2.4f}"\
                        .format(self.search_algorithm.para["graph"]["init_distance"][self.search_algorithm.para["graph"]["init_knapsack_sol"][i]], 
                                self.search_algorithm.para["graph"]["init_duration"][self.search_algorithm.para["graph"]["init_knapsack_sol"][i]], 
                                self.search_algorithm.para["graph"]["init_demand"][self.search_algorithm.para["graph"]["init_knapsack_sol"][i]], 
                                self.search_algorithm.para["graph"]["init_profit"][self.search_algorithm.para["graph"]["init_knapsack_sol"][i]]))
                print("\033[37;41m the sum of duration - {:>5.3f} \033[0m".format(init_dura))
                print("\033[37;41m the sum of profit - {:>5.3f} \033[0m".format(init_profit))
                for i in [100, 200, 300, 500, 800, 1000, 1500, 2000]:
                    if i in cur_result["record"]:
                        print("\033[1;30;47m{}\033[0m".format("the solution after {} iteration - ".format(i)))
                        sol_dura, sol_profit = self.solution_parser.solutionParser(cur_result["record"][i]["maximal_profit_solution"], self.search_algorithm.para)
                        print("\033[37;41m the sum of duration - {:>5.3f} \033[0m".format(sol_dura))
                        print("\033[37;41m the sum of profit - {:>5.3f} \033[0m".format(sol_profit))
                print("the TIME                     - {:<15.5f}".format(end-start))
                print("####################################################################################################################")
                print()
        if plot: self._experimentResultSummary()


    def _experimentResultSummary(self):
        for k,v in self.result.items():
            print("####################################################")
            print("############### THE {:>5d} EXPERIMENT ###############".format(k))
            self.search_result_summary.plotProfit(v["record"], is_tqdm=False)
            print("####################################################")
            print()
        
    def _anExperiment(self, para, focus):
        self.search_algorithm.changeHyperPara(para)
        currentReport("initial", {}, para, focus)
        self.search_algorithm.run(is_tqdm=False)
        # self.search_algorithm.run()
        return [self.search_algorithm.info, self.search_algorithm.record]

        