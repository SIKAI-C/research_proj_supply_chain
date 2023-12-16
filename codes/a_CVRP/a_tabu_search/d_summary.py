from math import dist
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

class SearchResultSummary:

    def __init__(self):
        pass

    def plotProfit(self, record, is_tqdm=True):
        record_points       = len(list(record.keys()))
        x                   = [0]*record_points
        cur_feasible_prft   = [0]*record_points
        cur_opt_prft        = [0]*record_points
        idx = 0

        if is_tqdm == True:
            for k in tqdm(record.keys()):
                x[idx]                       = k
                cur_feasible_prft[idx]       = record[k]["cur_feasible_profit"]
                cur_opt_prft[idx]            = record[k]["maximal_profit"]
                idx += 1
        else:
            for k in record.keys():
                x[idx]                       = k
                cur_feasible_prft[idx]       = record[k]["cur_feasible_profit"]
                cur_opt_prft[idx]            = record[k]["maximal_profit"]
                idx += 1
                
        plt.plot(x, cur_feasible_prft, label="feasible")
        plt.plot(x, cur_opt_prft, label="optimal")
        try:
            plt.ylim(min(cur_feasible_prft)-1, max(cur_feasible_prft)+1)
        except:
            pass
        plt.legend()
        plt.title("Profit versus Iterations.")
        plt.show()


    

