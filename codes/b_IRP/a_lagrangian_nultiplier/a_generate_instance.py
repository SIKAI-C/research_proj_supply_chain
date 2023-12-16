import json
import random
import pandas as pd
from ...a_CVRP.a_tabu_search.b_fenge_input import FengEInfo

class GenerateGraph:

    def __init__(self):
        pathToJson = r"C:\Users\sikai\OneDrive\Desktop\A2_Research\code_simulator\\1_data\df_sample_new.json"
        self.graphs = [578,636,784]
        with open(pathToJson, 'r') as f:
            self.df = pd.DataFrame(json.load(f))
        self.df = self.df.sort_values(by='date', ascending=True)
        self.days = sorted(self.df['date'].unique())
        self.shelves = sorted(self.df['shelf_code'].unique())
        self.read_instance = FengEInfo()

    def generateAGraph(self, T=4, idx=0):
        start_day_idx = random.choice(range(len(self.days)-T))
        days = [self.days[i+start_day_idx] for i in range(T)]
        graph_input = self.read_instance.readRouteInstance(self.graphs[idx])
        result = {}
        demand = []
        for day in days:
            daily_demand = []
            # shelves = random.shuffle(self.shelves)
            shelves = self.shelves
            for shelf in shelves:
                daily_demand.append(int(self.df[(self.df['date']==day) & (self.df['shelf_code']==shelf)]['quantity_act'].sum()))
            daily_demand = [0] + daily_demand
            demand.append(daily_demand)
        
        result["capacity"] = graph_input['capacity']
        result["dimension"] = graph_input['dimension']    
        result['demand'] = demand
        result["duration"] = graph_input['duration']
        return result
        



