import pandas as pd
from models.params import Params
from models.rank import Rank

class Result:
    def __init__(self, params: Params):
        self.output = pd.read_csv(params.output_data, na_filter=False, dtype=str)
        self.clean()

    def clean(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tot = 0
    
    def count(self, id, rank: Rank):
        std_ans = set(self.output.iloc[id]['stream_ans'].split(','))
        cur_ans = set(rank.rank.keys())

        self.tp += len(std_ans & cur_ans)
        self.fp += len(cur_ans - std_ans)
        self.fn += len(std_ans - cur_ans)
        self.tot += 1

        pass

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def precision(self):
        return self.tp / (self.tp + self.fp)
    
    def print(self):
        print(f'{self.tot} | TP={self.tp} FP={self.fp} FN={self.fn}')

        r = self.recall()
        p = self.precision()

        print(f'recall    = {r}')
        print(f'precision = {p}')
        print(f'f1_score  = {2 * r * p / (r + p)}')
