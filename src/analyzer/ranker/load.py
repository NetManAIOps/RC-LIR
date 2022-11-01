from models import Data, Params, Rank
from analyzer.ranker import Ranker

class Load(Ranker):
    def __init__(self, file):
        self.ans = {}
        with open(file) as f:
            for line in f:
                if ',' in line:
                    x = line[:-1].split(',')
                    self.ans[x[0]] = x[1:]

    def rank(self, data: Data, params: Params) -> Rank:
        id = str(data.id)
        if id in self.ans:
            rank = {}
            for x in self.ans[id]:
                rank[x] = 1
            return Rank(rank)
        else:
            print(f'id {id} not found')
            return Rank({})
