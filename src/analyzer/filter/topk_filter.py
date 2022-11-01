from analyzer.filter import Filter
from models.data import Data
from models.rank import Rank

class TopKFilter(Filter):
    def __init__(self, k):
        self.k = k

    def filter(self, data: Data, rank: Rank) -> Rank:
        ret = {}
        for r in rank.sorted_fields()[:self.k]:
            ret[r] = rank.rank[r]
        return Rank(ret)
