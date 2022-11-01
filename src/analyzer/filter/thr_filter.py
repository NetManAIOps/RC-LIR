from analyzer.filter import Filter
from models.data import Data
from models.rank import Rank

class ThrFilter(Filter):
    def __init__(self, thr):
        self.thr = thr

    def filter(self, data: Data, rank: Rank) -> Rank:
        ret = {}
        for r in rank.rank:
            if rank.rank[r] >= self.thr:
                ret[r] = rank.rank[r]
        return Rank(ret)
