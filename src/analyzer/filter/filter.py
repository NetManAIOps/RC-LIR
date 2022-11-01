from models.data import Data
from models.rank import Rank

class Filter:
    def filter(self, data: Data, rank: Rank) -> Rank:
        return rank