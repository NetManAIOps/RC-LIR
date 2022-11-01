from models.params import Params

class Rank:
    def __init__(self, rank: dict):
        self.rank: dict = rank
    
    def sorted_fields(self) -> list:
        return sorted(self.rank.keys(), key=lambda x: -self.rank[x])

    def print(self):
        for i in self.sorted_fields():
            print(f'{i}: {self.rank[i]}')
        print('---')