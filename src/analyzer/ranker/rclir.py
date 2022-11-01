import pandas as pd
import numpy as np
import math
from models import Data, Params, Rank
from analyzer.ranker import Ranker
import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, data: Data, params: Params):
        super(LogisticRegression, self).__init__()
        
        self.fields = set(data.normal.keys()) - set(params.disable_cols)
        self.w = torch.nn.Parameter(torch.ones(len(self.fields)), requires_grad=True)

        normal = pd.concat([data.normal, data.anomaly, data.anomaly]).drop_duplicates(keep=False)
        self.anomaly = torch.from_numpy(data.anomaly[self.fields].applymap(hash).values)
        self.normal = torch.from_numpy(normal[self.fields].applymap(hash).values)
        self.scale = 10000
        
        if torch.cuda.is_available():
            self.anomaly = self.anomaly.cuda()
            self.normal = self.normal.cuda()

        self.l = params.LAMBDA
        self.sample = params.SAMPLE
    
    def dist(self, a, b):
        if b.shape[0] > self.sample:
            index = torch.tensor(np.random.choice(range(b.shape[0]), self.sample), dtype=torch.int64)
            if torch.cuda.is_available():
                index = index.cuda()
            index = index.unsqueeze(1).repeat(1, b.shape[1])
            b = torch.gather(b, 0, index)
        

        return torch.unsqueeze(a, 1) != torch.unsqueeze(b, 0)

    def cost(self, dist_hh, dist_hm):
        dist_hh = torch.sum(dist_hh * self.w.reshape((1, 1, -1)), 2) # len(H) * len(S)
        dist_hm = torch.sum(dist_hm * self.w.reshape((1, 1, -1)), 2) # len(H) * len(S)

        d0 = len(self.fields) / 2
        pnear_hh = torch.clamp(1 - dist_hh / d0, min=0)
        pnear_hh = torch.where(pnear_hh <= (1 - 1e-6), pnear_hh, torch.zeros_like(pnear_hh))
        pnear_hh = pnear_hh / torch.clamp(torch.sum(pnear_hh, dim=(1,), keepdim=True), min=1)
        pnear_hm = torch.clamp(1 - dist_hm / d0, min=0)
        pnear_hm = torch.where(pnear_hm <= (1 - 1e-6), pnear_hm, torch.zeros_like(pnear_hm))
        pnear_hm = pnear_hm / torch.clamp(torch.sum(pnear_hm, dim=(1,), keepdim=True), min=1)
        score = (torch.sum(pnear_hm * dist_hm, 1) - torch.sum(pnear_hh * dist_hh, 1)) / self.scale # len(H)

        return torch.mean(torch.log(1 + torch.exp(-score)))


    def forward(self):
        dist_nn = self.dist(self.normal, self.normal)
        dist_na = self.dist(self.normal, self.anomaly)
        dist_aa = self.dist(self.anomaly, self.anomaly)
        dist_an = self.dist(self.anomaly, self.normal)
        return self.scale * (self.cost(dist_nn, dist_na) + self.cost(dist_aa, dist_an)) + self.l * torch.sum(self.w ** 2)

class RCLIR(Ranker):
    def rank(self, data: Data, params: Params) -> Rank:
        logistic_model = LogisticRegression(data, params)
        if torch.cuda.is_available():
            logistic_model.cuda()
        optimizer = torch.optim.SGD(logistic_model.parameters(), lr=params.LR, momentum=params.MOMENTUM)

        for epoch in range(params.RUNS_EPOCH):
            cost = logistic_model()
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
        rank = dict(zip(logistic_model.fields, (logistic_model.w ** 2).tolist()))
        return Rank(rank)
