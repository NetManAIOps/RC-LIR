from pandas import DataFrame
import reader.reader as reader
from models.params import Params

class Data:
    def __init__(self, id, params: Params):
        reader = params.reader
        self.id = id
        self.normal: DataFrame = reader.get_df(params.normal_data(id))
        self.anomaly: DataFrame = reader.get_df(params.anomaly_data(id))
