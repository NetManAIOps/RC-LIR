import pandas as pd
import numpy as np
from reader.reader import Reader

class JsonReader(Reader):
    def get_df(self, filename):
        with open(filename) as f:
            df = pd.read_json(f, lines=True, dtype=np.dtype('float64'))
        return df