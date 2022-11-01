import pandas as pd
import numpy as np
from reader.reader import Reader

class CsvReader(Reader):
    def get_df(self, filename):
        return pd.read_csv(filename, na_filter=False, dtype=str)