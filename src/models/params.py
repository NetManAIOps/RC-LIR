class Params:
    def __init__(self):
        from reader import CsvReader
        from analyzer.ranker import RCLIR, Load
        from analyzer.filter import TopKFilter, ThrFilter

        # data
        self.reader = CsvReader()
        self.normal_data = lambda id: f'data/inputs/{id}.csv'
        self.anomaly_data = lambda id: f'data/inputs/{id}f.csv'
        self.output_data = 'data/inputs/std.csv'
        self.test_range = range(1000)

        # options
        self.max_candidate = 10
        self.disable_cols = ['streams']
        self.print_res = False
        self.save_res = None

        name = 'RCLIR'
        if not name:
            raise(Exception('algorithm name is empty'))
        
        if name == 'RCLIR':
            # RCLIR
            self.ranker = RCLIR()
            self.filters = [ThrFilter(1), TopKFilter(self.max_candidate)]
            self.RUNS_EPOCH = 2000
            self.LR = 0.01
            self.LAMBDA = 0.01
            self.MOMENTUM = 0
            self.SAMPLE = 128
            self.save_res = 'data/test_cases/outputs/rclir.csv'

        elif name[:5] == 'load ':
            self.ranker = Load(f'data/test_cases/outputs/{name[5:]}.csv')
            self.filters = [TopKFilter(self.max_candidate)]

        else:
            raise(Exception(f'algorithm {name} not found'))
