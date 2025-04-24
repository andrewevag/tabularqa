import json

class ResultsLoader:
    def __init__(self, path: str):
        self.path = path
        # Load the json in the path
        with open(self.path, 'r') as f:
            self.results = json.load(f)

