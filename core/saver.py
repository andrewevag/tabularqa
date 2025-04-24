import datetime
import os
import json
from typing import Dict, Any
from collections import defaultdict

class SerialSaver:
    def __init__(self, save_prefix: str, msg: str, lite: bool = False):
        self.save_prefix = save_prefix
        self.lite = lite
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)

        # The list as of now on what to save is:
        # 1. prompts
        # 2. raw_responses
        # 3. responses (postprocessed)
        # 4. predictions
        # 5. datasets
        # 6. questions
        # 7. answers
        # 8. accuracy_score
        # 9. accuracy score per type of question
        # 10. Boolean list of correct sequences
        # 11. Experiment message
        
        self.results : Dict[str, Any] = defaultdict(list) 
        self.results['message'] = f"Running experiment @{datetime.datetime.now()}\n" + msg


    def save(self, prompt, raw_response, response, runned_response, dataset, question, answer, idx):
        self.results['prompts'].append(prompt)
        self.results['raw_responses'].append(raw_response)
        self.results['responses'].append(str(response))
        self.results['predictions'].append(str(runned_response))
        self.results['datasets'].append(dataset)
        self.results['questions'].append(question)
        self.results['answers'].append(str(answer))
        
    def finish(self):
        results_path = os.path.join(self.save_prefix, f'results.json')
        with open(results_path, 'w') as outfile:
            json.dump(self.results, outfile, indent=4)

    def write_accuracy(self, accuracy):
        self.results['accuracy'] = accuracy

    def write_accuracy_per_type(self, accuracy_per_type_dict):
        for key, value in accuracy_per_type_dict.items():
            self.results[f'accuracy_{key}'] = value
    
    def write_correct_seq(self, correct_seq):
        self.results['correct_seq'] = correct_seq

    def write_total_cost(self, cost):
        self.results['total_cost'] = cost
