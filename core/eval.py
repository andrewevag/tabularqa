from typing import List, Union, Optional, Tuple, Any
from core.result_loader import ResultsLoader
from datasets import Dataset
from tqdm import tqdm
import databench_eval

class Evaluator:
    def __init__(
        self,
        qa : Dataset
    ):
        
        """
            Is used to evaluate the model. `compare` function is the function that compares the model response with the truth.
            By default is the one provided by the `databench_eval` package.
            qa is the Dataset to evaluate on.
        """
        self.qa = qa
        self.evaluator = databench_eval.Evaluator(qa=self.qa)
        self.compare = lambda value, truth, semantic: self.evaluator.default_compare(value, truth, semantic)

    def default_compare(self, value : Any, truth : Any, semantic=None) -> bool:
        """ This is a sample compare function. Not used."""
        return str(value).strip() == str(truth).strip()
    
    def eval(
        self,
        responses: Union[List[str], str],
        lite: bool = False,
        save_seq: bool = False
    ) -> Union[float, Tuple[float, List[int]]]:
        """
            Evaluate the model on the dataset.
            
            Args:
                responses: The responses to evaluate. Can be a list of responses or a path to a file with responses.
                lite: Whether to use databench_lite or not.
                save_seq: Whether to save the sequence of results or not.
            
            Returns:
                float: The accuracy of the model on the dataset.
        """
        if isinstance(responses, str):
            responses = ResultsLoader(responses).results['predictions']

        correct = 0
        truths = self.qa["answer"] if not lite else self.qa["sample_answer"]
        semantics = self.qa['type']
        
        results = []
        for response, truth, semantic in tqdm(zip(responses, truths, semantics), total=len(truths)):
            try:
                if self.compare(response, truth, semantic):
                    correct += 1
                    results.append(1)
                else:
                    results.append(0)
            except: 
                results.append(0)
                
        try: 
            # Τhe correct accuracy should be using the eval provided
            accuracy = self.evaluator.eval(responses, lite=lite)
        except:
            print('[ERROR] databench_eval\'s Evaluator failed. Using local definition of compare for accuracy accuracy')
            accuracy = correct / len(truths)
        if save_seq:
            return accuracy, results


        return accuracy
    

    def split_results(self, results : dict[str, Any], lite: bool=False) -> Tuple[dict[str, Tuple[float, int, int]], float]:
        responses = results['predictions']
        types = set(self.qa["type"])
        per_type = {t: [] for t in types}
        semantics = self.qa['type']

        for row, response, semantic in tqdm(zip(self.qa, responses, semantics), total=len(self.qa)):
            truth = row['answer'] if not lite else row['sample_answer']
            try:
                if self.compare(response, truth, semantic):
                    per_type[row['type']].append(1)
                else:
                    per_type[row['type']].append(0)
            except:
                per_type[row['type']].append(0)

        accuracies = {t: (sum(per_type[t]) / len(per_type[t]), sum(per_type[t]), len(per_type[t])) for t in types}
        total_accuracy = sum([sum(per_type[t]) for t in types]) / sum([len(per_type[t]) for t in types])

        return accuracies, total_accuracy

    def __str__(self) -> str:
        return f"Evaluator\n\tcompare : {self.compare}\n\tqa : {self.qa})"

