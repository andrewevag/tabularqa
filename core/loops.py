from tqdm import tqdm
from core.pipelines import Pipeline
from core.cost import CostAccumulator
from core.eval import Evaluator
from core.saver import SerialSaver
from abc import ABC, abstractmethod
from datasets import Dataset
import sys


class Looper(ABC):
    def __init__(
        self,
        qa : Dataset,
        pipe : Pipeline,
        costpredictor : CostAccumulator,
        evaluator : Evaluator,
        saver : SerialSaver,
        debug=False,
        lite=False,
        save_seq=False
    ):
        self.qa = qa
        self.pipe = pipe
        self.costpredictor = costpredictor
        self.evaluator = evaluator
        self.saver = saver
        self.debug = debug
        self.lite = lite
        self.save_seq = save_seq

    @abstractmethod
    def loop(self):
        pass

class SerialLooper(Looper):
    def __init__(
        self,
        qa : Dataset,
        pipe : Pipeline,
        costpredictor : CostAccumulator,
        evaluator : Evaluator,
        saver : SerialSaver,
        debug=False,
        lite=False,
        save_seq=False,
        indexed=False,
    ):
        super().__init__(qa, pipe, costpredictor, evaluator, saver, debug, lite, save_seq)
        self.indexed = indexed
        

    def loop(self):
        broke_debug = False
        for i in tqdm(range(len(self.qa))):
            
            dataset = self.qa[i]['dataset']
            question = self.qa[i]['question']
            answer = self.qa[i]['answer']

            if self.indexed:
                prompt, raw_output, postprocessed, output, input_tokens, output_tokens = self.pipe.run_one((self.qa[i], i))
            else:
                prompt, raw_output, postprocessed, output, input_tokens, output_tokens = self.pipe.run_one(self.qa[i])
            
            # Save info for the cost calculator
            self.costpredictor.add(input_tokens, output_tokens)

            # if self.debug: print('Returned from cost predictor')

            self.saver.save(prompt, raw_output, postprocessed, output, dataset, question, answer, i)

            # if self.debug: print('Saved')
            
            if i == 0 and self.debug:
                print(f"Prompt: {prompt}")
                print(f"Raw output: {raw_output}")
                print(f"Postprocessed: {postprocessed}")
                print(f"Output: {output}")

                print("Shall I continue? (y/n)")
                sys.stdout.flush()
                if input() == 'n':
                    broke_debug = True
                    break

        
        # Calculate cost
        cost = self.costpredictor.total_cost()
        self.saver.write_total_cost(cost)

        # ===================================== Evaluate and write accuracy =====================================
        if not self.save_seq:
            accuracy = self.evaluator.eval(self.saver.results['predictions'], lite=self.lite, save_seq=self.save_seq)
        else:
            accuracy, correct_seq = self.evaluator.eval(self.saver.results['predictions'], lite=self.lite, save_seq=self.save_seq)
        
        self.saver.write_accuracy(accuracy)
        self.saver.write_correct_seq(correct_seq)
        # =======================================================================================================


        # ===================================== Evaluate and write accuracy per type ============================
        if not broke_debug:
            accuracies, _ = self.evaluator.split_results(self.saver.results, lite=self.lite)
            self.saver.write_accuracy_per_type(accuracies)
        else:
            accuracies = []
        # =======================================================================================================

        # =======================================================================================================
        # Finish the saver
        self.saver.finish()

        return accuracy, cost, accuracies
        
    def prediction_loop(self):
        for i in tqdm(range(len(self.qa))):
            
            dataset = self.qa[i]['dataset']
            question = self.qa[i]['question']

            if self.indexed:
                prompt, raw_output, postprocessed, output, input_tokens, output_tokens = self.pipe.run_one((self.qa[i], i))
            else:
                prompt, raw_output, postprocessed, output, input_tokens, output_tokens = self.pipe.run_one(self.qa[i])
            
            
            # Save info for the cost calculator
            self.costpredictor.add(input_tokens, output_tokens)

            self.saver.save(prompt, raw_output, postprocessed, output, dataset, question, None, i)

            if i == 0 and self.debug:
                print(f"Prompt: {prompt}")
                print(f"Raw output: {raw_output}")
                print(f"Postprocessed: {postprocessed}")
                print(f"Output: {output}")

                print("Shall I continue? (y/n)")
                sys.stdout.flush()
                if input() == 'n':
                    break

        # Calculate cost
        cost = self.costpredictor.total_cost()
        self.saver.write_total_cost(cost)

        # Finish the saver
        self.saver.finish()

        return cost
