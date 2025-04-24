import multiprocessing
from typing import List
from core.santypes import DatasetRow
import core.utils as utils
from core.model_calls import Answerer
from core.prompt_generators import PromptGenerator
from core.postprocessors import PostProcessor
from core.executors import Executor, SaferMultiLineStatementExecutorListPassing
from collections import Counter
from typing import Union

class Pipeline:
	def __init__(
		self,
		prompt_generator: PromptGenerator,
		answerer: Answerer,
		postprocessor: PostProcessor,
		executor: Executor,
		parallelize_model: bool = False,
		model_timeout:int = 20,
		model_attempts:int = 3,
		executor_timeout:int = 60,
		debug: bool = False,
		indexed: bool = False
	):
		self.prompt_generator = prompt_generator
		self.answerer = answerer
		self.postprocessor = postprocessor
		self.executor = executor
		self.paralellize_model = parallelize_model
		self.model_timeout = model_timeout
		self.model_attempts = model_attempts
		self.executor_timeout = executor_timeout
		self.debug = debug
		self.indexed = indexed


	def __call__(self, row: DatasetRow) -> str:
		_, _, _, output, _, _ = self.run_one(row)
		return output

	def run_one(self, inp: Union[DatasetRow, tuple[DatasetRow, int]]):
		if self.indexed:
			row, i = inp
		else:
			row = inp
		with utils.debug_time("[DEBUG] Done generating prompt", self.debug):
			if self.indexed: prompt = self.prompt_generator((row, i))
			else: prompt = self.prompt_generator(row)

		with utils.debug_time("[DEBUG] Done generating model response", self.debug):
			raw_output, input_tokens, output_tokens = self.answerer(prompt)
		
		with utils.debug_time("[DEBUG] Done postprocessing model response", self.debug):
			postprocessed = self.postprocessor(raw_output)

		with utils.debug_time("[DEBUG] Done running code from response", self.debug):
			output = self.executor((postprocessed, row['dataset']))

		return prompt, raw_output, postprocessed, output, input_tokens, output_tokens

	def run(self, qa, debug=False, n_jobs=4):
		with utils.debug_time("[DEBUG] Done generating prompts", debug), multiprocessing.Pool(n_jobs) as pool:
			prompts = pool.map(self.prompt_generator, qa)


		with utils.debug_time('[DEBUG] Done Generating model responses', debug):
			if self.paralellize_model:
				with multiprocessing.Pool(n_jobs) as pool:
					raw_responses = pool.map(self.answerer, prompts)
			else:
				raw_responses = [self.answerer(prompt) for prompt in prompts]

		input_tokens = sum(map(lambda x: x[1], raw_responses))
		output_tokens = sum(map(lambda x: x[2], raw_responses))
		raw_responses = list(map(lambda x: x[0], raw_responses))

		with utils.debug_time('[DEBUG] Done Postprocessing model responses'), multiprocessing.Pool(n_jobs) as pool:
			responses = pool.map(self.postprocessor, raw_responses)

		datasets = qa['dataset'].copy()
		args = list(zip(responses, datasets))
		with utils.debug_time('[DEBUG] Done running code from responses', debug), multiprocessing.Pool(n_jobs) as pool:
			runned_responses = pool.map(self.executor, args)

		return prompts, raw_responses, responses, runned_responses, input_tokens, output_tokens
	
	def __str__(self) -> str:
		prompt_generator_info = "\n\t".join(str(self.prompt_generator).split('\n'))
		answerer_info = "\n\t".join(str(self.answerer).split('\n'))
		postprocessor_info = "\n\t".join(str(self.postprocessor).split('\n'))
		executor_info = "\n\t".join(str(self.executor).split('\n'))
		return f"Pipeline\n\t{prompt_generator_info}\n\t{answerer_info}\n\t{postprocessor_info}\n\t{executor_info}"

class MajorityPipeline:
	def __init__(self, pipelines: List[Pipeline]):
		self.pipelines = pipelines
		self.executor = self.pipelines[0].executor

	def __call__(self, row: DatasetRow):
		_, _, _, output, _, _ = self.run_one(row)
		return output
	
	def most_common(self, lst):
		serialized_items = [repr(item) for item in lst]
		most_common_serialized = Counter(serialized_items).most_common(1)[0][0]

		for item in lst:
			if repr(item) == most_common_serialized:
				return item
		
	
	def run(self, qa, debug=False, n_jobs=4):
		outputs = [pipeline.run(qa, debug, n_jobs) for pipeline in self.pipelines]
		return outputs
	
	def run_one(self, row: DatasetRow):
		outputs = [pipeline.run_one(row) for pipeline in self.pipelines]

		clear_outputs = []
		input_tokens_total = 0
		output_tokens_total = 0
		prompt = outputs[0][0]
		raw_output_total = []
		postprocessed_total = []
		for prompt, raw_output, postprocessed, output, input_tokens, output_tokens in outputs:
			clear_outputs.append(output)
			input_tokens_total += input_tokens
			output_tokens_total += output_tokens
			raw_output_total.append(raw_output)
			postprocessed_total.append(postprocessed)

		output_final = self.most_common(clear_outputs)
		# print(f"clear_outputs: {clear_outputs}")
		return prompt, raw_output_total, postprocessed_total, output_final, input_tokens_total, output_tokens_total
	
	def run_one_return_all(self, row: DatasetRow):
		outputs = [pipeline.run_one(row) for pipeline in self.pipelines]
		return outputs

	def __str__(self) -> str:
		return "MajorityPipeline:\n\t"+ "\n\t".join([str(pipeline) for pipeline in self.pipelines])

class ErrorFixPipeline(Pipeline):
	def __init__(
		self,
		prompt_generator: PromptGenerator,
		answerer: Answerer,
		postprocessor: PostProcessor,
		executor: Executor,
		executor_timeout:int = 300,
		debug: bool = False,
	):
		self.prompt_generator = prompt_generator
		self.answerer = answerer
		self.postprocessor = postprocessor
		self.executor = executor
		self.executor_timeout = executor_timeout
		self.debug = debug


	def __call__(self, inp) -> str:
		_, _, _, output, _, _ = self.run_one(inp)
		return output

	def run_one(self, inp):
		row, postprocessed, error_msg = inp
		prompt = self.prompt_generator(inp)
		raw_output, input_tokens, output_tokens = self.answerer(prompt)
		postprocessed = self.postprocessor(raw_output)
		prediction = self.executor((postprocessed, row['dataset']))
		return prompt, raw_output, postprocessed, prediction, input_tokens, output_tokens


	def run(self, qa, debug=False, n_jobs=4):
		pass
	
	def __str__(self) -> str:
		prompt_generator_info = "\n\t".join(str(self.prompt_generator).split('\n'))
		answerer_info = "\n\t".join(str(self.answerer).split('\n'))
		postprocessor_info = "\n\t".join(str(self.postprocessor).split('\n'))
		executor_info = "\n\t".join(str(self.executor).split('\n'))
		return f"ErrorFixPipeline\n\t{prompt_generator_info}\n\t{answerer_info}\n\t{postprocessor_info}\n\t{executor_info}"

class ErrorFixingAndTimeoutPipeline(Pipeline):
	def __init__(
		self,
		main_pipeline: Pipeline,
		error_fix_pipeline: ErrorFixPipeline,
		max_timeout: int = 300,
		num_attempts: int = 2 
	):
		self.main_pipeline = main_pipeline
		self.error_fix_pipeline = error_fix_pipeline
		self.max_timeout = max_timeout
		self.num_attempts = num_attempts


	def __call__(self, inp) -> str:
		row = inp
		_, _, _, output, _, _ = self.run_one(row)
		return output

	def run_one(self, inp):
		row = inp
		prompt, raw_output, postprocessed, output, input_tokens, output_tokens = self.main_pipeline.run_one(row)

		valid_output = True
		
		prompts = [prompt]
		raw_outputs = [raw_output]
		postprocesseds = [postprocessed]

		if str(output).startswith("__TIMEOUT__"):
			valid_output = False
			previous_timeout = self.main_pipeline.executor.timeout 
			self.main_pipeline.executor.timeout = self.max_timeout
			executor_backup = self.main_pipeline.executor
			self.main_pipeline.executor =  SaferMultiLineStatementExecutorListPassing(utils.generic_load_table, timeout=self.max_timeout)
			prompt_timeout, raw_output_timeout, postprocessed_timeout, output_timeout, input_tokens_timeout, output_tokens_timeout = self.main_pipeline.run_one(row)

			if not str(output_timeout).startswith("__TIMEOUT__"):
				output = output_timeout
				valid_output = True

				prompts += ['TIMEOUT FIXED', str(prompt_timeout)]
				raw_outputs += ['TIMEOUT FIXED', str(raw_output_timeout)]
				postprocesseds += ['TIMEOUT FIXED', str(postprocessed_timeout)]
			else:
				prompts += ['TIMEOUT NOT FIXED', str(prompt_timeout)]
				raw_outputs += ['TIMEOUT NOT FIXED', str(raw_output_timeout)]
				postprocesseds += ['TIMEOUT NOT FIXED', str(postprocessed_timeout)]

			input_tokens += input_tokens_timeout
			output_tokens += output_tokens_timeout
			# restore timeout & executor
			self.main_pipeline.executor = executor_backup
			self.main_pipeline.executor.timeout = previous_timeout
			


		if str(output).startswith("__CODE_ERROR__"):
			# print(str(output))
			valid_output = False

		count = 0

		while not valid_output:
			print(f'{count}th error fixing')
			error_msg = output
			prompt_error_fix, raw_output_error_fix, postprocessed_error_fix, output_error_fix, input_tokens_error_fix, output_tokens_error_fix = self.error_fix_pipeline.run_one((row, postprocessed, error_msg))
			count += 1
			# If the error is fixed
			if not str(output_error_fix).startswith("__CODE_ERROR__") and not str(output_error_fix).startswith("__TIMEOUT__"):
				output = output_error_fix
				prompts += ['ERROR FIXED', str(prompt_error_fix)]
				raw_outputs += ['ERROR FIXED', str(raw_output_error_fix)]
				postprocesseds += ['ERROR FIXED', str(postprocessed_error_fix)]
				input_tokens += input_tokens_error_fix
				output_tokens += output_tokens_error_fix
				valid_output = True
				break
			
			prompts += ['ERROR NOT FIXED', str(prompt_error_fix)]
			raw_outputs += ['ERROR NOT FIXED', str(raw_output_error_fix)]
			postprocesseds += ['ERROR NOT FIXED', str(postprocessed_error_fix)]
			input_tokens += input_tokens_error_fix
			output_tokens += output_tokens_error_fix
			output = output_error_fix

			if count > self.num_attempts: break

		return prompts, raw_outputs, postprocesseds, output, input_tokens, output_tokens
	

	def run(self, qa, debug=False, n_jobs=4):
		pass

	def __str__(self) -> str:
		return f"ErrorFixingAndTimeoutPipeline\n\t{str(self.main_pipeline)}\n\t{str(self.error_fix_pipeline)}"

