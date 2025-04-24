from dataclasses import dataclass

@dataclass
class CostAccumulator:
	"""
		Class to calculate cost of running an experiment. All are caclulated in terms of tokens per thousand.
	"""
	pit: float = 0.0
	pot: float = 0.0
	input_lengths: int = 0
	output_lengths: int = 0

	def add(self, len_input, len_output):
		"""Adds the input and output lengths to the total cost."""
		self.input_lengths += len_input
		self.output_lengths += len_output

	def total_cost(self) -> float:
		"""Calculates the total cost of the experiment."""
		cost = (self.input_lengths / 1000) * self.pit + (self.output_lengths / 1000) * self.pot
		return cost
	
	def reset(self):
		"""Resets the cost calculator."""
		self.input_lengths = 0
		self.output_lengths = 0

