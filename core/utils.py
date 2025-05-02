import pandas as pd
from datasets import load_dataset, Dataset
from typing import Optional
from contextlib import contextmanager
import timeit
import json


# Caches the same table for faster loading since subsequent question tend to use the same table
# It does not cache every table to save on memory
CACHE_TABLE = None
CACHE_TABLE_NAME = None
def load_table(name : str) -> pd.DataFrame:
    """
        Load the table corresponding to the given name. The table is loaded from the `cardiffnlp/databench` dataset.
        This table is the table corresponding to a question in the QA dataset.

        Args:
            name: The name of the table to load.
        
        Returns:
            pd.DataFrame: The table corresponding to the given name.
    """
    global CACHE_TABLE
    global CACHE_TABLE_NAME
    if CACHE_TABLE_NAME != name:
        CACHE_TABLE = pd.read_parquet(
            f"https://huggingface.co/datasets/cardiffnlp/databench/resolve/main/data/{name}/all.parquet"
        )
        CACHE_TABLE_NAME = name
    return CACHE_TABLE

CACHE_SAMPLE = None
CACHE_SAMPLE_NAME = None
def load_sample(name: str) -> pd.DataFrame:
    """
        Load the sample table like the ones used to answer the questions.

        Args:
            name: The name of the table to load.

        Returns:
            pd.DataFrame: The sample table corresponding to the given name.
    """
    global CACHE_SAMPLE
    global CACHE_SAMPLE_NAME
    if CACHE_SAMPLE_NAME != name:
        CACHE_SAMPLE = pd.read_parquet(
            f"https://huggingface.co/datasets/cardiffnlp/databench/resolve/main/data/{name}/sample.parquet"
        )
        CACHE_SAMPLE_NAME = name
    return CACHE_SAMPLE

def load_test_data(test_data_dir='./competition'):
    """Loads the data for test set. test_qa."""
    test_data = pd.read_csv(f'{test_data_dir}/test_qa.csv')
    test_data = [{'question': row[1]['question'], 'dataset': row[1]['dataset']} for row in test_data.iterrows()]
    return test_data

def load_test_data_with_answers(test_data_dir='./competition'):
    test_data = pd.read_csv('./competition/test_qa.csv')

    questions = [row[1]['question'] for row in test_data.iterrows()]
    datasets  = [row[1]['dataset'] for row in test_data.iterrows()]

    test_data_ground_truth_file = f'{test_data_dir}/answers/answers.txt'
    with open(test_data_ground_truth_file, "r") as f:
        ground_truth = f.read().splitlines()

    test_data_semantics_f = f'{test_data_dir}/answers/semantics.txt'
    with open(test_data_semantics_f, "r") as f:
        test_data_semantics = f.read().splitlines()

    test_data_ground_truth_lite = f'{test_data_dir}/answers/answers_lite.txt'
    with open(test_data_ground_truth_lite, "r") as f:
        ground_truth_lite = f.read().splitlines()
    
    qa = Dataset.from_dict({"question" : questions, "dataset" : datasets, "answer": ground_truth, "type": test_data_semantics, "sample_answer" : ground_truth_lite})
    return qa

def test_load_table(table_name, test_data_dir='./competition'):
    # Loads a table for the test set
    path = f'{test_data_dir}/{table_name}/all.parquet'
    table = pd.read_parquet(path)
    return table

def test_load_sample(table_name, test_data_dir='./competition'):
    path = f'{test_data_dir}/{table_name}/sample.parquet'
    table = pd.read_parquet(path)
    return table

def generic_load_table(table_name : str) -> pd.DataFrame:
    try:
        return test_load_table(table_name)
    except:
        return load_table(table_name)

def generic_load_sample(table_name : str) -> pd.DataFrame:
    try:
        return test_load_sample(table_name)
    except:
        return load_sample(table_name)

@contextmanager
def debug_time(context: Optional[str] = None, debug=False):
    """Context manager for timing execution.

    This context manager measures the execution time of the code block within it and
    prints the time taken.

    Args:
        context (Optional[str], optional): The context of the timer, which is included
            in the printed message. Defaults to None.
    """

    if context is None:
        context = "[DEBUG] Execution completed"

    start_time: float = timeit.default_timer()

    try:
        yield
    finally:
        end_time: float = timeit.default_timer()
        execution_time: float = end_time - start_time
        if debug: print(f"{context} ({execution_time:.6f}s)")

def annotation_reader(file_path):
	with open(file_path, 'r') as f:
		annotations = json.load(f)
	header = annotations['dataset_split']
	name, split = header.strip().split(' ')
	print(f'Loading annotations from {name} {split}')
	
	dataset = load_dataset("cardiffnlp/databench", name=name, split=split)
	indices = [exemplar['index'] for exemplar in annotations['exemplars']]
	cotes = [exemplar['annotation'] for exemplar in annotations['exemplars']]

	test_subset = dataset.select(indices)
	exemplars = list(zip(test_subset, cotes))
	return indices, exemplars
