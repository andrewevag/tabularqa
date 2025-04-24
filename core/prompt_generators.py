from pyarrow import Table
from abc import ABC, abstractmethod
from core.santypes import DatasetRow, TableLoader
import pandas as pd
from typing import Any, Union

# =========================================== Hierarchy of Prompt Generators ===========================================
# PromptGenerator (ABC)
# │
# ├── ZeroShotGenerator (ABC)
# │   ├── ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsVol2
# │   └── ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsUniquesStrExamples
# │
# ├── ExemplarBuilder 
# │
# ├── FewShotGenerator (ABC)
# │   └── FewShotChainOfThoughtBuilderWithTypesRowsAndPredictingResultingTypesVol2
# │
# ├── ErrorFixingGenerator (ABC)
# │   ├── ErrorFixingGeneratorClaude
# │   ├── ErrorFixingGeneratorClaudeVol2
# │   ├── ErrorFixingGeneratorOLlama
# │   └── ErrorFixingGeneratorLlama
# │
# └── WrapperGenerator (ABC)
#     └── ClaudeMessageEmbeddingPromptGenerator
# ======================================================================================================================

class PromptGenerator(ABC):
    def __init__(self, loader: TableLoader):
        self.loader = loader
    
    @abstractmethod
    def __call__(self, input_data) -> Any:
        pass

class ZeroShotGenerator(PromptGenerator):
    def __init__(self, loader: TableLoader):
        super().__init__(loader)
    
    @abstractmethod
    def __call__(self, row: DatasetRow) -> str:
        pass

class ExemplarBuilderGenerator(PromptGenerator):
    def __init__(self, loader: TableLoader, prompt_generator: ZeroShotGenerator):
        super().__init__(loader)
        self.prompt_generator = prompt_generator
    
    def __call__(self, exemplar: tuple[DatasetRow, str]) -> str:
        # Implementation as before
        pass

class FewShotGenerator(PromptGenerator):
    def __init__(self, loader: TableLoader, prompt_generator: PromptGenerator, exemplar_builder: ExemplarBuilderGenerator, shots: list[tuple[DatasetRow, str]]):
        super().__init__(loader)
        self.prompt_generator = prompt_generator
        self.exemplar_builder = exemplar_builder
        self.shots = shots
    
    @abstractmethod
    def __call__(self, row: DatasetRow) -> str:
        pass

class ErrorFixingGenerator(PromptGenerator):
    def __init__(self, loader: TableLoader, zero_shot_generator: ZeroShotGenerator):
        super().__init__(loader)
        self.zero_shot_generator = zero_shot_generator
    
    @abstractmethod
    def __call__(self, inp: tuple[DatasetRow, str, str]) -> Any:  # (row, postprocessed, error_msg)
        pass

class WrapperGenerator(PromptGenerator):
    def __init__(self, loader: TableLoader, prompt_generator: PromptGenerator):
        super().__init__(loader)
        self.prompt_generator = prompt_generator
    
    @abstractmethod
    def __call__(self, row: DatasetRow) -> Any:
        pass

# =========================================== Custom Functions for Generating CSV Info ===========================================
def custom_info_csv_2(df : pd.DataFrame) -> str:
    # Add the header for the output
    output = ["#,Column,Non-Null CounT,Dtype,Types of Elements,Values"]
    
    for i, col in enumerate(df.columns):
        non_null_count = df[col].notnull().sum()
        dtype = df[col].dtype
        python_inner_types = df[col].apply(lambda x: type(x)).unique().tolist()
        # category-specific logic
        values = ""
        if dtype.name == "category":
            unique_values = df[col].cat.categories.tolist()
            if len(unique_values) <= 5:
                values = f"All values are {unique_values}"
            else:
                example_values = unique_values[:5]
                values = f"5 example values are {example_values}"
        
        # Append row to the output
        output.append(f"{i},{col},{non_null_count},{dtype},{python_inner_types},{values}")
    
    # Join the output list into a single string
    return "\n".join(output)

def custom_info_csv_5(df : pd.DataFrame) -> str:
    # Add the header for the output
    output = ["#,Column,Non-Null CounT,Dtype,Types of Elements,Values, Are all values unique?"]
    
    for i, col in enumerate(df.columns):
        non_null_count = df[col].notnull().sum()
        dtype = df[col].dtype
        python_inner_types = df[col].apply(lambda x: type(x)).unique().tolist()
        # category-specific logic
        values = ""
        if dtype.name == "category":
            unique_values = df[col].cat.categories.tolist()
            if len(unique_values) <= 5:
                values = f"All values are {unique_values}"
            else:
                example_values = unique_values[:5]
                values = f"5 example values are {example_values}"
        
        if dtype.name == 'object' and type("") in python_inner_types:
            unique_values = list(set((df[col])))
            if len(unique_values) <= 5:
                values = f"All values are {unique_values}"
            else:
                example_values = unique_values[:5]
                values = f"5 example values are {example_values}"
        
        try:
            are_all_values_unique = df[col].nunique() == df[col].count()
        except:
            are_all_values_unique = "Unhashable type"

        # Append row to the output
        output.append(f"{i},{col},{non_null_count},{dtype},{python_inner_types},{values}, {are_all_values_unique}")
    
    # Join the output list into a single string
    return "\n".join(output)

# =========================================== Prompt Generators Implementations ===========================================
class ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsVol2(ZeroShotGenerator):

    def __init__(self, loader : TableLoader, num_rows : int=5, lite : bool=False):
        super().__init__(loader)
        self.num_rows = num_rows
        self.lite = lite

    def __call__(self, row: DatasetRow) -> str:
        dataset = row["dataset"]
        question = row["question"]
        df = self.loader(dataset)

        if not self.lite:
            res = f"""\
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
        {"\n        ".join(custom_info_csv_2(df).split('\n'))}

        The first {self.num_rows} rows from the dataframe:
        {"\n        ".join(df.head(self.num_rows).to_string(max_colwidth=50).split('\n'))}
    \"\"\"


    df.columns = {list(df.columns)}
    """
        else:
            res = f"""\
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
        {"\n        ".join(custom_info_csv_2(df).split('\n'))}

        All {self.num_rows} rows from the dataframe:
        {"\n        ".join(df.head(self.num_rows).to_string(max_colwidth=50).split('\n'))}
    \"\"\"


    df.columns = {list(df.columns)}
    """
        return res

    def __str__(self):
        return f"ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsVol2(num_rows={self.num_rows}, lite={self.lite})"

class ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsUniquesStrExamples(ZeroShotGenerator):

    def __init__(self, loader : TableLoader, num_rows : int=5, lite : bool=False):
        super().__init__(loader)
        self.num_rows = num_rows
        self.lite = lite

    def __call__(self, row: DatasetRow) -> str:
        dataset = row["dataset"]
        question = row["question"]
        df = self.loader(dataset)
        

        if not self.lite:
            res = f"""\
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
        {"\n        ".join(custom_info_csv_5(df).split('\n'))}

        The first {self.num_rows} rows from the dataframe:
        {"\n        ".join(df.head(self.num_rows).to_string(max_colwidth=100).split('\n'))}
    \"\"\"


    df.columns = {list(df.columns)}
    """
        else:
            res = f"""\
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
        {"\n        ".join(custom_info_csv_5(df).split('\n'))}

        All {self.num_rows} rows from the dataframe:
        {"\n        ".join(df.head(self.num_rows).to_string(max_colwidth=100).split('\n'))}
    \"\"\"


    df.columns = {list(df.columns)}
    """
        return res

    def __str__(self):
        return f"ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsUniquesStrExamples(num_rows={self.num_rows}, lite={self.lite})"

class ExemplarBuilder(ExemplarBuilderGenerator):
    def __init__(self, loader : TableLoader, prompt_generator : PromptGenerator):
        super().__init__(loader, prompt_generator)

    def __call__(self, exemplar : tuple[DatasetRow, str]) -> str:
        row, annotation = exemplar
        pre = self.prompt_generator(row)
        columns_used = row['columns_used']
        column_types = row['column_types']
        type_of_answer = row['type']
        intermediate = f"""
    # The columns used to answer the question: {columns_used}
    # The types of the columns used to answer the question: {column_types}
    # The type of the answer: {type_of_answer}"""
        annotation_lines = list(map(lambda x: "    " + x, annotation.split('\n')))
        annotation = "\n".join(annotation_lines)
        
        return pre + intermediate + "\n\n" + annotation

    def __str__(self) -> str:
        return f"ExemplarBuilder(PromptGenerator)\n\tprompt_generator: {self.prompt_generator}"

class FewShotChainOfThoughtBuilderWithTypesRowsAndPredictingResultingTypesVol2(FewShotGenerator):
    def __init__(self, loader : TableLoader, prompt_generator : ZeroShotGenerator, exemplar_builder : ExemplarBuilderGenerator, shots : list[tuple[DatasetRow, str]]):
        super().__init__(loader, prompt_generator, exemplar_builder, shots)
        shottemplates = []
        for (shotrow, predicted) in self.shots:
            template = self.exemplar_builder((shotrow, predicted))
            shottemplates.append(template)
        
        self.shottemplates = shottemplates

    def __call__(self, row: DatasetRow) -> str:
        

        prompt = self.prompt_generator(row)
        intermediate = f"""
    # The columns used to answer the question: """
        incomplete = prompt + intermediate
        templates = self.shottemplates + [incomplete]
        out = "\n".join(templates)
        return out
    
    def __str__(self) -> str:
        return f"FewShotChainOfThoughtBuilderWithTypesRowsAndPredictingResultingTypesVol2(PromptGenerator)\n\tprompt_generator: {self.prompt_generator}\n\texemplar_builder: {self.exemplar_builder}\n\tshots: {self.shots}"

class ClaudeChatBuilder:
    def __init__(self):
        self.payload = {
            "messages": []
        }



    def add_user_message(self, message):
        self.payload["messages"].append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ]
        })

    def add_assistant_message(self, message):
        self.payload["messages"].append({
            "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
        })

class ClaudeMessageEmbeddingPromptGenerator(WrapperGenerator):
    def __init__(self, loader : TableLoader, prompt_generator : PromptGenerator):
        super().__init__(loader, prompt_generator)

    def __call__(self, row: DatasetRow) -> Any:
        txt = self.prompt_generator(row)
        chat = ClaudeChatBuilder()
        chat.add_assistant_message(txt[:-1])
        return chat.payload

    def __str__(self) -> str:
        return f"ClaudeMessageEmbeddingPromptGenerator(PromptGenerator)\n\tprompt_generator: {self.prompt_generator}"
    
class ErrorFixingGeneratorClaude(ErrorFixingGenerator):    
    def __init__(self, loader : TableLoader, zero_shot_generator : ZeroShotGenerator, num_rows : int=5):
        super().__init__(loader, zero_shot_generator)
        self.num_rows = num_rows

    def __call__(self, inp):
        row, postprocessed, error_msg = inp
        chat = ClaudeChatBuilder()

        chat.add_user_message(
        f"""
# Help me fix the code error of the following function by rewriting it. The function should return the answer to the question in the TODO comment below:

{self.zero_shot_generator(row)}

{postprocessed}


# The function outputs the following error:
# {error_msg}"""

        )

        chat.add_assistant_message(
            f"""{self.zero_shot_generator(row).rstrip()}"""
        )

        # print(chat.payload['messages'][0]['content'][0]['text'])
        return chat.payload
    
    def __str__(self) -> str:
        return f"ErrorFixingGenerator(PromptGenerator)\n\tloader: {self.loader}, \n\tnum_rows: {self.num_rows}, \n\tzero_shot_generator: {self.zero_shot_generator}"

class ErrorFixingGeneratorClaudeVol2(ErrorFixingGenerator):    
    def __init__(self, loader : TableLoader, zero_shot_generator : ZeroShotGenerator, num_rows : int=5):
        super().__init__(loader, zero_shot_generator)
        self.num_rows = num_rows

    def __call__(self, inp):
        row, postprocessed, error_msg = inp
        chat = ClaudeChatBuilder()

        chat.add_user_message(
        f"""
# Help me fix the code error of the following function by rewriting it. Try to parse columns with list types yourself instead of using the `eval` function. Some lists may be written without the necessary '' to be parsed correctly. If rare or special characters are included as values, test equality by substring detection e.g. "query" in df[col].
# The function should return the answer to the question in the TODO comment below:

{self.zero_shot_generator(row)}

{postprocessed}


# The function outputs the following error:
# {error_msg}"""

        )

        chat.add_assistant_message(
            f"""{self.zero_shot_generator(row).rstrip()}"""
        )


        return chat.payload
        
    
    def __str__(self) -> str:
        return f"ErrorFixingGeneratorClaudeVol2(PromptGenerator)\n\tloader: {self.loader}, \n\tnum_rows: {self.num_rows}, \n\tzero_shot_generator: {self.zero_shot_generator}"

class ErrorFixingGeneratorOLlama(ErrorFixingGenerator):    
    def __init__(self, loader : TableLoader, zero_shot_generator : ZeroShotGenerator, num_rows: int=5):
        super().__init__(loader, zero_shot_generator)
        self.num_rows = num_rows

    def __call__(self, inp):
        row, postprocessed, error_msg = inp
        chat = []

        chat.append(
        {
            'role': 'user',
            'content' : f"""
# Help me fix the code error of the following function by rewriting it. Try to parse columns with list types yourself instead of using the `eval` function. Some lists may be written without the necessary '' to be parsed correctly. If rare or special characters are included as values, test equality by substring detection e.g. "query" in df[col].
# The function should return the answer to the question in the TODO comment below:

{self.zero_shot_generator(row)}

{postprocessed}


# The function outputs the following error:
# {error_msg}"""

        })

        chat.append(
            {
                'role' : 'assistant',
                'content': f"""{self.zero_shot_generator(row).rstrip()}""".replace('def answer', 'def answer_fixed')
            }
        )
        
        
        return chat
        
    
    def __str__(self) -> str:
        return f"ErrorFixingGeneratorOLlama(PromptGenerator)\n\tloader: {self.loader}, \n\tnum_rows: {self.num_rows}, \n\tzero_shot_generator: {self.zero_shot_generator}"

def llama3_instr_format_second(system_prompt: str, user_prompt : str, assistant: str) -> str:
    return f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}"""

class ErrorFixingGeneratorLlama(ErrorFixingGenerator):    
    def __init__(self, loader : TableLoader, zero_shot_generator : ZeroShotGenerator, num_rows : int=5):
        super().__init__(loader, zero_shot_generator)
        self.num_rows = num_rows

    def __call__(self, inp) -> str:
        row, postprocessed, error_msg = inp
        prompt = llama3_instr_format_second(
            "You are an assistant tasked with helping a user fix a code error. The user has written a function that is supposed to answer a question about a table.",
            f"""
# Help me fix the code error of the following function by rewriting it. Try to parse columns with list types yourself instead of using the `eval` function. Some lists may be written without the necessary '' to be parsed correctly. If rare or special characters are included as values, test equality by substring detection e.g. "query" in df[col].
# The function should return the answer to the question in the TODO comment below:

{self.zero_shot_generator(row)}

{postprocessed}


# The function outputs the following error:
# {error_msg}""",
            f"""{'\n'.join(self.zero_shot_generator(row).split('\n')[1:])}"""
        )
        

        return prompt
        
    
    def __str__(self) -> str:
        return f"ErrorFixingGeneratorLlama(PromptGenerator)\n\tloader: {self.loader}, \n\tnum_rows: {self.num_rows}, \n\tzero_shot_generator: {self.zero_shot_generator}"

