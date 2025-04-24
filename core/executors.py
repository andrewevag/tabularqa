import pandas as pd
import multiprocessing
#import pandasql as ps
import numpy as np
import ast
from abc import ABC, abstractmethod
from typing import Any
from core.santypes import TableLoader


class Executor(ABC):
    """
        Abstract class for executors

        Parameters:
            loader (callable[[str], pd.DataFrame]): The loader function to load the dataset

        Example:
        ```python
            executor = Executor(loader)
            result = executor(statement, dataset_name)
        ```
    """
    def __init__(self, loader : TableLoader):
        self.loader = loader

    @abstractmethod
    def __call__(self, inp: tuple[str, str]) -> Any:
        pass

class SaferMultiLineStatementExecutor(Executor):
    def __init__(self, loader : TableLoader, timeout: int=60):
        super().__init__(loader)
        self.timeout = timeout
        

    def __call__(self, inp: tuple[str, str]) -> Any:
        statement, dataset = inp
        lead = """

def answer(df):
"""
        total_str = "global ans\n"+ lead + statement + f"\nans = answer(df)"

        try:
            df = self.loader(dataset)
            
            def exec_code(queue):
                try:
                    local_namespace = {}
                    global_namespace = {'df': df, 'ans': None, 'pd': pd, 'np': np, 'ast': ast}

                    exec(total_str, global_namespace, local_namespace)

                    ans = global_namespace.get('ans', None)

                    queue.put(ans)
                except Exception as e:
                    queue.put(f"__CODE_ERROR__: {e}")
    
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=exec_code, args=(queue,))
            p.start()
            p.join(self.timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                return f"__TIMEOUT__"
            
            if not queue.empty():
                ans = queue.get()
            else:
                ans = "__CODE_ERROR__: No result returned"

            # return str(ans).split("\n")[0] if "\n" in str(ans) else ans
            return ans
        except Exception as e:
            return f"__CODE_ERROR__: {e}"
        
    def __str__(self) -> str:
        s = f'SaferMultiLineStatementExecutor(Executor)\n\tloader: {self.loader}\n\ttimeout: {self.timeout}\n'
        return s
    

class SaferMultiLineStatementExecutorListPassing(Executor):
    def __init__(self, loader : TableLoader, timeout: int = 60):
        super().__init__(loader)
        self.timeout = timeout

    def __call__(self, inp: tuple[str, str]) -> Any:
        statement, dataset = inp
        lead = """

def answer(df):
"""
        total_str = "global ans\n" + lead + statement + f"\nans = answer(df)"
        
        try:
            df = self.loader(dataset)
            
            def exec_code(queue):
                try:
                    local_namespace = {}
                    global_namespace = {'df': df, 'ans': None, 'pd': pd, 'np': np, 'ast': ast}

                    exec(total_str, global_namespace, local_namespace)

                    ans = global_namespace.get('ans', None)
                    
                    if isinstance(ans, list):
                        queue.put("__LIST_START__")  # Indicator for list
                        for item in ans:
                            queue.put(item)
                        queue.put("__LIST_END__")  # Indicator for list end
                    else:
                        queue.put(ans)
                except Exception as e:
                    queue.put(f"__CODE_ERROR__: {e}")
    
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=exec_code, args=(queue,))
            p.start()
            
            result = []
            while p.is_alive() or not queue.empty():
                if not queue.empty():
                    item = queue.get()
                    if item == "__LIST_START__":
                        temp_list = []
                        while True:
                            element = queue.get()
                            if element == "__LIST_END__":
                                break
                            temp_list.append(element)
                        return temp_list  # Return as a full list
                    elif isinstance(item, str) and item.startswith("__CODE_ERROR__"):
                        return item
                    else:
                        result.append(item)
            
            p.join(self.timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                return "__TIMEOUT__"
            
            return result[0] if result else "__CODE_ERROR__: No result returned"
        
        except Exception as e:
            return f"__CODE_ERROR__: {e}"
    
    def __str__(self) -> str:
        return (f'SaferMultiLineStatementExecutorListPassing(Executor)\n'
                f'\tloader: {self.loader}\n\ttimeout: {self.timeout}\n')

