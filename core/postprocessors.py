from pandas.core.computation.ops import Op
from abc import ABC, abstractmethod
from typing import Callable, Optional
from core.santypes import TableLoader

from core.santypes import TableLoader

class PostProcessor(ABC):
    def __init__(self, loader : TableLoader):
        self.loader = loader
    
    @abstractmethod
    def __call__(self, response: str, dataset: Optional[str]):
        pass

class TillReturnLinePostProcessorMultipleIndents(PostProcessor):
    def __init__(self, loader : TableLoader, prefix: int=4, first_prefix : str='', return_indent : int=4):
        super().__init__(loader)
        self.prefix = prefix
        self.first_prefix = first_prefix
        self.return_indent = return_indent

    def __call__(self, response: str, dataset : Optional[str]=None) -> str:
        lines = response.split("\n")
        xs = []; indents = []
        for i, line in enumerate(lines):
            indent = len(line) - len(line.lstrip())
            indents.append(indent)
            xs.append(line.strip())
            if line.startswith(((' ' * self.return_indent) + "return")):
                break

        indents = list(map(lambda x: x - self.prefix, indents))
        lines = list(map(lambda x: (self.prefix * ' ') + (x[1] * ' ') +  x[0], zip(xs, indents)))
        lines[0] = self.first_prefix + lines[0].strip()
        return "\n".join(lines[:i+1])
    
    def __str__(self) -> str:
        return f"TillReturnLinePostProcessorMultipleIndents(PostProcessor)\n\tloader: {self.loader}"



