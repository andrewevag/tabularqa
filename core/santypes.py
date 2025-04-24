from typing import Any, Callable
import pandas as pd
type DatasetRow = dict[str, Any]
type TableLoader = Callable[[str], pd.DataFrame]
