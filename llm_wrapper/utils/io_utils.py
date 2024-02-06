import logging
from pathlib import Path
from typing import List, Optional, Union

import fsspec
import pandas as pd

from llm_wrapper.constants.input_data import IODataConstants
from llm_wrapper.domain.input_data import InputData

logger = logging.getLogger(__name__)


def load_data(
        path: str,
        limit: Optional[int] = None
) -> List[InputData]:
    logger.info(f"Loading test data from {path}")
    input_df = pd.read_csv(path)
    input_df = input_df.head(limit) if limit else input_df
    return load_input_data(input_df)


def load_input_data(input_df: pd.DataFrame) -> List[InputData]:
    return list(
        map(
            lambda row: InputData(input_mappings=row[1].drop(IODataConstants.ID).to_dict(), id=str(row[1].id)),
            input_df.iterrows()
        )
    )


def load_credentials(path: Union[str, Path]) -> str:
    with fsspec.open(path, "r") as credentials_file:
        return credentials_file.readline()
