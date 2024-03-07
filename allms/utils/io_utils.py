import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, OrderedDict

import fsspec

from allms.constants.input_data import IODataConstants
from allms.domain.input_data import InputData

logger = logging.getLogger(__name__)


def load_csv(
        path: str,
        limit: Optional[int] = None
) -> List[OrderedDict[Any, Any]]:
    logger.info(f"Loading test data from {path}")
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)
        return data[:limit] if limit else data


def load_csv_to_input_data(path: str, limit: Optional[int] = None) -> List[InputData]:
    csv_data = load_csv(path, limit=limit)
    return list(
        map(
            lambda row: InputData(input_mappings=drop_dict_key(row, IODataConstants.ID),
                                  id=str(row[IODataConstants.ID])),
            csv_data
        )
    )


def drop_dict_key(dictionary: Dict[Any, Any], key: Any) -> Dict[Any, Any]:
    dict_copy = dictionary.copy()
    dict_copy.pop(key)
    return dict_copy


def load_credentials(path: Union[str, Path]) -> str:
    with fsspec.open(path, "r") as credentials_file:
        return credentials_file.readline()
