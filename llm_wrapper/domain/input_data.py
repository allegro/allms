import typing

from pydantic import BaseModel


class InputData(BaseModel):
    input_mappings: typing.Dict[str, str]
    id: str

    def get_input_keys(self) -> typing.List[str]:
        return list(self.input_mappings.keys())
