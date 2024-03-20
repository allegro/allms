import typing

from pydantic import BaseModel

from allms.domain.input_data import InputData


class ResponseParsingOutput(BaseModel):
    response: typing.Optional[typing.Any]
    error_message: typing.Optional[str]


class ResponseData(BaseModel):
    response: typing.Optional[typing.Any] = None
    input_data: typing.Optional[InputData] = None

    number_of_prompt_tokens: typing.Optional[int] = None
    number_of_generated_tokens: typing.Optional[int] = None
    error: typing.Optional[str] = None

    # Without this, only classes inheriting from the pydantic BaseModel are allowed as field types. Exception isn't
    # such a class and that's why we need it.
    class Config:
        arbitrary_types_allowed = True
