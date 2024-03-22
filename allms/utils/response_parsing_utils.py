import re
import typing

from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

from allms.domain.response import ResponseData, ResponseParsingOutput


class ResponseParser:
    def __init__(self, parser: PydanticOutputParser) -> None:
        self._json_pattern = re.compile(r"{.*?}", re.DOTALL)
        self._parser = parser

    def _clean_extracted_json(self, extracted_json: str) -> str:
        json_without_newlines = extracted_json.replace("\\n", "")
        json_without_backslashes = json_without_newlines.replace("\\", "")

        return json_without_backslashes

    def _extract_json_from_response(self, model_response_data: ResponseData) -> str:
        search_results = self._json_pattern.findall(model_response_data.response)
        
        if len(search_results) == 0:
            return model_response_data.response

        return self._clean_extracted_json(search_results[0])
        
    def _parse_response(
        self, 
        model_response_data: ResponseData
    ) -> ResponseParsingOutput:
        raw_response = self._extract_json_from_response(model_response_data)

        try:
            return ResponseParsingOutput(
                response=self._parser.parse(raw_response), 
                error_message=None
            )
        except OutputParserException as output_parser_exception:
            return ResponseParsingOutput(
                response=None, 
                error_message=f"""
                    An OutputParserException has occurred for the model response: {raw_response}
                    The exception message: {output_parser_exception}
                    """
            )
        
    def parse_model_output(
        self, 
        model_responses_data: typing.List[ResponseData]
    ) -> typing.List[ResponseData]:
        parsed_responses = []

        for model_response_data in model_responses_data:
            if not model_response_data.error:
                response_with_error = self._parse_response(model_response_data)

                parsed_responses.append(ResponseData(
                    input_data=model_response_data.input_data,
                    response=response_with_error.response,
                    error=response_with_error.error_message,
                    number_of_prompt_tokens=model_response_data.number_of_prompt_tokens,
                    number_of_generated_tokens=model_response_data.number_of_generated_tokens

                ))
            else:
                parsed_responses.append(model_response_data)

        return parsed_responses