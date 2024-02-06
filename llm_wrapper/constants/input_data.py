import typing


class IODataConstants:
    TEXT = "text"
    ID = "id"

    PROMPT_TOKENS_NUMBER = "number_of_prompt_tokens"
    GENERATED_TOKENS_NUMBER = "number_of_generated_tokens"

    RESPONSE_STR_NAME = "response"

    ERROR_MESSAGE_STR = "Response error"
    VALUE_ERROR_MESSAGE = "Value Error has occurred"
    INVALID_ARGUMENT_MESSAGE = "Invalid Argument Exception"
    CONTENT_FILTER_MESSAGE = "Content Filter Message"
    TIMEOUT_ERROR_MESSAGE = "Timeout Error"

    SUPPORTED_INPUT_DATA_FORMAT = "csv"
    DEFAULT_ID = "DEFAULT_ID"

    @staticmethod
    def get_columns_for_df_with_responses(input_keys: typing.List[str]) -> typing.List[str]:
        return input_keys + [
            IODataConstants.ID,
            IODataConstants.RESPONSE_STR_NAME,
            IODataConstants.PROMPT_TOKENS_NUMBER,
            IODataConstants.GENERATED_TOKENS_NUMBER
        ]
