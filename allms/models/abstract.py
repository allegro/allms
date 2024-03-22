import asyncio
import logging
import re
import typing
import urllib
from abc import ABC, abstractmethod
from functools import partial

import google
import openai
from google.api_core.exceptions import InvalidArgument
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel

import allms.exceptions.validation_input_data_exceptions as input_exception_message
import allms.models as models
from allms.chains.long_text_processing_chain import (
    LongTextProcessingChain,
    load_long_text_processing_chain
)
from allms.constants.input_data import IODataConstants
from allms.constants.prompt import PromptConstants
from allms.defaults.general_defaults import GeneralDefaults
from allms.defaults.long_text_chain import LongTextChainDefaults
from allms.domain.enumerables import AggregationLogicForLongInputData, LanguageModelTask
from allms.domain.input_data import InputData
from allms.domain.prompt_dto import SummaryOutputClass, KeywordsOutputClass
from allms.domain.response import ResponseData
from allms.utils.long_text_processing_utils import get_max_allowed_number_of_tokens
from allms.utils.response_parsing_utils import ResponseParser

logger = logging.getLogger(__name__)


class AbstractModel(ABC):
    def __init__(
            self,
            temperature: float,
            max_output_tokens: int,
            model_total_max_tokens: int,
            event_loop: typing.Optional[asyncio.AbstractEventLoop] = None,
            max_concurrency: int = GeneralDefaults.MAX_CONCURRENCY,
            max_retries: int = GeneralDefaults.MAX_RETRIES
    ):
        self._model_total_max_tokens = model_total_max_tokens
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # TODO: To be changed after implementing support for long sequences
        self._task = LanguageModelTask.KEYWORDS
        self._is_long_text_bypass_enabled: bool = False  # Should be false till we fully implement support for long sequences in our package
        self._aggregation_strategy: AggregationLogicForLongInputData = AggregationLogicForLongInputData.SIMPLE_CONCATENATION
        self._parser: typing.Optional[PydanticOutputParser] = None
        self._json_pattern = re.compile(r"{.*?}", re.DOTALL)
        self._is_json_format_injected_into_prompt: bool = True

        if max_output_tokens >= model_total_max_tokens:
            raise ValueError("max_output_tokens has to be lower than model_total_max_tokens")

        self._llm = self._create_llm()
        self._event_loop = event_loop if event_loop is not None else asyncio.get_event_loop()

        self._predict_example = create_base_retry_decorator(
            error_types=[
                openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout,
                openai.error.APIConnectionError, openai.error.ServiceUnavailableError,
                google.api_core.exceptions.ResourceExhausted, urllib.error.HTTPError
            ],
            max_retries=max_retries,
        )(self._predict_example)

    @abstractmethod
    def _create_llm(self) -> BaseChatModel:
        ...

    def _get_prompt_tokens_number(self, prompt: ChatPromptTemplate, input_data: InputData) -> int:
        return self._llm.get_num_tokens(prompt.format_prompt(**input_data.input_mappings).to_string())

    def _get_model_response_tokens_number(self, model_response: typing.Optional[str]) -> int:
        if model_response:
            return self._llm.get_num_tokens(model_response)
        return 0

    def generate(
            self,
            prompt: str,
            input_data: typing.Optional[typing.List[InputData]] = None,
            output_data_model_class: typing.Optional[typing.Type[BaseModel]] = None,
            system_prompt: typing.Optional[str] = None
    ) -> typing.List[ResponseData]:
        model_responses = self._event_loop.run_until_complete(
            self._generate(
                prompt=prompt,
                input_data=input_data,
                output_data_model_class=output_data_model_class,
                system_prompt=system_prompt
            )
        )

        if output_data_model_class:
            return ResponseParser(self._parser).parse_model_output(model_responses)
        return model_responses

    async def _generate(
            self,
            prompt: str,
            input_data: typing.Optional[typing.List[InputData]] = None,
            output_data_model_class: typing.Optional[typing.Type[BaseModel]] = None,
            system_prompt: typing.Optional[str] = None
    ) -> typing.List[ResponseData]:
        self._validate_system_prompt(system_prompt=system_prompt)
        self._validate_input(prompt=prompt, input_data=input_data)
        if input_data is None:
            # Prompt without symbolic variables is passed - create input_data accordingly
            input_data = [InputData(input_mappings={}, id=IODataConstants.DEFAULT_ID)]

        prompt_template_args = {
            PromptConstants.TEMPLATE_STR: prompt,
            PromptConstants.INPUT_VARIABLES_STR: list(input_data[0].get_input_keys())
        }

        if output_data_model_class:
            self._parser = PydanticOutputParser(pydantic_object=output_data_model_class)

            if self._is_json_format_injected_into_prompt:
                prompt_template_args[PromptConstants.PARTIAL_VARIABLES_STR] = {
                    PromptConstants.OUTPUT_DATA_MODEL: self._parser.get_format_instructions(),
                }
                prompt_template_args[PromptConstants.TEMPLATE_STR] = self._add_output_data_format(prompt=prompt)

        chat_prompts = await self._build_chat_prompts(prompt_template_args, system_prompt)

        prompt_template = ChatPromptTemplate.from_messages(chat_prompts)

        chain = self._get_chain(prompt_template)
        long_chain = self._get_chain_for_long_text(prompt_template)

        predict_example_any_length_partial = partial(
            self._predict_example_of_any_length,
            prompt_template=prompt_template,
            standard_chain=chain,
            long_chain=long_chain
        )

        logger.info("Generating responses...")
        results = list(map(lambda data: predict_example_any_length_partial(input_data=data), input_data))

        responses = await asyncio.gather(*results)

        return responses

    async def _build_chat_prompts(
            self,
            prompt_template_args: dict,
            system_prompt: SystemMessagePromptTemplate
    ) -> list[SystemMessagePromptTemplate | HumanMessagePromptTemplate]:
        human_message = HumanMessagePromptTemplate(prompt=PromptTemplate(**prompt_template_args))
        if not system_prompt:
            return [human_message]
        system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)

        return [system_message_template, human_message]

    @staticmethod
    def _add_output_data_format(prompt: str) -> str:
        return f"{prompt}{PromptConstants.OUTPUT_DATA_MODEL_CLASS_SEPARATOR}{{{PromptConstants.OUTPUT_DATA_MODEL}}}"

    def _validate_input_data_len(
            self,
            input_data: InputData,
            number_of_prompt_tokens: int
    ):
        if number_of_prompt_tokens > self._model_total_max_tokens:
            raise ValueError(
                f"Prompt is too long. Entire prompt has {number_of_prompt_tokens} tokens, where the max allowed number "
                f"of tokens of the model is {self._model_total_max_tokens}. This leaves no space for the model to "
                f"generate a response and will lead to errors. Example id: {input_data.id}"
            )
        elif number_of_prompt_tokens + self._max_output_tokens > self._model_total_max_tokens:
            logger.warning(
                f"Number of prompt tokens plus generated tokens may exceed the the max allowed number of tokens of the "
                f"model. Entire prompt has {number_of_prompt_tokens} tokens, the max number of tokens to generate is "
                f"{self._max_output_tokens} and the max allowed number of tokens of the model is "
                f"{self._model_total_max_tokens}. Consider lowering the max_output_tokens param or truncating the "
                f"input, because otherwise it may lead to unexpected errors. Example id: {input_data.id}"
            )

    def _predict_example_of_any_length(
            self,
            input_data: InputData,
            prompt_template: ChatPromptTemplate,
            standard_chain: LLMChain,
            long_chain: LLMChain
    ) -> ResponseData:
        number_of_prompt_tokens = self._get_prompt_tokens_number(
            prompt=prompt_template,
            input_data=input_data
        )
        max_token_limit = get_max_allowed_number_of_tokens(self._model_total_max_tokens, self._max_output_tokens)
        is_example_too_long = number_of_prompt_tokens > max_token_limit

        predict_example_partial = partial(
            self._predict_example,
            input_data=input_data,
            prompt_tokens_number=number_of_prompt_tokens
        )
        if is_example_too_long and self._is_long_text_bypass_enabled:
            return predict_example_partial(chain=long_chain)
        return predict_example_partial(chain=standard_chain)

    async def _predict_example(
            self,
            chain: LLMChain,
            input_data: InputData,
            prompt_tokens_number: int
    ) -> ResponseData:
        error_message: typing.Optional[str] = None
        number_of_input_mappings = len(input_data.input_mappings)

        try:
            self._validate_input_data_len(input_data=input_data, number_of_prompt_tokens=prompt_tokens_number)
        except ValueError as value_error:
            logger.info(f"Error for id {input_data.id} has occurred. Message: {value_error} ")
            error_message = f"{IODataConstants.VALUE_ERROR_MESSAGE}: {value_error}"
            return ResponseData(
                input_data=None if number_of_input_mappings == 0 else input_data,
                response=None,
                number_of_prompt_tokens=prompt_tokens_number,
                number_of_generated_tokens=0,
                error=error_message
            )

        try:
            async with self._semaphore:
                if number_of_input_mappings == 0:
                    # Workaround when prompt without symbolic variables is passed - arun() can't be called without any arg
                    model_response = await chain.arun({})
                else:
                    model_response = await chain.arun(**input_data.input_mappings)
        except openai.error.InvalidRequestError as invalid_request_error:
            logger.info(f"Error for id {input_data.id} has occurred. Message: {invalid_request_error} ")
            if invalid_request_error.error.code == "content_filter":
                model_response = None
                error_message = f"{IODataConstants.CONTENT_FILTER_MESSAGE}: {invalid_request_error}"
            else:
                model_response = None
                error_message = f"{IODataConstants.ERROR_MESSAGE_STR}: {invalid_request_error}"

        except (InvalidArgument, ValueError, TimeoutError, openai.error.Timeout) as other_error:
            model_response = None
            logger.info(f"Error for id {input_data.id} has occurred. Message: {other_error} ")
            error_message = f"{type(other_error).__name__}: {other_error}"

        return ResponseData(
            input_data=None if number_of_input_mappings == 0 else input_data,
            response=model_response,
            number_of_prompt_tokens=prompt_tokens_number,
            number_of_generated_tokens=self._get_model_response_tokens_number(model_response),
            error=error_message
        )

    def _get_number_of_tokens_in_prompt(self, prompt: PromptTemplate, input_data: InputData) -> int:
        return self._llm.get_num_tokens(prompt.format_prompt(**input_data.input_mappings).to_string())

    def _get_chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self._llm,
            prompt=prompt,
        )

    # TODO: When adding support for long documents, we'll need to rethink how output_data_model will be passed to the
    # TODO: aggregation prompt
    def _get_chain_for_long_text(
            self,
            prompt_template: PromptTemplate,
    ) -> LongTextProcessingChain:
        parser = PydanticOutputParser(
            pydantic_object=SummaryOutputClass if self._task == LanguageModelTask.SUMMARY else KeywordsOutputClass)
        reduce_prompt_template = PromptTemplate(
            template=LongTextChainDefaults.AGGREGATION_PROMPT,
            input_variables=["text"],
            partial_variables={PromptConstants.OUTPUT_DATA_MODEL: parser.get_format_instructions()}
        )

        return load_long_text_processing_chain(
            task=self._task,
            llm=self._llm,
            model_total_max_tokens=self._model_total_max_tokens,
            max_output_tokens=self._max_output_tokens,
            map_prompt=prompt_template,
            reduce_prompt=reduce_prompt_template,
            aggregation_strategy=self._aggregation_strategy
        )

    def _validate_input(self, prompt: str, input_data: typing.Optional[typing.List[InputData]] = None) -> None:
        # Extracts text inside the {} but escapes the text inside {{}}
        # This behaviour allows to pass JSON-like strings to the prompt
        # reference: https://github.com/langchain-ai/langchain/issues/1660#issuecomment-1469320129
        prompt_input_variables_set = AbstractModel._extract_input_variables_from_prompt(prompt)
        if PromptConstants.OUTPUT_DATA_MODEL in prompt_input_variables_set:
            prompt_input_variables_set.remove(PromptConstants.OUTPUT_DATA_MODEL)

        if input_data:
            for data in input_data:
                self._validate_input_data(prompt_input_variables_set, data)
        elif len(prompt_input_variables_set) > 0:
            raise ValueError(
                input_exception_message.get_prompt_contains_input_key_when_missing_input_data())

    def _validate_system_prompt(self, system_prompt: typing.Optional[str] = None) -> None:
        if isinstance(self, models.AzureMistralModel) and system_prompt is not None:
            raise ValueError(input_exception_message.get_system_prompt_is_not_supported_by_model())
        elif system_prompt:
            prompt_input_variables_set = AbstractModel._extract_input_variables_from_prompt(system_prompt)
            if prompt_input_variables_set:
                raise ValueError(input_exception_message.get_system_prompt_contains_input_variables())

    @staticmethod
    def _extract_input_variables_from_prompt(prompt: str) -> set[str]:
        input_variables_pattern = r'(?<!\{)\{([^{}]+)\}(?!\})'
        input_variables_set = set(re.findall(input_variables_pattern, prompt))
        return input_variables_set

    @staticmethod
    def _validate_input_data(
            prompt_input_variables: typing.Set[str],
            input_data: typing.Optional[InputData] = None
    ) -> None:
        if len(input_data.input_mappings.keys()) > 0 and len(prompt_input_variables) == 0:
            raise ValueError(input_exception_message.get_missing_input_data_in_prompt_message(input_data.id))

        if len(input_data.input_mappings.keys()) == 0 and len(prompt_input_variables) > 0:
            raise ValueError(input_exception_message.get_missing_input_data_in_input_data_message(input_data.id))

        if len(input_data.input_mappings.keys()) != len(prompt_input_variables):
            raise ValueError(input_exception_message.get_different_number_of_inputs_message(input_data.id))

        if not prompt_input_variables == set(input_data.get_input_keys()):
            raise ValueError(input_exception_message.get_different_input_keys_message(input_data.id))
