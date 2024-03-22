import typing
from asyncio import AbstractEventLoop
from typing import List, Type

from langchain_community.chat_models.azureml_endpoint import LlamaChatContentFormatter
from pydantic import BaseModel

from allms.defaults.azure_defaults import AzureLlama2Defaults
from allms.defaults.general_defaults import GeneralDefaults
from allms.domain.configuration import AzureSelfDeployedConfiguration
from allms.domain.input_data import InputData
from allms.domain.response import ResponseData
from allms.models.abstract import AbstractModel
from allms.models.azure_base import AzureMLOnlineEndpointAsync


class AzureLlama2Model(AbstractModel):

    def __init__(
            self,
            config: AzureSelfDeployedConfiguration,
            temperature: float = AzureLlama2Defaults.TEMPERATURE,
            top_p: float = AzureLlama2Defaults.TOP_P,
            max_output_tokens: int = AzureLlama2Defaults.MAX_OUTPUT_TOKENS,
            model_total_max_tokens: int = AzureLlama2Defaults.MODEL_TOTAL_MAX_TOKENS,
            max_concurrency: int = GeneralDefaults.MAX_CONCURRENCY,
            max_retries: int = GeneralDefaults.MAX_RETRIES,
            event_loop: typing.Optional[AbstractEventLoop] = None
    ) -> None:
        self._top_p = top_p
        self._config = config

        super().__init__(
            temperature=temperature,
            model_total_max_tokens=model_total_max_tokens,
            max_output_tokens=max_output_tokens,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            event_loop=event_loop
        )

        self._is_json_format_injected_into_prompt = False

    def _create_llm(self) -> AzureMLOnlineEndpointAsync:
        model_kwargs = {"max_new_tokens": self._max_output_tokens, "top_p": self._top_p, "do_sample": False}
        if self._temperature > 0:
            model_kwargs["temperature"] = self._temperature
            model_kwargs["do_sample"] = True

        return AzureMLOnlineEndpointAsync(
            endpoint_api_key=self._config.api_key,
            endpoint_url=self._config.endpoint_url,
            model_kwargs=model_kwargs,
            content_formatter=LlamaChatContentFormatter(),
            deployment_name=self._config.deployment
        )
