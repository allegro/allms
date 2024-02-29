import typing
from asyncio import AbstractEventLoop

from langchain_community.chat_models.azureml_endpoint import LlamaChatContentFormatter

from allms.defaults.azure_defaults import AzureMistralAIDefaults
from allms.defaults.general_defaults import GeneralDefaults
from allms.domain.configuration import AzureSelfDeployedConfiguration
from allms.models.abstract import AbstractModel
from allms.models.azure_base import AzureMLOnlineEndpointAsync


class AzureMistralModel(AbstractModel):

    def __init__(
            self,
            config: AzureSelfDeployedConfiguration,
            temperature: float = AzureMistralAIDefaults.TEMPERATURE,
            top_p: float = AzureMistralAIDefaults.TOP_P,
            max_output_tokens: int = AzureMistralAIDefaults.MAX_OUTPUT_TOKENS,
            model_total_max_tokens: int = AzureMistralAIDefaults.MODEL_TOTAL_MAX_TOKENS,
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

    def _create_llm(self) -> AzureMLOnlineEndpointAsync:
        model_kwargs = {
            "max_new_tokens": self._max_output_tokens, "top_p": self._top_p, "do_sample": False,
            "return_full_text": False
        }
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
