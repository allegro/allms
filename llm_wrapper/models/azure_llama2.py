import typing
from asyncio import AbstractEventLoop

from llm_wrapper.defaults.azure_defaults import AzureLlama2Defaults
from llm_wrapper.defaults.general_defaults import GeneralDefaults
from llm_wrapper.domain.configuration import AzureSelfDeployedConfiguration
from llm_wrapper.models.abstract import AbstractModel
from llm_wrapper.models.azure_base import AzureMLOnlineEndpointAsync, ChatModelContentFormatter


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

    def _create_llm(self) -> AzureMLOnlineEndpointAsync:
        model_kwargs = {"max_new_tokens": self._max_output_tokens, "top_p": self._top_p, "do_sample": False}
        if self._temperature > 0:
            model_kwargs["temperature"] = self._temperature
            model_kwargs["do_sample"] = True

        return AzureMLOnlineEndpointAsync(
            endpoint_api_key=self._config.api_key,
            endpoint_url=self._config.endpoint_url,
            model_kwargs=model_kwargs,
            content_formatter=ChatModelContentFormatter(),
            deployment_name=self._config.deployment
        )
