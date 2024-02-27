from asyncio import AbstractEventLoop
from typing import Optional

from langchain.chat_models import AzureChatOpenAI

from allms.defaults.azure_defaults import AzureGptTurboDefaults
from allms.defaults.general_defaults import GeneralDefaults
from allms.domain.configuration import AzureOpenAIConfiguration
from allms.models.abstract import AbstractModel


class AzureOpenAIModel(AbstractModel):
    def __init__(
            self,
            config: AzureOpenAIConfiguration,
            temperature: float = AzureGptTurboDefaults.TEMPERATURE,
            max_output_tokens: int = AzureGptTurboDefaults.MAX_OUTPUT_TOKENS,
            request_timeout_s: int = AzureGptTurboDefaults.REQUEST_TIMEOUT_S,
            model_total_max_tokens: int = AzureGptTurboDefaults.MODEL_TOTAL_MAX_TOKENS,
            max_concurrency: int = GeneralDefaults.MAX_CONCURRENCY,
            max_retries: int = GeneralDefaults.MAX_RETRIES,
            event_loop: Optional[AbstractEventLoop] = None
    ) -> None:
        self._request_timeout_s = request_timeout_s
        self._config = config

        super().__init__(
            temperature=temperature,
            model_total_max_tokens=model_total_max_tokens,
            max_output_tokens=max_output_tokens,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            event_loop=event_loop
        )

    def _create_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            deployment_name=self._config.deployment,
            api_version=self._config.api_version,
            model_name=self._config.model_name,
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            request_timeout=self._request_timeout_s
        )
