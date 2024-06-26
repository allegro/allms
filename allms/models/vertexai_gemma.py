from asyncio import AbstractEventLoop

from langchain_google_vertexai import VertexAIModelGarden
from typing import Optional

from allms.defaults.general_defaults import GeneralDefaults
from allms.defaults.vertex_ai import GemmaModelDefaults
from allms.domain.configuration import VertexAIModelGardenConfiguration
from allms.models.vertexai_base import VertexAIModelGardenWrapper
from allms.models.abstract import AbstractModel


class VertexAIGemmaModel(AbstractModel):
    def __init__(
            self,
            config: VertexAIModelGardenConfiguration,
            temperature: float = GemmaModelDefaults.TEMPERATURE,
            top_k: int = GemmaModelDefaults.TOP_K,
            top_p: float = GemmaModelDefaults.TOP_P,
            max_output_tokens: int = GemmaModelDefaults.MAX_OUTPUT_TOKENS,
            model_total_max_tokens: int = GemmaModelDefaults.MODEL_TOTAL_MAX_TOKENS,
            max_concurrency: int = GeneralDefaults.MAX_CONCURRENCY,
            max_retries: int = GeneralDefaults.MAX_RETRIES,
            verbose: bool = GemmaModelDefaults.VERBOSE,
            event_loop: Optional[AbstractEventLoop] = None
    ) -> None:
        self._top_p = top_p
        self._top_k = top_k
        self._verbose = verbose
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

    def _create_llm(self) -> VertexAIModelGarden:
        return VertexAIModelGardenWrapper(
            model_name=GemmaModelDefaults.GCP_MODEL_NAME,
            max_tokens=self._max_output_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            verbose=self._verbose,
            project=self._config.cloud_project,
            location=self._config.cloud_location,
            endpoint_id=self._config.endpoint_id
        )
