from asyncio import AbstractEventLoop
from langchain_google_vertexai import VertexAI
from typing import Optional

from allms.defaults.general_defaults import GeneralDefaults
from allms.defaults.vertex_ai import PalmModelDefaults
from allms.domain.configuration import VertexAIConfiguration
from allms.models.vertexai_base import CustomVertexAI
from allms.models.abstract import AbstractModel


class VertexAIPalmModel(AbstractModel):
    def __init__(
            self,
            config: VertexAIConfiguration,
            temperature: float = PalmModelDefaults.TEMPERATURE,
            top_k: int = PalmModelDefaults.TOP_K,
            top_p: float = PalmModelDefaults.TOP_P,
            max_output_tokens: int = PalmModelDefaults.MAX_OUTPUT_TOKENS,
            model_total_max_tokens: int = PalmModelDefaults.MODEL_TOTAL_MAX_TOKENS,
            max_concurrency: int = GeneralDefaults.MAX_CONCURRENCY,
            max_retries: int = GeneralDefaults.MAX_RETRIES,
            verbose: bool = PalmModelDefaults.VERBOSE,
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

    def _create_llm(self) -> VertexAI:
        return CustomVertexAI(
            model_name=self._config.palm_model_name,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            verbose=self._verbose,
            project=self._config.cloud_project,
            location=self._config.cloud_location
        )