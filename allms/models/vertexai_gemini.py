import logging
import typing
from asyncio import AbstractEventLoop
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from vertexai.preview import tokenization
from vertexai.tokenization._tokenizers import Tokenizer

from allms.defaults.general_defaults import GeneralDefaults
from allms.defaults.vertex_ai import GeminiModelDefaults
from allms.domain.configuration import VertexAIConfiguration
from allms.domain.input_data import InputData
from allms.models.abstract import AbstractModel
from allms.models.vertexai_base import CustomVertexAI
from allms.utils.logger_utils import setup_logger


logger = logging.getLogger(__name__)
setup_logger()

BASE_GEMINI_MODEL_NAMES = [
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # TODO: add `gemini-2.0-flash` when available
]


class VertexAIGeminiModel(AbstractModel):
    def __init__(
            self,
            config: VertexAIConfiguration,
            temperature: float = GeminiModelDefaults.TEMPERATURE,
            top_k: int = GeminiModelDefaults.TOP_K,
            top_p: float = GeminiModelDefaults.TOP_P,
            max_output_tokens: int = GeminiModelDefaults.MAX_OUTPUT_TOKENS,
            model_total_max_tokens: int = GeminiModelDefaults.MODEL_TOTAL_MAX_TOKENS,
            max_concurrency: int = GeneralDefaults.MAX_CONCURRENCY,
            max_retries: int = GeneralDefaults.MAX_RETRIES,
            verbose: bool = GeminiModelDefaults.VERBOSE,
            event_loop: Optional[AbstractEventLoop] = None
    ) -> None:
        self._top_p = top_p
        self._top_k = top_k
        self._verbose = verbose
        self._config = config

        self._gcp_tokenizer = self._get_gcp_tokenizer(self._config.gemini_model_name)

        super().__init__(
            temperature=temperature,
            model_total_max_tokens=model_total_max_tokens,
            max_output_tokens=max_output_tokens,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            event_loop=event_loop
        )

    def _create_llm(self) -> CustomVertexAI:
        llm = CustomVertexAI(
            model_name=self._config.gemini_model_name,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            safety_settings=self._config.gemini_safety_settings,
            verbose=self._verbose,
            project=self._config.cloud_project,
            location=self._config.cloud_location,
            api_endpoint=self._config.api_endpoint,
            api_transport=self._config.api_transport,
            credentials=self._config.credentials,
        )
        # NOTE: this param is for some reason not passed, see: langchain_google_vertexai.llms.VertexAI.validate_environment
        # `endpoint_version` is not passed to the `ChatVertexAI` constructor
        # but in _VertexAIBase, grandparent of VertexAI (VertexAI ->  _VertexAICommon -> _VertexAIBase)
        # it's set v1beta1 by default
        if self._config.endpoint_version:
            llm.client.endpoint_version = self._config.endpoint_version
        # NOTE: `ChatVertexAI` is child of _VertexAICommon and grandchild of _VertexAIBase, the same as `VertexAI`,
        # so they use the same validation in langchain_google_vertexai._base._VertexAIBase.validate_params_base.
        # In `validate_params_base` the `default_metadata` is set to `additional_headers`.
        # And in constructor of `ChatVertexAI` `additional_headers` is not passed.
        # So `default_metadata` is always set to default value.
        if self._config.extra_headers:
            llm.client.default_metadata = self._config.extra_headers
        return llm

    def _get_prompt_tokens_number(self, prompt: ChatPromptTemplate, input_data: InputData) -> int:
        return self._gcp_tokenizer.count_tokens(
            prompt.format_prompt(**input_data.input_mappings).to_string()
        ).total_tokens

    def _get_model_response_tokens_number(self, model_response: typing.Optional[str]) -> int:
        if model_response:
            return self._gcp_tokenizer.count_tokens(model_response).total_tokens
        return 0


    @staticmethod
    def _get_gcp_tokenizer(model_name) -> Tokenizer:
        try:
            return tokenization.get_tokenizer_for_model(model_name)
        except ValueError:
            for base_model_name in BASE_GEMINI_MODEL_NAMES:
                if model_name.startswith(base_model_name):
                    return tokenization.get_tokenizer_for_model(base_model_name)
                else:
                    # Currently supported models for token listing and counting
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/list-token#supported-models
                    # `gemini-2.0` family of models is not supported yet, hence we need this workaround
                    logger.info(
                        f"Model %s is not supported for tokenization, using default tokenizer:"
                        f" {GeminiModelDefaults.GCP_MODEL_NAME}",
                        model_name
                    )
                    return tokenization.get_tokenizer_for_model(GeminiModelDefaults.GCP_MODEL_NAME)
            raise

