from typing import Type

from allms.domain.enumerables import AvailableModels
from allms.models.abstract import AbstractModel
from allms.models.azure_llama2 import AzureLlama2Model
from allms.models.azure_mistral import AzureMistralModel
from allms.models.azure_openai import AzureOpenAIModel
from allms.models.vertexai_gemini import VertexAIGeminiModel
from allms.models.vertexai_palm import VertexAIPalmModel

__all__ = [
    "AzureOpenAIModel",
    "AzureLlama2Model",
    "AzureMistralModel",
    "VertexAIPalmModel",
    "VertexAIGeminiModel",
    "get_available_models"
]


def get_available_models() -> dict[str, Type[AbstractModel]]:
    return {
        AvailableModels.AZURE_OPENAI_MODEL: AzureOpenAIModel,
        AvailableModels.AZURE_LLAMA2_MODEL: AzureLlama2Model,
        AvailableModels.AZURE_MISTRAL_MODEL: AzureMistralModel,
        AvailableModels.VERTEXAI_PALM2_MODEL: VertexAIPalmModel,
        AvailableModels.VERTEXAI_GEMINI_MODEL: VertexAIGeminiModel
    }

