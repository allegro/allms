from typing import Type

from allms.models.abstract import AbstractModel
from allms.models.azure_llama2 import AzureLlama2Model
from allms.models.azure_mistral import AzureMistralModel
from allms.models.azure_openai import AzureOpenAIModel
from allms.models.vertexai_gemini import VertexAIGeminiModel
from allms.models.vertexai_palm import VertexAIPalmModel

__all__ = ["AzureOpenAIModel", "AzureLlama2Model", "AzureMistralModel", "VertexAIPalmModel", "VertexAIGeminiModel"]


def get_available_models() -> dict[str, Type[AbstractModel]]:
    return {
        "azure_openai": AzureOpenAIModel,
        "azure_llama2": AzureLlama2Model,
        "azure_mistral": AzureMistralModel,
        "vertexai_palm2": VertexAIPalmModel,
        "vertexai_gemini": VertexAIGeminiModel
    }

