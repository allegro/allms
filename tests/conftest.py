import asyncio
import typing
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from langchain_community.llms.fake import FakeListLLM

from allms.domain.configuration import (
    AzureOpenAIConfiguration, AzureSelfDeployedConfiguration, VertexAIConfiguration, VertexAIModelGardenConfiguration)
from allms.models import AzureOpenAIModel, VertexAIPalmModel, AzureLlama2Model
from allms.models.azure_mistral import AzureMistralModel
from allms.models.vertexai_gemini import VertexAIGeminiModel
from allms.models.vertexai_gemma import VertexAIGemmaModel


class AzureOpenAIEnv:
    OPENAI_API_BASE: str = "https://dummy-endpoint.openai.azure.com/"
    OPENAI_API_VERSION: str = "dummy-api-version"
    OPENAI_DEPLOYMENT_NAME: str = "dummy-deployment-name"


@dataclass
class GenerativeModels:
    azure_gpt: typing.Optional[AzureOpenAIModel] = None
    vertex_palm: typing.Optional[VertexAIPalmModel] = None


class ModelWithoutAsyncRequestsMock(FakeListLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(responses=["{}"])


@pytest.fixture(scope="function")
def models():
    event_loop = asyncio.new_event_loop()

    with ExitStack() as stack:
        stack.enter_context(patch("allms.models.vertexai_palm.CustomVertexAI", ModelWithoutAsyncRequestsMock))
        stack.enter_context(patch("allms.models.vertexai_gemini.CustomVertexAI", ModelWithoutAsyncRequestsMock))
        stack.enter_context(patch("allms.models.vertexai_gemma.VertexAIModelGardenWrapper", ModelWithoutAsyncRequestsMock))
        stack.enter_context(patch("allms.models.azure_llama2.AzureMLOnlineEndpointAsync", ModelWithoutAsyncRequestsMock))
        stack.enter_context(patch("allms.models.azure_mistral.AzureMLOnlineEndpointAsync", ModelWithoutAsyncRequestsMock))

        return {
                "azure_open_ai": AzureOpenAIModel(
                    config=AzureOpenAIConfiguration(
                        api_key="dummy_api_key",
                        base_url=AzureOpenAIEnv.OPENAI_API_BASE,
                        api_version=AzureOpenAIEnv.OPENAI_API_VERSION,
                        deployment=AzureOpenAIEnv.OPENAI_DEPLOYMENT_NAME,
                        model_name="gpt-4"
                    ),
                    event_loop=event_loop
                ),
                "vertex_palm": VertexAIPalmModel(
                    config=VertexAIConfiguration(
                        cloud_project="dummy-project-id",
                        cloud_location="us-central1"
                    ),
                    event_loop=event_loop
                ),
                "vertex_gemini": VertexAIGeminiModel(
                    config=VertexAIConfiguration(
                        cloud_project="dummy-project-id",
                        cloud_location="us-central1"
                    ),
                    event_loop=event_loop
                ),
                "vertex_gemma": VertexAIGemmaModel(
                    config=VertexAIModelGardenConfiguration(
                        cloud_project="dummy-project-id",
                        cloud_location="us-central1",
                        endpoint_id="dummy-endpoint-id"
                    ),
                    event_loop=event_loop
                ),
                "azure_llama2": AzureLlama2Model(
                    config=AzureSelfDeployedConfiguration(
                        api_key="dummy_api_key",
                        endpoint_url="https://dummy-endpoint.dummy-region.inference.ml.azure.com/score",
                        deployment="dummy_deployment_name"
                    ),
                    event_loop=event_loop
                ),
                "azure_mistral": AzureMistralModel(
                    config=AzureSelfDeployedConfiguration(
                        api_key="dummy_api_key",
                        endpoint_url="https://dummy-endpoint.dummy-region.inference.ml.azure.com/score",
                        deployment="dummy_deployment_name"
                    ),
                    event_loop=event_loop
                )
            }
