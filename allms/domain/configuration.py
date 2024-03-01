from dataclasses import dataclass


@dataclass
class AzureOpenAIConfiguration:
    base_url: str
    deployment: str
    model_name: str
    api_version: str
    api_key: str


@dataclass
class AzureSelfDeployedConfiguration:
    api_key: str
    deployment: str
    endpoint_url: str


@dataclass
class VertexAIConfiguration:
    cloud_project: str
    cloud_location: str


@dataclass
class VertexAIModelGardenConfiguration(VertexAIConfiguration):
    endpoint_id: str
