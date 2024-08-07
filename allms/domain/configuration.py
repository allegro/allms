from dataclasses import dataclass
from typing import Dict, Optional

from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

from allms.defaults.vertex_ai import GeminiModelDefaults, PalmModelDefaults


@dataclass
class AzureOpenAIConfiguration:
    base_url: str
    deployment: str
    model_name: str
    api_version: str
    api_key: Optional[str] = None
    azure_ad_token: Optional[str] = None


@dataclass
class AzureSelfDeployedConfiguration:
    api_key: str
    deployment: str
    endpoint_url: str


@dataclass
class VertexAIConfiguration:
    cloud_project: str
    cloud_location: str
    palm_model_name: Optional[str] = PalmModelDefaults.GCP_MODEL_NAME
    gemini_model_name: Optional[str] = GeminiModelDefaults.GCP_MODEL_NAME
    gemini_safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None


class VertexAIModelGardenConfiguration(VertexAIConfiguration):
    def __init__(
        self,
        cloud_project: str,
        cloud_location: str,
        endpoint_id: str
    ):
        super().__init__(
            cloud_project=cloud_project,
            cloud_location=cloud_location,
            palm_model_name=None,
            gemini_model_name=None
        )
        self.endpoint_id = endpoint_id
