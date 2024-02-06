import asyncio
import json
import typing
from concurrent.futures import ThreadPoolExecutor

from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.llms.azureml_endpoint import AzureMLOnlineEndpoint, ContentFormatterBase
from langchain_community.llms.azureml_endpoint import AzureMLEndpointApiType
from langchain_core.outputs import Generation


class ChatModelContentFormatter(ContentFormatterBase):
    def format_request_payload(
            self,
            prompt: str,
            model_kwargs: typing.Dict,
            api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.realtime
    ) -> bytes:
        prompt = self.escape_special_characters(prompt)
        request_payload = json.dumps(
            {
                "input_data": {
                    "input_string": [
                        {
                            "role": "user",
                            "content": prompt
                        }],
                    "parameters": model_kwargs,
                }
            }
        )
        return str.encode(request_payload)

    def format_response_payload(
            self,
            output: bytes,
            api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.realtime
    ) -> Generation:
        return Generation(text=json.loads(output)["output"])


class AzureMLOnlineEndpointAsync(AzureMLOnlineEndpoint):

    async def _acall(
            self,
            prompt: str,
            stop: typing.Optional[typing.List[str]] = None,
            run_manager: typing.Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: typing.Any,
    ) -> str:
        # Under the hood, langchain uses urllib.request to query the Azure ML Endpoint. urllib.request is not compatible
        # with asyncio, and that's why we had to implement the function this way
        task_executor = ThreadPoolExecutor()
        return await asyncio.wrap_future(
            task_executor.submit(self._call, prompt, stop, run_manager, **kwargs)
        )
