import asyncio
import typing
from concurrent.futures import ThreadPoolExecutor

from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint


class AzureMLOnlineEndpointAsync(AzureMLChatOnlineEndpoint):

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
