from typing import List, Optional, Any, Dict

from google.cloud.aiplatform.models import Prediction
from langchain_community.llms.vertexai import VertexAI, VertexAIModelGarden
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from pydash import chain

from allms.constants.vertex_ai import VertexModelConstants


class CustomVertexAI(VertexAI):
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        def was_response_blocked(generation: Generation) -> bool:
            return (
                generation.generation_info is not None 
                    and 'is_blocked' in generation.generation_info 
                    and generation.generation_info['is_blocked']
            )

        result = await super()._agenerate(
            prompts=prompts,
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )

        return LLMResult(
            generations=(
                chain(result.generations)
                .map(lambda generation_candidates: (
                    chain(generation_candidates)
                    .map(
                        lambda single_candidate: Generation(
                            text=VertexModelConstants.RESPONSE_BLOCKED_STR
                        ) if was_response_blocked(single_candidate) else single_candidate
                    )
                    .value()
                ))
                .value()
            ),
            llm_output=result.llm_output,
            run=result.run
        )


class VertexAIModelGardenWrapper(VertexAIModelGarden):
    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 0.95
    top_k: int = 40
    n: int = 1

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.allowed_model_args = list(self._default_params.keys())

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "n": self.n
        }

    def _parse_response(self, predictions: "Prediction", prompts: List[str]) -> LLMResult:
        generations: List[List[Generation]] = []
        for result, prompt in zip(predictions.predictions, prompts):
            if isinstance(result, str):
                generations.append([Generation(text=self._parse_prediction(result, prompt))])
            else:
                generations.append(
                    [
                        Generation(text=self._parse_prediction(prediction, prompt))
                        for prediction in result
                    ]
                )
        return LLMResult(generations=generations)

    def _parse_prediction(self, prediction: Any, prompt: str) -> str:
        parsed_prediction = super()._parse_prediction(prediction)
        try:
            text_to_remove = f"Prompt:\n{prompt}\nOutput:\n"
            return parsed_prediction.rsplit(text_to_remove, maxsplit=1)[1]
        except Exception:
            raise ValueError(f"Output returned from the model doesn't follow the expected format.")

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        kwargs = {**kwargs, **self._default_params}
        instances = self._prepare_request(prompts, **kwargs)
        response = await self.async_client.predict(
            endpoint=self.endpoint_path, instances=instances
        )
        return self._parse_response(response, prompts)

