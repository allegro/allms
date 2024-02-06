from typing import List, Optional, Any

from langchain_community.llms.vertexai import VertexAI
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from pydash import chain

from llm_wrapper.constants.vertex_ai import VertexModelConstants

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
