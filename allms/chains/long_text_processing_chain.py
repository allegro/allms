import asyncio
from functools import reduce
from typing import List, Any, Tuple, Optional, Union

from langchain import LLMChain, BasePromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.schema import Document

from allms.domain.enumerables import AggregationLogicForLongInputData, LanguageModelTask
from allms.domain.input_data import InputData
from allms.domain.prompt_dto import (AggregateOutputClass, KeywordsOutputClass, SummaryOutputClass)
from allms.utils.long_text_processing_utils import split_text_to_max_size


class LongTextProcessingChain(BaseCombineDocumentsChain):
    task: LanguageModelTask
    model_total_max_tokens: int
    max_output_tokens: int
    map_llm_chain: LLMChain
    reduce_llm_chain: LLMChain
    input_data_variable_name: str
    aggregation_strategy: AggregationLogicForLongInputData

    @property
    def _chain_type(self) -> str:
        return "long_description_chain"

    async def combine_docs(self, input_data: Document, **kwargs: Any) -> Tuple[str, dict]:
        chunked_input: List[Document] = split_text_to_max_size(
            llm=self.map_llm_chain.llm,
            prompt_template=self.map_llm_chain.prompt,
            text=input_data,
            model_total_max_tokens=self.model_total_max_tokens,
            max_output_tokens=self.max_output_tokens
        )

        chunk_responses = await self._map_step(chunked_input)
        aggregated_response = self._reduce_step(chunk_responses)

        return aggregated_response, {}

    async def acombine_docs(self, input_data: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        return await self.combine_docs(input_data)

    async def _map_step(self, chunked_document: List[Document]) -> List[str]:
        results = list(map(lambda document: self.map_llm_chain.arun(document), chunked_document))
        return await asyncio.gather(*results)

    def _reduce_step(self, chunk_responses: List[InputData]) -> str:
        if self.aggregation_strategy == AggregationLogicForLongInputData.REDUCE_BY_LLM_PROMPTING:
            return self._construct_input_from_list_and_run_reduce_chain(chunk_responses)
        elif self.aggregation_strategy == AggregationLogicForLongInputData.SIMPLE_CONCATENATION:
            if self.task == LanguageModelTask.SUMMARY:
                return self._aggregate_results_for_summary(chunk_responses).json()
            elif self.task == LanguageModelTask.KEYWORDS:
                return self._aggregate_results_for_keywords(chunk_responses).json()

    def _deserialize_response(self, response: str) -> Union[SummaryOutputClass, KeywordsOutputClass]:
        if self.task == LanguageModelTask.SUMMARY:
            return SummaryOutputClass.parse_raw(response)
        elif self.task == LanguageModelTask.KEYWORDS:
            return KeywordsOutputClass.parse_raw(response)

    def _construct_input_from_list_and_run_reduce_chain(self, response_list: List[InputData]) -> str:
        aggregate_input = Document(
            page_content=AggregateOutputClass(summaries=[
                self._deserialize_response(response) for response in response_list]
            ).json()
        )

        return self.reduce_llm_chain.run(aggregate_input.text)

    @staticmethod
    def _aggregate_results_for_summary(chunk_responses: List[Document]) -> SummaryOutputClass:
        return SummaryOutputClass(summary=" ".join([
            SummaryOutputClass.parse_raw(response_json).summary for response_json in chunk_responses
        ]))

    @staticmethod
    def _aggregate_results_for_keywords(chunk_responses: List[str]) -> KeywordsOutputClass:
        return KeywordsOutputClass(keywords=list(reduce(
            lambda x, y: x + y,
            [KeywordsOutputClass.parse_raw(response_json).keywords for response_json in chunk_responses],
            []
        )))


def load_long_text_processing_chain(
        task: LanguageModelTask,
        llm: BaseLanguageModel,
        model_total_max_tokens: int,
        max_output_tokens: int,
        map_prompt: BasePromptTemplate,
        reduce_prompt: BasePromptTemplate,
        aggregation_strategy: AggregationLogicForLongInputData,
        input_data_variable_name: str = "text",
        verbose: Optional[bool] = None
) -> LongTextProcessingChain:
    map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=verbose)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=verbose)

    return LongTextProcessingChain(
        task=task,
        model_total_max_tokens=model_total_max_tokens,
        max_output_tokens=max_output_tokens,
        map_llm_chain=map_chain,
        reduce_llm_chain=reduce_chain,
        input_data_variable_name=input_data_variable_name,
        aggregation_strategy=aggregation_strategy,
        verbose=verbose
    )
