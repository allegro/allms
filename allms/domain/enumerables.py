from enum import Enum
from typing import List


class ListConvertableEnum(Enum):
    @classmethod
    def get_values(cls) -> List[str]:
        return list(map(lambda field: field.value, cls))


class AggregationLogicForLongInputData(str, ListConvertableEnum):
    SIMPLE_CONCATENATION = "SIMPLE_CONCATENATION"
    REDUCE_BY_LLM_PROMPTING = "REDUCE_BY_LLM_PROMPTING"


class AvailableModels(str, ListConvertableEnum):
    AZURE_OPENAI_MODEL = "azure_openai"
    AZURE_LLAMA2_MODEL = "azure_llama2"
    AZURE_MISTRAL_MODEL = "azure_mistral"
    VERTEXAI_PALM2_MODEL = "vertexai_palm2"
    VERTEXAI_GEMINI_MODEL = "vertexai_gemini"
    VERTEXAI_GEMMA_MODEL = "vertexai_gemma"


class LanguageModelTask(str, ListConvertableEnum):
    SUMMARY = "SUMMARY"
    KEYWORDS = "KEYWORDS"
