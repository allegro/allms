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
    OPENAI_GPT_35_TURBO = "GPT-3.5-TURBO"
    GOOGLE_PALM = "PALM"
    GOOGLE_GEMINI = "GEMINI"
    AZURE_LLAMA2 = "LLAMA2"


class LanguageModelTask(str, ListConvertableEnum):
    SUMMARY = "SUMMARY"
    KEYWORDS = "KEYWORDS"
