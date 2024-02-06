from dataclasses import dataclass


@dataclass
class AzureGptTurboDefaults:
    OPENAI_API_TYPE: str = "azure"
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"

    # These values were chosen based on the default values used by the LLM provider
    MODEL_TOTAL_MAX_TOKENS: int = 4096
    TEMPERATURE = 0.0
    MAX_OUTPUT_TOKENS: int = 512
    REQUEST_TIMEOUT_S = 60


@dataclass
class AzureLlama2Defaults:
    MODEL_TOTAL_MAX_TOKENS: int = 4096
    MAX_OUTPUT_TOKENS: int = 512
    TEMPERATURE = 0.0
    TOP_P = 1.0


@dataclass
class AzureMistralAIDefaults:
    MODEL_TOTAL_MAX_TOKENS: int = 8192
    MAX_OUTPUT_TOKENS: int = 1024
    TEMPERATURE = 0.0
    TOP_P = 1.0
