class PalmModelDefaults:
    # These values were chosen based on the default values used by the LLM provider
    GCP_MODEL_NAME = "text-bison@001"
    MODEL_TOTAL_MAX_TOKENS = 8192
    MAX_OUTPUT_TOKENS = 1024
    TEMPERATURE = 0.0
    TOP_P = 0.95
    TOP_K = 40
    VERBOSE = True


class GeminiModelDefaults:
    GCP_MODEL_NAME = "gemini-pro"
    MODEL_TOTAL_MAX_TOKENS = 30720
    MAX_OUTPUT_TOKENS = 2048
    TEMPERATURE = 0.0
    TOP_P = 0.95
    TOP_K = 40
    VERBOSE = True


class GemmaModelDefaults:
    GCP_MODEL_NAME = "gemma"
    MODEL_TOTAL_MAX_TOKENS = 8192
    MAX_OUTPUT_TOKENS = 1024
    TEMPERATURE = 0.0
    TOP_P = 0.95
    TOP_K = 40
    VERBOSE = True
