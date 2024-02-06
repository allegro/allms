class LongTextChainDefaults:
    OVERLAP_SIZE = 50
    AGGREGATOR_DEFAULT_STR_SEPARATOR = ", "

    # TODO Refactor along with adding support for LongDocument processing
    AGGREGATION_PROMPT = """You're an AI agent that combines product summaries. Write a summary of the provided summaries. Keep the most important information and discard
            redundant information. The input is a JSON object.

            {output_data_model}

            {text}
        """
