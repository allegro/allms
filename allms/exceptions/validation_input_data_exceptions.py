def get_missing_input_data_in_prompt_message(example_id: str) -> str:
    return f"Missing input_keys in the prompt. Error occurred for id={example_id}"


def get_missing_input_data_in_input_data_message(example_id: str) -> str:
    return f"Missing input_keys in the input data. Error occurred for id={example_id}"


def get_different_number_of_inputs_message(example_id: str) -> str:
    return (f"Number of input keys in input_data and prompt are different."
            f"If your intention is to instruct the model to output a JSON, make sure you are using double curly brackets."
            f" Please make sure the input_keys are consistent."
            f" Error has occurred for id={example_id}")


def get_different_input_keys_message(example_id: str) -> str:
    return (f"Input variables in the prompt and in the input_data are different. Please make sure"
            f"the input_keys are consistent. "
            f"If your intention is to instruct the model to output a JSON, make sure you are using double curly brackets."
            f"Error has occurred for id={example_id}")


def get_prompt_contains_input_key_when_missing_input_data() -> str:
    return f"When no input_data is provided prompt cannot contain any input_key."


def get_system_prompt_contains_input_variables() -> str:
    return "System prompt cannot contain any input variables. Please fix your system message and try again."


def get_system_prompt_is_not_supported_by_model() -> str:
    return "Mistral-based models don't support `system_prompt` parameter."
