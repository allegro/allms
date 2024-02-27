from typing import List

from langchain import BasePromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.schema import Document

from allms.defaults.long_text_chain import LongTextChainDefaults


def truncate_text_to_max_size(
        llm: BaseLanguageModel,
        prompt_template: BasePromptTemplate,
        text: str,
        model_total_max_tokens: int,
        max_output_tokens: int,
) -> str:
    """
    This function is supposed to truncate the input to fit the maximum context size of a model. The problem is that
    the max context size is in tokens and in our code we operate on raw, un-tokenized strings. We can only calculate
    how many tokens given string has. So to find the point on which we should truncate, this function calculates in
    tokens how many times the current `text` is longer than the allowed limit. Then it assumes that the ration is true
    also when reasoning on words instead of tokens. And based on this the `split_point_index` is calculated.
    This is only an approximation (ration calculated on tokens is only similar to the ratio calculated on words, but in
    most of the cases it won't be the same). That's why this function is used in a recursive way. It calculates the
    split point, truncates the text and checks again if the total prompt length is lower than the max context size of
    a model. If not it reruns itself again and if yes, then it returns the truncated text.

    Another possibility would be to implement this function using tokenizer to tokenize text to tokens, then truncate
    the text, detokenize it to string and return truncated text. But for this solution, first we'd need to have a list
    of tokenizers used by every model we'd like to support (now it's provided inside langchain) and second, the
    tokenization and de-tokenization steps could change the input prompt by introducing some artifacts.
    """
    max_token_limit = get_max_allowed_number_of_tokens(model_total_max_tokens, max_output_tokens)
    num_tokens = int(llm.get_num_tokens(prompt_template.format(text=text)))

    if num_tokens <= max_token_limit:
        return text

    # We add `text="text"` and not empty string, because the empty string may be tokenized together with the whitespaces
    # that are around it in the prompt. But when joining the actual `{text}` with the prompt instructions we get one
    # additional token
    num_tokens_prompt_wo_text = int(llm.get_num_tokens(prompt_template.format(text="text")))
    num_tokens_text = int(llm.get_num_tokens(text))
    num_tokens_left_for_text = max_token_limit - num_tokens_prompt_wo_text
    if num_tokens_left_for_text <= 0:
        raise ValueError("Prompt instruction (without the actual text) is longer than the allowed model input length")

    # How many times the current text is longer than the allowed length
    current_to_allowed_length_ration = num_tokens_text / num_tokens_left_for_text
    words = text.split()
    split_point_index = int(len(words) / current_to_allowed_length_ration)

    text_truncated = " ".join(words[:split_point_index])

    return truncate_text_to_max_size(
        llm=llm,
        prompt_template=prompt_template,
        text=text_truncated,
        model_total_max_tokens=model_total_max_tokens,
        max_output_tokens=max_output_tokens
    )


def split_text_to_max_size(
        llm: BaseLanguageModel,
        prompt_template: BasePromptTemplate,
        text: str,
        model_total_max_tokens: int,
        max_output_tokens: int,
        overlap_size: int = LongTextChainDefaults.OVERLAP_SIZE
) -> List[Document]:
    max_token_limit = get_max_allowed_number_of_tokens(model_total_max_tokens, max_output_tokens)
    if int(llm.get_num_tokens(prompt_template.format(text=text))) < max_token_limit:
        return [Document(page_content=text)]

    words = text.split()
    middle_word_index = len(words) // 2

    overlap_left = overlap_size // 2
    overlap_right = overlap_size - overlap_left
    data_left_half = " ".join(words[:middle_word_index + overlap_left])
    data_right_half = " ".join(words[middle_word_index - overlap_right:])

    return (
            split_text_to_max_size(llm=llm, prompt_template=prompt_template, text=data_left_half,
                                   model_total_max_tokens=model_total_max_tokens, max_output_tokens=max_output_tokens)
            + split_text_to_max_size(llm=llm, prompt_template=prompt_template, text=data_right_half,
                                     model_total_max_tokens=model_total_max_tokens, max_output_tokens=max_output_tokens)
    )


def get_max_allowed_number_of_tokens(model_total_max_tokens: int, max_output_tokens: int) -> int:
    buffer = 50  # for things like BOS, EOS and other unexpected things
    return model_total_max_tokens - max_output_tokens - buffer
