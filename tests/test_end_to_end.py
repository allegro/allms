import re

from allms.constants.input_data import IODataConstants
from allms.domain.prompt_dto import KeywordsOutputClass
from allms.utils import io_utils
from tests.conftest import AzureOpenAIEnv


class TestEndToEnd:

    def test_model_is_queried_successfully(
            self,
            mock_aioresponse,
            models
    ):
        # GIVEN
        mock_aioresponse.post(
            url=re.compile(f"^{AzureOpenAIEnv.OPENAI_API_BASE}.*$"),
            payload={
                "choices": [{
                    "message": {
                        "content": "{\"keywords\": [\"Indywidualna racja żywnościowa\", \"wojskowa\", \"S-R-9\", \"set nr 9\", \"Makaron po bolońsku\", \"Konserwa tyrolska\", \"Suchary\", \"Koncentrat napoju herbacianego instant o smaku owoców leśnych\", \"Dżem malinowy\", \"Baton zbożowo-owocowy o smaku figowym\"]}",
                        "role": ""
                    }
                }],
                "usage": {}
            },
            repeat=True
        )

        input_data = io_utils.load_csv_to_input_data(
            limit=5,
            path="./tests/resources/test_input_data.csv"
        )
        prompt_template_text = """Extract at most 10 keywords that could be used as features in a search index from this Polish product description.

        {text}
        """

        # WHEN
        parsed_responses = models["azure_open_ai"].generate(
            prompt=prompt_template_text,
            input_data=input_data,
            output_data_model_class=KeywordsOutputClass,
            system_prompt="This is a system prompt."
        )
        parsed_responses = sorted(parsed_responses, key=lambda key: key.input_data.id)

        # THEN
        expected_output = io_utils.load_csv("./tests/resources/test_end_to_end_expected_output.csv")
        expected_output = sorted(expected_output, key=lambda example: example[IODataConstants.ID])
        for idx in range(len(expected_output)):
            expected_output[idx]["response"] = eval(expected_output[idx]["response"])

        assert list(map(lambda output: output[IODataConstants.ID], expected_output)) == list(
            map(lambda example: example.input_data.id, parsed_responses))

        assert list(map(lambda output: output[IODataConstants.TEXT], expected_output)) == list(
            map(lambda example: example.input_data.input_mappings["text"], parsed_responses))

        assert list(map(lambda output: output[IODataConstants.RESPONSE_STR_NAME], expected_output)) == list(
            map(lambda example: example.response.keywords, parsed_responses))

        assert list(map(lambda output: int(output[IODataConstants.PROMPT_TOKENS_NUMBER]), expected_output)) == list(
            map(lambda example: example.number_of_prompt_tokens, parsed_responses))

        assert list(map(lambda output: int(output[IODataConstants.GENERATED_TOKENS_NUMBER]), expected_output)) == list(
            map(lambda example: example.number_of_generated_tokens, parsed_responses))

    def test_model_times_out(
            self,
            mock_aioresponse,
            models
    ):
        # GIVEN
        mock_aioresponse.post(
            url=re.compile(f"^{AzureOpenAIEnv.OPENAI_API_BASE}.*$"),
            exception=TimeoutError("Request timed out!"),
            repeat=True
        )

        # WHEN
        responses = models["azure_open_ai"].generate("Some prompt")

        # THEN
        assert responses[0].response is None
        assert "Request timed out" in responses[0].error



