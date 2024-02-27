import re

import pandas as pd

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

        input_data = io_utils.load_data(
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
            output_data_model_class=KeywordsOutputClass
        )
        parsed_responses = sorted(parsed_responses, key=lambda key: key.input_data.id)

        # THEN
        expected_output = pd.read_csv("./tests/resources/test_end_to_end_expected_output.csv")
        expected_output = expected_output.astype({"id": "str", "text": "str"})
        expected_output = expected_output.sort_values(by="id").reset_index(drop=True)
        expected_output["response"] = expected_output["response"].apply(lambda x: eval(x))

        assert expected_output["id"].values.tolist() == list(
            map(lambda example: example.input_data.id, parsed_responses))

        assert expected_output["text"].values.tolist() == list(
            map(lambda example: example.input_data.input_mappings["text"], parsed_responses))

        assert expected_output["response"].values.tolist() == list(
            map(lambda example: example.response.keywords, parsed_responses))

        assert expected_output["number_of_prompt_tokens"].values.tolist() == list(
            map(lambda example: example.number_of_prompt_tokens, parsed_responses))

        assert expected_output["number_of_generated_tokens"].values.tolist() == list(
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
        
        #THEN
        assert responses[0].response is None
        assert "Request timed out" in responses[0].error

    

