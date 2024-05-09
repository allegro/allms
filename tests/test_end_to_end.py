import re

from unittest.mock import patch
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate

from allms.constants.input_data import IODataConstants
from allms.domain.configuration import VertexAIConfiguration
from allms.domain.prompt_dto import KeywordsOutputClass
from allms.models.vertexai_gemini import VertexAIGeminiModel
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
        
    
    def test_prompt_is_not_modified_for_open_source_models(self, mock_aioresponse, models, mocker):
        # GIVEN
        open_source_models = ["azure_llama2", "azure_mistral", "vertex_gemma"]

        mock_aioresponse.post(
            url=re.compile(f"^https:\/\/dummy-endpoint.*$"),
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
        prompt_template_spy = mocker.spy(ChatPromptTemplate, "from_messages")

        # WHEN & THEN
        for model_name, model in models.items():
            model.generate(
                prompt=prompt_template_text,
                input_data=input_data,
                output_data_model_class=KeywordsOutputClass,
                system_prompt=None if model_name == "azure_mistral" else "This is a system prompt."
            )

            if model_name in open_source_models:
                messages = [
                    HumanMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=["text"], 
                            template=prompt_template_text
                        )
                    )
                ]
                if model_name != "azure_mistral":
                    messages = [
                        SystemMessagePromptTemplate(
                            prompt=PromptTemplate(
                                input_variables=[], 
                                template="This is a system prompt."
                            )
                        )
                    ] + messages
                prompt_template_spy.assert_called_with(messages)
            else:
                prompt_template_spy.assert_called_with([
                    SystemMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=[], 
                            template="This is a system prompt."
                        )
                    ), 
                    HumanMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=["text"], 
                            partial_variables={
                                'output_data_model': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"keywords": {"title": "Keywords", "description": "List of keywords", "type": "array", "items": {"type": "string"}}}, "required": ["keywords"]}\n```'
                            }, 
                            template=f"{prompt_template_text}\n\n{{output_data_model}}"
                        )
                    )
                ])

    def test_gemini_version_is_passed_to_model(self):
        # GIVEN
        model_config = VertexAIConfiguration(
                cloud_project="dummy-project-id",
                cloud_location="us-central1",
                gemini_model_name="gemini-model-name"
            )
        
        # WHEN
        gemini_model = VertexAIGeminiModel(config=model_config)

        # WHEN
        gemini_model._llm.model_name == "gemini-model-name"

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



