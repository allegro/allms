import json
from unittest.mock import patch

from langchain.schema import OutputParserException

from llm_wrapper.domain.input_data import InputData
from llm_wrapper.domain.prompt_dto import SummaryOutputClass, KeywordsOutputClass


class TestOutputModelParserForDifferentModelOutputs:
    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_output_parser_returns_desired_format(self, tokens_mock, chain_run_mock, models):
        # GIVEN
        text_output = "This is the model output"
        expected_model_response = json.dumps({"summary": text_output})
        chain_run_mock.return_value = expected_model_response
        tokens_mock.return_value = 1

        input_data = [InputData(input_mappings={"text": "Some dummy text"}, id="1")]
        prompt = "Some Dummy Prompt {text}"

        # WHEN & THEN
        for model in models.values():
            model_response = model.generate(prompt, input_data, SummaryOutputClass)
            assert type(model_response[0].response) == SummaryOutputClass
            assert model_response[0].response.summary == text_output

    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_output_parser_returns_error_when_model_output_returns_different_field(self, tokens_mock, chain_run_mock, models):
        # GIVEN
        text_output = "This is the model output"
        expected_model_response = json.dumps({"other_key": text_output})
        chain_run_mock.return_value = expected_model_response
        tokens_mock.return_value = 1

        input_data = [InputData(input_mappings={"text": "Some dummy text"}, id="1")]
        prompt = "Some Dummy Prompt {text}"

        # WHEN & THEN
        for model in models.values():
            model_response = model.generate(prompt, input_data, SummaryOutputClass)
            assert type(model_response[0].error) == OutputParserException
            assert model_response[0].response is None

    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_output_parser_returns_parsed_class_when_model_output_returns_too_many_fields(self, tokens_mock, chain_run_mock, models):
        # GIVEN
        text_output = "This is the model output"
        expected_model_response = json.dumps({"other_key": text_output, "summary": text_output})
        chain_run_mock.return_value = expected_model_response
        tokens_mock.return_value = 1

        input_data = [InputData(input_mappings={"text": "Some dummy text"}, id="1")]
        prompt = "Some Dummy Prompt {text}"

        # WHEN & THEN
        for model in models.values():
            model_response = model.generate(prompt, input_data, SummaryOutputClass)
            assert type(model_response[0].response) == SummaryOutputClass
            assert model_response[0].response.summary == text_output

    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_model_returns_output_as_python_list_correctly(self, tokens_mock, chain_run_mock, models):
        # GIVEN
        text_output = [1, 2, 3]
        expected_model_response = json.dumps({"text": text_output, "keywords": text_output})
        chain_run_mock.return_value = expected_model_response
        tokens_mock.return_value = 1

        input_data = [InputData(input_mappings={"text": "Some dummy text"}, id="1")]
        prompt = "Some Dummy Prompt {text}"

        # WHEN & THEN
        for model in models.values():
            model_response = model.generate(prompt, input_data, KeywordsOutputClass)
            assert type(model_response[0].response) == KeywordsOutputClass
            assert model_response[0].response.keywords == list(map(str, text_output))

    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_model_output_when_input_data_is_empty(self, tokens_mock, chain_run_mock, models):
        # GIVEN
        expected_model_response = "2+2 is 4"
        chain_run_mock.return_value = expected_model_response
        tokens_mock.return_value = 1

        prompt = "2+2 is..."

        # WHEN & THEN
        for model in models.values():
            model_response = model.generate(prompt, None, KeywordsOutputClass)
            assert model_response[0].response is None
            assert type(model_response[0].error) == OutputParserException
