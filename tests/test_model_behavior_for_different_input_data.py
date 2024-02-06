import logging

import llm_wrapper.exceptions.validation_input_data_exceptions as input_validation_messages
from unittest.mock import patch

import pytest

from llm_wrapper.domain.input_data import InputData
from llm_wrapper.domain.response import ResponseData


class TestModelBehaviorForDifferentInput:

    @patch("langchain.chains.base.Chain.arun")
    def test_no_input_variables_provided_in_the_prompt_raise_exception(self, chain_run_mock, models):
        for model in models.values():
            chain_run_mock.return_value = "{}"

            input_data = [InputData(input_mappings={"text": "Some dummy text"}, id="1")]

            prompt = "Some Dummy Prompt without input variable"

            with pytest.raises(ValueError, match=input_validation_messages.get_missing_input_data_in_prompt_message(
                    input_data[0].id)) as expected_value_exception:
                model.generate(prompt, input_data)

    @patch("langchain.chains.base.Chain.arun")
    def test_no_input_variables_provided_in_the_input_data_raise_exception(self, chain_run_mock, models):
        for model in models.values():
            chain_run_mock.return_value = "{}"

            input_data = [InputData(input_mappings={}, id="1")]

            prompt = "Some Dummy Prompt without input variable :{text}"

            with pytest.raises(ValueError, match=input_validation_messages.get_missing_input_data_in_input_data_message(
                    input_data[0].id)) as expected_value_exception:
                model.generate(prompt, input_data)

    @patch("langchain.chains.base.Chain.arun")
    def test_different_input_keys_provided_in_input_data_and_prompt(self, chain_run_mock, models):
        for model in models.values():
            chain_run_mock.return_value = "{}"

            input_data = [InputData(input_mappings={"text": "Some dummy text", "text_2": "Another dummy text"}, id="1")]

            prompt = "Some Dummy Prompt without input variable {text} {text_1}"

            with pytest.raises(ValueError, match=input_validation_messages.get_different_input_keys_message(
                    input_data[0].id)) as expected_value_exception:
                model.generate(prompt, input_data)

    @patch("langchain.chains.base.Chain.arun")
    def test_different_number_of_input_keys_provided_in_input_data_and_prompt(self, chain_run_mock, models):
        for model in models.values():
            chain_run_mock.return_value = "{}"

            input_data = [InputData(input_mappings={"text": "Some dummy text", "text_2": "Another dummy text"}, id="1")]

            prompt = "Some Dummy Prompt without input variable {text} {text_1} {text_2}"

            with pytest.raises(ValueError, match=input_validation_messages.get_different_number_of_inputs_message(
                    input_data[0].id)) as expected_value_exception:
                model.generate(prompt, input_data)

    @patch("langchain.chains.base.Chain.arun")
    def test_exception_when_input_data_is_missing_and_prompt_contains_input_key(self, chain_run_mock, models):
        for model in models.values():
            chain_run_mock.return_value = "{}"

            prompt = "Some Dummy Prompt without input variable {text} {text_1}"

            with pytest.raises(
                    ValueError,
                    match=input_validation_messages.get_prompt_contains_input_key_when_missing_input_data()
            ):
                model.generate(prompt, None)

    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_exception_when_num_prompt_tokens_larger_than_model_total_max_tokens(self, tokens_mock, chain_run_mock, models):
        # GIVEN
        chain_run_mock.return_value = "{}"

        input_data = [InputData(input_mappings={"text": "Some dummy text", "text_2": "Another dummy text"}, id="1")]

        prompt = "Some dummy really, really long prompt. " * 10000 + "input variables: {text} {text_2}"
        tokens_mock.return_value = len(prompt.split())

        # WHEN & THEN
        for model in models.values():
            response = model.generate(prompt, input_data)[0]

            assert isinstance(response, ResponseData)
            assert response.response is None
            assert "Value Error has occurred: Prompt is too long" in response.error

    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_whether_curly_brackets_are_not_breaking_the_prompt(self, tokens_mock, chain_run_mock, models):
        # GIVEN
        chain_run_mock.return_value = "{}"

        input_data = [InputData(input_mappings={"text": "Some dummy text"}, id="1")]

        prompt = "Extract parameters from this text: {text} and output them as a JSON: {{name: parameter_name, value: parameter_value}}"
        tokens_mock.return_value = len(prompt.split())

        # WHEN & THEN
        for model in models.values():
            response = model.generate(prompt, input_data)[0]

            assert isinstance(response, ResponseData)
            assert response.response is not None

    @patch("langchain.chains.base.Chain.arun")
    @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
    def test_warning_when_num_prompt_tokens_plus_max_output_tokens_larger_than_model_total_max_tokens(
            self,
            tokens_mock,
            chain_run_mock,
            models,
            caplog
    ):
        # GIVEN
        chain_run_mock.return_value = "{}"

        input_data = [InputData(input_mappings={"text": "Some dummy text", "text_2": "Another dummy text"}, id="1")]

        prompt = "Some dummy prompt. input variables: {text} {text_2}"
        tokens_mock.return_value = len(prompt.split())

        # WHEN & THEN
        for model in models.values():
            model._max_output_tokens = 100000

            with caplog.at_level(logging.WARNING):
                model.generate(prompt, input_data)

            log_records = caplog.records
            assert len(log_records) == 1
            assert log_records[0].levelname == "WARNING"
            assert "Number of prompt tokens plus generated tokens may exceed the the max allowed number of tokens of the model." in log_records[0].message

            caplog.clear()
