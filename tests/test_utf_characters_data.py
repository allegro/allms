import html.entities
from unittest.mock import patch

import pytest


# class TestModelBehaviorForSpecialCharacters:
#     @patch("langchain.chains.base.Chain.arun")
#     @patch("langchain_community.llms.vertexai.VertexAI.get_num_tokens")
#     @pytest.mark.parametrize("input_character", list(html.entities.entitydefs.values()))
#     def test_model_is_not_broken_by_special_characters(self, tokens_mock, arun_mock, input_character, models):
#         # GIVEN
#         print(tokens_mock)
#         arun_mock.return_value = f"{input_character}"
#         tokens_mock.return_value = 1
#
#         # WHEN & THEN
#         for model in models.values():
#             response = model.generate(
#                 f"This is prompt with broken sign {input_character} and the model should work.")
#             assert response[0].error is None
#             assert response[0].response == input_character
